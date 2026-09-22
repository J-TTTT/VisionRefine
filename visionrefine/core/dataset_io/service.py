"""Persistence, revision selection and packaging, independent of file formats."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

from . import registry
from .common import dump_json
from .models import Annotation, Category, Dataset, ImageRecord, contained_path, safe_relative_path
from .portable import file_digest, portable_dataset, redact


def prepare_import(root: Path, format_id: str, source: Path | None, labels: list[str], task: str, split: str, *, trust_reviewed=False) -> Dataset:
    adapter = registry.get(format_id, task, "importer")
    dataset = adapter.importer.read(root.resolve(), source, labels, split, **({"trust_reviewed": trust_reviewed} if format_id == "visionrefine" else {}))
    dataset.task = task
    dataset.provenance = {**dataset.provenance,
        "format": format_id, "imported_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source) if source else None,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source and source.is_file() else None,
    }
    return dataset


def persist_import(store, project: dict, dataset: Dataset, revision: str | None = None) -> dict:
    for image in dataset.images:
        if not image.sha256:
            _, path = image_location(project, image, dataset)
            image.sha256 = file_digest(path)
    revision = revision or f"import-{uuid.uuid4().hex}"
    dataset.provenance["parent_revision_id"] = project.get("dataset_revision")
    dataset.provenance["revision_id"] = revision
    destination = store.root / project["id"] / "datasets" / f"{revision}.json"
    dump_json(destination, dataset.model_dump())
    project["dataset_revision"] = revision
    project["dataset_format"] = dataset.provenance["format"]
    project["labels"] = [c.name for c in dataset.categories]
    project["annotation_path"] = dataset.provenance.get("source_file")
    project["import_summary"] = dataset.report.model_dump(exclude={"issues"})
    project["import_summary"]["issue_count"] = len(dataset.report.issues)
    # Imported categories are referenced by historical annotations.
    if project["dataset_format"] != "images" and (not dataset.sources or any(s.format != "images" for s in dataset.sources)):
        project["labels_locked"] = True
    project["analysis"] = None
    store.save(project)
    return project


def load_dataset(store, project: dict) -> Dataset | None:
    revision = project.get("dataset_revision")
    if not revision:
        return None
    if not re.fullmatch(r"import-[0-9a-f]{32}", revision):
        raise ValueError("Invalid dataset revision")
    path = store.root / project["id"] / "datasets" / f"{revision}.json"
    return Dataset.model_validate_json(path.read_text(encoding="utf-8"))


def image_location(project: dict, record: ImageRecord, dataset: Dataset) -> tuple[Path, Path]:
    """Logical image paths never double as physical paths in a multi-source dataset."""
    if record.source_id:
        source = next(s for s in dataset.sources if s.id == record.source_id)
        root = Path(source.root).resolve()
        return root, contained_path(root, record.source_path)
    root = Path(project["dataset_path"]).resolve()
    return root, contained_path(root, record.path)


def document_path(store, project_id: str, image: str, kind: str) -> Path:
    digest = hashlib.sha256(safe_relative_path(image).encode("utf-8")).hexdigest()[:24]
    return store.root / project_id / kind / f"{digest}.json"


def effective_annotation(store, project: dict, image: str, dataset: Dataset | None = None,
                         imported_record: ImageRecord | None = None) -> dict:
    image = safe_relative_path(image)
    # An explicitly reviewed empty image must take precedence over every proposal.
    for folder, status in (("annotations", "human_reviewed"), ("suggestions", "ai_suggestion")):
        path = document_path(store, project["id"], image, folder)
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            return {**document, "image": image, "status": status}
    latest = project.get("latest_suggestion") or {}
    if latest.get("image") == image and re.fullmatch(r"pilot-[0-9TZ]+", latest.get("revision_id", "")):
        path = store.root / project["id"] / "revisions" / latest["revision_id"] / "suggestion.json"
        if path.is_file():
            suggestion = json.loads(path.read_text(encoding="utf-8"))
            return dict(image=image, objects=suggestion.get("objects", []), status="ai_suggestion", revision_id=latest["revision_id"])
    if dataset is None:
        dataset = load_dataset(store, project)
    if dataset:
        record = imported_record or next((i for i in dataset.images if i.path == image), None)
        if record:
            return dict(image=image, objects=[o.model_dump() for o in record.objects], status=record.status,
                        revision_id=dataset.provenance.get("revision_id"), split=record.split)
    return dict(image=image, objects=[], status="unreviewed", revision_id=None)


def select_snapshot(store, project: dict, policy: str, splits: list[str]):
    accepted = {
        "reviewed": {"human_reviewed", "final_reviewed"},
        "reviewed_or_ai": {"human_reviewed", "final_reviewed", "ai_suggestion"},
        "all_annotated": {"human_reviewed", "final_reviewed", "ai_suggestion", "imported_coarse"},
    }
    if policy not in accepted:
        raise ValueError("Unknown export revision policy")
    dataset = load_dataset(store, project)
    if dataset is None:
        # Legacy projects gain a complete image manifest on their next analysis.
        raise ValueError("Analyze the dataset before exporting")
    selected = dataset.model_copy(deep=True)
    previous = {c.name: c for c in dataset.categories}
    selected.categories = [Category(id=i, name=label, source_id=previous[label].source_id if label in previous else None,
                                   provenance=previous[label].provenance if label in previous else {})
                           for i, label in enumerate(project["labels"])]
    categories = {c.name: c for c in selected.categories}
    selected.images = []
    skipped = []
    selected_revisions = []
    for record in dataset.images:
        if splits and record.split not in splits:
            continue
        annotation = effective_annotation(store, project, record.path, dataset, record)
        if annotation["status"] not in accepted[policy]:
            skipped.append(dict(image=record.path, status=annotation["status"]))
            continue
        image = record.model_copy(deep=True)
        image.status = annotation["status"]
        image.objects = []
        for index, obj in enumerate(annotation["objects"]):
            category = categories.get(obj.get("label"))
            if category is None:
                raise ValueError(f"Unknown label in {record.path}: {obj.get('label')}")
            image.objects.append(Annotation.model_validate({**obj, "id": str(obj.get("id") or f"object-{index}"),
                "category_id": category.id, "source": annotation["status"]}))
        image.provenance["revision_id"] = annotation.get("revision_id")
        selected.images.append(image)
        selected_revisions.append(dict(image=record.path, status=image.status, revision_id=annotation.get("revision_id")))
    if not selected.images:
        raise ValueError("No images match the chosen splits and review policy; save a human review or include AI/imported annotations")
    selected = Dataset.model_validate(selected.model_dump())
    selected.report.image_count = len(selected.images)
    selected.report.object_count = sum(len(i.objects) for i in selected.images)
    selected.report.empty_images = sum(not i.objects for i in selected.images)
    return selected, dict(skipped_images=skipped, revisions=selected_revisions)


def export_dataset(store, project: dict, format_id: str, policy: str, splits: list[str], include_images: bool) -> dict:
    """Legacy synchronous API, retaining warning-only behavior for compatibility."""
    registry.get(format_id, project["task"], "exporter")
    selected, selection = select_snapshot(store, project, policy, splits)
    return package_snapshot(store, project, selected, format_id, policy, include_images, selection)


def package_snapshot(store, project, selected, format_id, policy, include_images, selection, *, checkpoint=lambda: None, progress=lambda *args: None):
    adapter = registry.get(format_id, selected.task, "exporter")
    source_files = {}
    for record in selected.images:
        checkpoint()
        _, source = image_location(project, record, selected)
        checksum = file_digest(source)
        if record.sha256 and checksum != record.sha256:
            raise ValueError(f"Source image changed since snapshot: {record.path}")
        record.sha256 = checksum
        source_files[record.path] = (source, checksum)
    export_id = f"export-{uuid.uuid4().hex}"
    output = store.root / project["id"] / "exports" / export_id
    output.mkdir(parents=True, exist_ok=False)
    public = portable_dataset(selected, {i.path: i.path for i in selected.images})
    metadata = adapter.exporter.write(public, output)
    if include_images:
        for index, (original, relative) in enumerate(metadata["image_paths"].items()):
            checkpoint()
            source, expected = source_files[original]
            target = contained_path(output, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if metadata.get("image_encoding") == "jpeg":
                with Image.open(source) as pixels:
                    if pixels.format == "JPEG":
                        shutil.copyfile(source, target)
                    else:
                        pixels.convert("RGB").save(target, format="JPEG", quality=95)
            else:
                with source.open("rb") as incoming, target.open("wb") as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        checkpoint()
                        outgoing.write(block)
            if file_digest(source) != expected or (metadata.get("image_encoding") != "jpeg" and file_digest(target) != expected):
                raise ValueError(f"Source image changed while packaging: {original}")
            progress(index + 1, len(source_files), "images")
    else:
        metadata["warnings"].append("Images are omitted. Copy the source files to the image_paths destinations before standalone use.")
        if metadata.get("image_encoding") == "jpeg":
            metadata["warnings"].append("VOC destinations must contain JPEG-encoded images; convert non-JPEG originals, do not simply rename them.")
    if policy != "reviewed":
        metadata["warnings"].append("This export may include unreviewed AI proposals or imported coarse annotations.")
    if selected.report.issues:
        metadata["warnings"].append("The original import had warnings/errors; consult the import report in visionrefine.json.")
    report = dict(export_id=export_id, format=format_id, task=selected.task, policy=policy,
                  created_at=datetime.now(timezone.utc).isoformat(), include_images=include_images,
                  image_count=len(selected.images), object_count=sum(len(i.objects) for i in selected.images),
                  **selection, **metadata)
    # Path/category mappings are semantic data, not extensible metadata. In
    # particular a label such as /m/01g317 must not be treated as a machine path.
    report["warnings"] = redact(report["warnings"])
    dump_json(output / "export-report.json", report)
    portable = portable_dataset(public, metadata["image_paths"])
    if include_images:
        for record in portable.images:
            record.sha256 = file_digest(contained_path(output, record.path))
    dump_json(output / "visionrefine.json", portable.model_dump())
    archive = output.with_suffix(".zip")
    # The downloadable archive appears only after all files are successfully written.
    pending = archive.with_suffix(".zip.part")
    with ZipFile(pending, "w", ZIP_DEFLATED) as zip_file:
        for file in sorted(output.rglob("*")):
            if file.is_file():
                checkpoint()
                with file.open("rb") as incoming, zip_file.open(file.relative_to(output).as_posix(), "w", force_zip64=True) as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        checkpoint()
                        outgoing.write(block)
    checkpoint()
    pending.replace(archive)
    return report
