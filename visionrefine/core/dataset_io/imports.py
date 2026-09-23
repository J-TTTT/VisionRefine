"""Preview/commit transactions and format-independent dataset merging.

Source files are read-only. A commit publishes one immutable manifest and never
writes human annotations, AI suggestions, or the original dataset directories.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .common import dump_json, finish
from .models import Category, Dataset, DatasetSource, Split, Task
from .service import image_location, load_dataset, persist_import, prepare_import

Policy = Literal["keep_existing", "update_coarse", "error"]


class StalePreview(ValueError):
    pass


class ImportSource(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    root: str = Field(min_length=1)
    format: str = "images"
    annotation_path: str | None = None
    split: Split = "unspecified"
    labels: list[str] = Field(default_factory=list, max_length=10000)
    category_mapping: dict[str, str] = Field(default_factory=dict)
    trust_reviewed: bool = False

    @field_validator("category_mapping")
    @classmethod
    def nonempty_targets(cls, mapping):
        if any(not name.strip() for name in mapping.values()):
            raise ValueError("Category mapping targets cannot be empty")
        return {key: value.strip() for key, value in mapping.items()}


class ImportPlanInput(BaseModel):
    project_id: str | None = None
    name: str = Field(default="Imported dataset", min_length=1, max_length=80)
    task: Task = "detection"
    model_max_side: int = Field(default=1536, ge=256, le=8192)
    sources: list[ImportSource] = Field(min_length=1, max_length=100)
    duplicate_policy: Policy = "keep_existing"
    conflict_resolutions: dict[str, Policy] = Field(default_factory=dict)


def preview_import(store, payload: ImportPlanInput) -> dict:
    project = store.get(payload.project_id) if payload.project_id else None
    task = project["task"] if project else payload.task
    original = load_dataset(store, project) if project else None
    if project and original is None:
        raise ValueError("Analyze the existing project once before appending a dataset")
    merged = original.model_copy(deep=True) if original else Dataset(task=task)
    merged.schema_version = "3.0" if task == "instance_segmentation" else "2.0"
    if project:
        # Unlocked image-only projects may have edited labels since analysis.
        prior = {c.name: c for c in merged.categories}
        merged.categories = [prior[name].model_copy(update={"id": i}) if name in prior else Category(id=i, name=name)
                             for i, name in enumerate(project["labels"])]
        if any(not image.source_id for image in merged.images):
            used = {s.id for s in merged.sources} | {s.id for s in payload.sources}
            legacy_id = "legacy"
            while legacy_id in used:
                legacy_id += "_old"
            merged.sources.append(DatasetSource(id=legacy_id, root=str(Path(project["dataset_path"]).resolve()),
                                  format=project.get("dataset_format", "images"), annotation_path=project.get("annotation_path")))
            for image in merged.images:
                if not image.source_id:
                    image.source_id, image.source_path = legacy_id, image.path
    if len({s.id for s in payload.sources}) != len(payload.sources):
        raise ValueError("Give each source row a unique source ID")
    descriptors = []
    for source in payload.sources:
        root = Path(source.root).expanduser().resolve()
        annotation = Path(source.annotation_path).expanduser().resolve() if source.annotation_path else None
        if source.format == "visionrefine" and annotation and annotation.suffix.lower() == ".zip":
            from .portable import extract_native
            extracted = store.root.parent / "cache" / "native-imports" / uuid.uuid4().hex
            extract_native(annotation, extracted)
            root, annotation = extracted, extracted / "dataset.json"
        if store.root.parent.resolve().is_relative_to(root):
            raise ValueError("A dataset source root must not contain the VisionRefine workspace directory")
        if not root.is_dir():
            raise ValueError(f"Source directory does not exist: {root}")
        if annotation:
            if not annotation.exists():
                raise ValueError(f"Annotation input does not exist: {annotation}")
        descriptors.append(DatasetSource(id=source.id, root=str(root), format=source.format,
                                        annotation_path=str(annotation) if annotation else None))
    by_path = {image.path: image for image in merged.images}
    catalog = {c.name: c for c in merged.categories}
    source_catalog = {s.id: s for s in merged.sources}
    summaries, conflicts = [], []
    added = updated = skipped = 0
    for source, descriptor in zip(payload.sources, descriptors):
        existing_source = source_catalog.get(source.id)
        if existing_source and existing_source.model_dump(exclude={"provenance"}) != descriptor.model_dump(exclude={"provenance"}):
            raise ValueError(f"Source ID {source.id} is already used for a different input; choose another ID")
        root = Path(descriptor.root)
        annotation = Path(descriptor.annotation_path) if descriptor.annotation_path else None
        labels = list(dict.fromkeys(label.strip() for label in source.labels if label.strip()))
        incoming = prepare_import(root, source.format, annotation, labels, task, source.split,
                                  trust_reviewed=source.trust_reviewed, fingerprint_source=False)
        unknown = set(source.category_mapping) - {c.name for c in incoming.categories}
        if unknown:
            raise ValueError(f"Unknown categories in mapping for {source.id}: {sorted(unknown)}")
        mappings = []
        category_map = {}
        for category in incoming.categories:
            target = source.category_mapping.get(category.name, category.name)
            if target not in catalog:
                canonical = category.model_copy(deep=True, update={
                    "id": max((c.id for c in merged.categories), default=-1) + 1, "name": target,
                })
                merged.categories.append(canonical)
                catalog[target] = canonical
            canonical = catalog[target]
            origin = dict(source=source.id, name=category.name, source_id=category.source_id)
            origins = canonical.provenance.setdefault("imports", [])
            if origin not in origins:
                origins.append(origin)
            category_map[category.id] = canonical
            mappings.append(dict(source_name=category.name, source_id=category.source_id,
                                 target_name=canonical.name, target_id=canonical.id))
        if not existing_source:
            merged.sources.append(descriptor)
            source_catalog[source.id] = descriptor
        for record in incoming.images:
            relative = record.path
            record.path = f"{source.id}/{relative}"
            record.id = record.path
            record.source_id, record.source_path = source.id, relative
            # The user owns source/image matching. Do not bind imported images
            # to file bytes or compare unrelated sources by content.
            record.sha256 = None
            record.provenance.update(import_source_id=source.id, original_path=relative)
            for obj in record.objects:
                original_category = obj.category_id
                obj.category_id = category_map[original_category].id
                obj.label = category_map[original_category].name
                obj.provenance.update(import_source_id=source.id, import_category_id=original_category)
            duplicate = by_path.get(record.path)
            if duplicate:
                action = payload.conflict_resolutions.get(record.path, payload.duplicate_policy)
                if action == "update_coarse" and record.status == "unreviewed":
                    action = "keep_existing"
                if action == "update_coarse" and duplicate.status in {"human_reviewed", "final_reviewed", "ai_suggestion"}:
                    action = "keep_existing"
                conflicts.append(dict(incoming=record.path, existing=duplicate.path, action=action,
                                      incoming_split=record.split, existing_split=duplicate.split,
                                      incoming_objects=len(record.objects), existing_objects=len(duplicate.objects),
                                      human_and_ai_preserved=True))
                if action == "update_coarse":
                    duplicate.objects = record.objects
                    duplicate.status = "imported_coarse"
                    for obj in duplicate.objects:
                        obj.source = "imported_coarse"
                    duplicate.provenance["coarse_update"] = record.provenance
                    updated += 1
                else:
                    skipped += 1
                # Keep the existing split for a repeated logical image path.
            else:
                merged.images.append(record)
                by_path[record.path] = record
                added += 1
        summaries.append(dict(id=source.id, format=source.format, root=str(root), mappings=mappings,
                              trust_reviewed=source.trust_reviewed if source.format == "visionrefine" else False,
                              report=incoming.report.model_dump()))
        merged.report.issues.extend(incoming.report.issues)
        merged.report.skipped_images += incoming.report.skipped_images
        merged.report.skipped_objects += incoming.report.skipped_objects
    if task in {"detection", "instance_segmentation", "grounding", "classification"} and not merged.categories:
        raise ValueError("Add at least one class for image-only datasets")
    merged = finish(merged)
    summary = dict(added_images=added, updated_coarse=updated, skipped_duplicates=skipped,
                   image_count=len(merged.images), object_count=merged.report.object_count,
                   category_count=len(merged.categories),
                   splits={s: sum(i.split == s for i in merged.images) for s in ("train", "val", "test", "unspecified")})
    preview_id = uuid.uuid4().hex
    revision = f"import-{uuid.uuid4().hex}"
    result = dict(preview_id=preview_id, project_id=payload.project_id, task=task, summary=summary,
                  sources=summaries, categories=[c.model_dump() for c in merged.categories], conflicts=conflicts,
                  commit_allowed=not any(c["action"] == "error" for c in conflicts),
                  warnings=["Only the same source ID and relative image path are treated as one image. Existing human/AI revisions are retained.",
                            "Only identical category names merge automatically; other renames require explicit mappings."])
    merged.provenance.update(format="mixed", source_file=None, source_sha256=None, imported_at=datetime.now(timezone.utc).isoformat(),
                             operation=dict(kind="append" if project else "create", preview_id=preview_id,
                                            summary=summary, sources=summaries, conflicts=conflicts))
    plan = dict(result=result, dataset=merged.model_dump(), input=payload.model_dump(),
                base_revision=project.get("dataset_revision") if project else None,
                base_labels=project["labels"] if project else None, revision=revision,
                target_project_id=payload.project_id or f"dataset-{uuid.uuid4().hex[:16]}")
    dump_json(store.root.parent / "import-previews" / f"{preview_id}.json", plan)
    return result


def dataset_history(store, project: dict) -> list[dict]:
    history, seen = [], set()
    revision = project.get("dataset_revision")
    while revision:
        if revision in seen or not re.fullmatch(r"import-[0-9a-f]{32}", revision):
            raise ValueError("Invalid dataset history chain")
        seen.add(revision)
        dataset = load_dataset(store, {**project, "dataset_revision": revision})
        history.append(dict(revision_id=revision, parent_revision_id=dataset.provenance.get("parent_revision_id"),
                            imported_at=dataset.provenance.get("imported_at"),
                            operation=dataset.provenance.get("operation", {"kind": "import", "format": dataset.provenance.get("format")}),
                            image_count=len(dataset.images), object_count=dataset.report.object_count))
        revision = dataset.provenance.get("parent_revision_id")
    return history


def commit_import(store, preview_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", preview_id):
        raise ValueError("Invalid import preview ID")
    path = store.root.parent / "import-previews" / f"{preview_id}.json"
    with store.locked(f"preview-{preview_id}"):
        plan = json.loads(path.read_text(encoding="utf-8"))
        if not plan["result"]["commit_allowed"]:
            raise ValueError("Resolve the duplicate-image errors and preview again before committing")
        project_id = plan["target_project_id"]
        with store.locked(project_id):
            try:
                project = store.get(project_id)
            except KeyError:
                project = None
            if plan.get("committed"):
                if not project:
                    raise StalePreview("The imported project has since been deleted")
                return project
            # Recover a successful publish even if the preview receipt write was interrupted.
            if project and any(row["revision_id"] == plan["revision"] for row in dataset_history(store, project)):
                return project
            if plan["input"]["project_id"]:
                if not project or project.get("dataset_revision") != plan["base_revision"] or project["labels"] != plan["base_labels"]:
                    raise StalePreview("Project data/categories changed since preview; preview again")
            elif project:
                raise StalePreview("The target project already exists")
            dataset = Dataset.model_validate(plan["dataset"])
            # A cheap existence check protects against committing a preview
            # after an image was removed, without reading every image again.
            for record in dataset.images:
                try:
                    _, image_path = image_location(project or {}, record, dataset)
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                except (KeyError, OSError, ValueError) as exc:
                    raise StalePreview(f"Image missing since preview: {record.path}") from exc
            if project is None:
                project = store.create(dict(name=plan["input"]["name"], task=dataset.task,
                                            dataset_path=dataset.sources[0].root,
                                            labels=[c.name for c in dataset.categories],
                                            model_max_side=plan["input"]["model_max_side"]),
                                       project_id=project_id, persist=False)
            # Read/save occurs under the same lock as edits and other appends.
            persist_import(store, project, dataset, revision=plan["revision"], hash_images=False)
            plan["committed"] = True
            pending = path.with_suffix(".json.tmp")
            dump_json(pending, plan)
            pending.replace(path)
            return project
