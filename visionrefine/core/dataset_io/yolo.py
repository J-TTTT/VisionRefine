from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath

import yaml

from .common import IMAGE_EXTENSIONS, clean_box, finish, make_categories, read_image
from .models import Annotation, Dataset, contained_path


def label_path(root: Path, image: str) -> Path:
    parts = list(PurePosixPath(image).parts)
    if "images" in parts[:-1]:
        # Match YOLO's final /images/ -> /labels/ convention.
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
    return contained_path(root, PurePosixPath(*parts).with_suffix(".txt").as_posix())


class YoloDetection:
    def read(self, root: Path, source: Path | None, labels: list[str], split: str) -> Dataset:
        if source is None:
            raise ValueError("YOLO Detection requires a data.yaml file")
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YOLO YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("YOLO YAML must contain a mapping")
        names = data.get("names")
        if isinstance(names, list):
            rows = list(enumerate(names))
        elif isinstance(names, dict):
            try:
                rows = sorted((int(k), v) for k, v in names.items())
            except (TypeError, ValueError) as exc:
                raise ValueError("YOLO names keys must be integer class indices") from exc
        else:
            raise ValueError("YOLO YAML requires names as a list or index/name mapping")
        if [r[0] for r in rows] != list(range(len(rows))):
            raise ValueError("YOLO class indices must be contiguous from zero")
        dataset = Dataset()
        dataset.categories, categories = make_categories(rows, dataset.report)
        if data.get("nc", len(rows)) != len(rows):
            raise ValueError("YOLO nc differs from the number of class names")
        if data.get("download"):
            dataset.report.add("download_ignored", str(source), "Download instructions are never executed")
        # The user-selected root is authoritative; YAML cannot redirect file reads.
        # Permit path: . or a path resolving to that root, report relocation otherwise.
        declared = data.get("path")
        if declared is not None:
            if not isinstance(declared, str):
                raise ValueError("YOLO path must be a string")
            declared_root = (source.parent / declared).resolve()
            if declared_root != root.resolve():
                dataset.report.add("root_override", "path", "Using selected dataset root instead of YAML path")
        selected: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            entries = data.get(split_name)
            if entries is None or entries == "":
                continue
            if isinstance(entries, str):
                entries = [entries]
            if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
                raise ValueError(f"YOLO {split_name} must be a path or a list of paths")
            for entry in entries:
                entry_path = self.resolve_entry(root, entry)
                if entry_path.is_dir():
                    paths = [p for p in sorted(entry_path.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                elif entry_path.is_file() and entry_path.suffix.lower() == ".txt":
                    paths = []
                    for line in entry_path.read_text(encoding="utf-8-sig").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            paths.append(self.resolve_entry(root, line, entry_path.parent if line.startswith("./") else root))
                        except ValueError as exc:
                            dataset.report.skipped_images += 1
                            dataset.report.add("invalid_image_path", line, str(exc), "error")
                else:
                    raise ValueError(f"YOLO split path does not exist or is not a directory/image list: {entry}")
                for path in paths:
                    try:
                        relative = path.resolve().relative_to(root.resolve()).as_posix()
                    except ValueError:
                        dataset.report.skipped_images += 1
                        dataset.report.add("invalid_image_path", str(path), "Image escapes selected root", "error")
                        continue
                    if relative in selected:
                        if selected[relative] != split_name:
                            raise ValueError(f"Image appears in multiple splits: {relative}")
                        dataset.report.add("duplicate_image", relative, "Duplicate split entry ignored")
                    selected[relative] = split_name
        if not selected:
            raise ValueError("YOLO YAML must reference images through train/val/test directories or lists")
        labels_seen: dict[Path, str] = {}
        for relative, split_name in selected.items():
            image = read_image(root, relative, split_name, dataset.report)
            if image is None:
                continue
            image.status = "imported_coarse"
            dataset.images.append(image)
            try:
                annotation_file = label_path(root, relative)
                if annotation_file in labels_seen:
                    raise ValueError(f"Ambiguous label file shared by {labels_seen[annotation_file]} and {relative}")
                labels_seen[annotation_file] = relative
                if not annotation_file.exists():
                    dataset.report.add("missing_label_file", relative, "YOLO permits no label file for a negative image; imported as empty coarse annotation")
                    continue
                lines = annotation_file.read_text(encoding="utf-8-sig").splitlines()
            except OSError as exc:
                dataset.report.add("unreadable_labels", relative, str(exc), "error")
                image.status = "unreviewed"
                continue
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                location = f"{annotation_file.relative_to(root)}:{number}"
                try:
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError("Detection expects exactly class cx cy w h; segmentation/pose rows are unsupported")
                    class_id = int(fields[0])
                    category = categories.get(class_id)
                    if category is None:
                        raise ValueError("Unknown YOLO class index")
                    cx, cy, w, h = [float(v) for v in fields[1:]]
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                        raise ValueError("Expected finite normalized coordinates in [0, 1] and positive width/height")
                    box = clean_box([(cx-w/2)*image.width, (cy-h/2)*image.height,
                                     (cx+w/2)*image.width, (cy+h/2)*image.height], image, dataset.report, location)
                    image.objects.append(Annotation(id=f"import-{len(dataset.images)}-{number}", category_id=category.id,
                        label=category.name, bbox=box, provenance={"line": number, "source_category_id": class_id}))
                except (TypeError, ValueError, OverflowError) as exc:
                    dataset.report.skipped_objects += 1
                    dataset.report.add("invalid_annotation", location, str(exc), "error")
        return finish(dataset)

    @staticmethod
    def resolve_entry(root: Path, entry: str, base: Path | None = None) -> Path:
        if "\\" in entry or "\x00" in entry:
            raise ValueError("Use portable image paths")
        path = (Path(entry) if Path(entry).is_absolute() else (base or root) / entry).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("YOLO path escapes selected dataset root")
        return path

    def write(self, dataset: Dataset, output: Path) -> dict:
        category_ids = {c.id: i for i, c in enumerate(dataset.categories)}
        image_paths, splits, used_labels = {}, {}, set()
        warnings = ["YOLO class IDs are remapped to contiguous zero-based indices; mappings, confidence and revision metadata are in the sidecars."]
        for image in dataset.images:
            split = image.split if image.split != "unspecified" else "train"
            if image.split == "unspecified" and "Unspecified images assigned to train; no validation set was invented." not in warnings:
                warnings.append("Unspecified images assigned to train; no validation set was invented.")
            parts = list(PurePosixPath(image.path).parts)
            if parts[0] == "images":
                parts = parts[1:]
                if len(parts) > 1 and parts[0] in {"train", "val", "test"}:
                    parts = parts[1:]
            # Only the outer images/ component may participate in YOLO's
            # last-/images/-to-/labels/ rule. Preserve relocated paths in the map.
            parts[:-1] = ["_images" if p == "images" else p for p in parts[:-1]]
            relative = PurePosixPath(*parts)
            image_out = PurePosixPath("images", split, relative)
            label_out = PurePosixPath("labels", split, relative).with_suffix(".txt")
            if str(label_out) in used_labels:
                suffix = hashlib.sha256(image.path.encode()).hexdigest()[:12]
                relative = relative.with_name(f"{relative.stem}__{suffix}{relative.suffix}")
                image_out = PurePosixPath("images", split, relative)
                label_out = PurePosixPath("labels", split, relative).with_suffix(".txt")
                warnings.append(f"Renamed conflicting label stem: {image.path} -> {image_out}")
            if str(label_out) in used_labels:
                raise ValueError("Unresolvable YOLO filename collision")
            used_labels.add(str(label_out))
            image_paths[image.path] = str(image_out)
            splits.setdefault(split, []).append(f"./{image_out}")
            lines = []
            for obj in image.objects:
                x1, y1, x2, y2 = obj.bbox
                values = [(x1+x2)/2/image.width, (y1+y2)/2/image.height, (x2-x1)/image.width, (y2-y1)/image.height]
                lines.append(f"{category_ids[obj.category_id]} " + " ".join(f"{v:.12g}" for v in values))
                if obj.attributes:
                    warning = "YOLO Detection cannot express object attributes (including iscrowd); retained in visionrefine.json."
                    if warning not in warnings:
                        warnings.append(warning)
            path = output / str(label_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        config = {"path": ".", "names": {category_ids[c.id]: c.name for c in dataset.categories}}
        for split, paths in splits.items():
            (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
            config[split] = f"{split}.txt"
        (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if "train" not in splits or "val" not in splits:
            warnings.append("Training frameworks may require both train and val splits; configure missing splits before training.")
        return dict(image_paths=image_paths, category_mapping=[dict(label=c.name, id=category_ids[c.id]) for c in dataset.categories], warnings=warnings)
