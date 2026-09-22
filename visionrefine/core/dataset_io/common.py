from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

from .models import Category, Dataset, ImageRecord, ImportReport, contained_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_image(root: Path, relative: str, split: str, report: ImportReport) -> ImageRecord | None:
    try:
        path = contained_path(root, relative)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Unsupported image extension")
        with Image.open(path) as image:
            width, height = image.size
        return ImageRecord(id=relative, path=relative, width=width, height=height, split=split)
    except (OSError, ValueError) as exc:
        report.skipped_images += 1
        report.add("unreadable_image", str(relative), str(exc), "error")
        return None


def clean_box(values, image: ImageRecord, report: ImportReport, location: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("Expected four bbox coordinates")
    box = [float(v) for v in values]
    if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Non-finite, reversed, or empty bbox")
    clipped = [max(0, min(image.width, box[0])), max(0, min(image.height, box[1])),
               max(0, min(image.width, box[2])), max(0, min(image.height, box[3]))]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("Box is entirely outside the image")
    if clipped != box:
        report.add("clipped_bbox", location, "Box clipped to original image bounds")
    return clipped


def finish(dataset: Dataset) -> Dataset:
    dataset.report.image_count = len(dataset.images)
    dataset.report.object_count = sum(len(i.objects) for i in dataset.images)
    dataset.report.empty_images = sum(not i.objects for i in dataset.images)
    if not dataset.images:
        raise ValueError("No readable images matched the dataset; check the root and annotation paths")
    return Dataset.model_validate(dataset.model_dump())


def make_categories(rows: list[tuple[int, str]], report: ImportReport) -> tuple[list[Category], dict[int, Category]]:
    categories: list[Category] = []
    mapping: dict[int, Category] = {}
    by_name: dict[str, Category] = {}
    for source_id, name in rows:
        if type(source_id) is not int or not isinstance(name, str) or not name.strip():
            raise ValueError("Categories require integer IDs and nonempty names")
        if source_id in mapping:
            raise ValueError(f"Duplicate category ID {source_id}")
        name = name.strip()
        if name in by_name:
            category = by_name[name]
            report.add("duplicate_category_name", str(source_id), f"Merged duplicate category name: {name}")
            category.provenance["source_ids"].append(source_id)
        else:
            category = Category(id=len(categories), name=name, source_id=source_id, provenance={"source_ids": [source_id]})
            categories.append(category)
            by_name[name] = category
        mapping[source_id] = category
    if not categories:
        raise ValueError("At least one category is required")
    return categories, mapping


class ImagesImporter:
    def read(self, root: Path, source: Path | None, labels: list[str], split: str) -> Dataset:
        if source is not None:
            raise ValueError("Choose an annotation format when supplying an annotation file")
        dataset = Dataset(categories=[Category(id=i, name=name) for i, name in enumerate(labels)])
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("._"):
                image = read_image(root, path.relative_to(root).as_posix(), split, dataset.report)
                if image:
                    dataset.images.append(image)
        return finish(dataset)
