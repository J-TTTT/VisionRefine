from __future__ import annotations

import json
from pathlib import Path

from .common import clean_box, dump_json, finish, make_categories, read_image
from .models import Annotation, Dataset


class CocoDetection:
    def read(self, root: Path, source: Path | None, labels: list[str], split: str) -> Dataset:
        if source is None:
            raise ValueError("COCO Detection requires an annotation JSON file")
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or any(not isinstance(data.get(k), list) for k in ("images", "categories", "annotations")):
            raise ValueError("COCO requires images, categories and annotations arrays")
        dataset = Dataset()
        extra = sorted(set(data) - {"images", "categories", "annotations"})
        if extra:
            dataset.report.add("unsupported_fields", "dataset", f"Detection import omits top-level fields: {', '.join(extra)}")
        try:
            dataset.categories, categories = make_categories(
                [(c["id"], c["name"]) for c in data["categories"]], dataset.report)
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid COCO category catalog") from exc
        for category in data["categories"]:
            omitted = sorted(set(category) - {"id", "name"})
            if omitted:
                dataset.report.add("unsupported_fields", f"category {category['id']}", f"Omitted category fields: {', '.join(omitted)}")
        images = {}
        seen_ids, seen_paths = set(), set()
        for row in data["images"]:
            location = str(row.get("id", "unknown")) if isinstance(row, dict) else "images"
            try:
                image_id, relative = row["id"], row["file_name"]
                if type(image_id) is not int or image_id in seen_ids:
                    raise ValueError("Missing/noninteger or duplicate image ID")
                seen_ids.add(image_id)
                image = read_image(root, relative, row.get("split", split), dataset.report)
                if image is None:
                    continue
                if image.path in seen_paths:
                    raise ValueError("Duplicate image file_name")
                seen_paths.add(image.path)
                if (row.get("width"), row.get("height")) != (image.width, image.height):
                    dataset.report.add("dimension_mismatch", relative, "Using actual local image dimensions")
                image.status = "imported_coarse"
                image.provenance = {"source_id": image_id}
                omitted = sorted(set(row) - {"id", "file_name", "width", "height", "split"})
                if omitted:
                    dataset.report.add("unsupported_fields", relative, f"Omitted image fields: {', '.join(omitted)}")
                images[image_id] = image
                dataset.images.append(image)
            except (KeyError, TypeError, ValueError) as exc:
                dataset.report.skipped_images += 1
                dataset.report.add("invalid_image", location, str(exc), "error")
        seen_annotations = set()
        for index, row in enumerate(data["annotations"]):
            location = f"annotations[{index}]"
            try:
                annotation_id = row["id"]
                if type(annotation_id) is not int or annotation_id in seen_annotations:
                    raise ValueError("Missing/noninteger or duplicate annotation ID")
                seen_annotations.add(annotation_id)
                image = images.get(row["image_id"])
                if image is None:
                    raise ValueError("Annotation references a missing or unreadable image")
                category = categories.get(row["category_id"])
                if category is None:
                    raise ValueError("Unknown category_id")
                x, y, width, height = [float(v) for v in row["bbox"]]
                box = clean_box([x, y, x + width, y + height], image, dataset.report, location)
                crowd = row.get("iscrowd", 0)
                if crowd not in (0, 1):
                    raise ValueError("iscrowd must be 0 or 1")
                ignored = sorted(set(row) - {"id", "image_id", "category_id", "bbox", "iscrowd", "area", "score"})
                if ignored:
                    dataset.report.add("unsupported_fields", location,
                                       f"Detection import omits fields: {', '.join(ignored)}")
                image.objects.append(Annotation(
                    id=f"import-{annotation_id}", category_id=category.id, label=category.name,
                    bbox=box, confidence=row.get("score"), attributes={"iscrowd": int(crowd)},
                    provenance={"source_id": annotation_id, "source_category_id": row["category_id"]},
                ))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                dataset.report.skipped_objects += 1
                dataset.report.add("invalid_annotation", location, str(exc), "error")
        return finish(dataset)

    def write(self, dataset: Dataset, output: Path) -> dict:
        category_ids = {}
        used = set()
        for category in dataset.categories:
            candidate = category.source_id
            if candidate is None or candidate < 0 or candidate in used:
                candidate = 1
                while candidate in used:
                    candidate += 1
            category_ids[category.id] = candidate
            used.add(candidate)
        rows, annotations, image_paths = [], [], {}
        split_counts = {}
        for image_id, image in enumerate(dataset.images, 1):
            rows.append(dict(id=image_id, file_name=image.path, width=image.width, height=image.height))
            # COCO has no split field: write a standard COCO file per split below.
            split_counts.setdefault(image.split, []).append(image_id)
            image_paths[image.path] = f"images/{image.path}"
            for obj in image.objects:
                x1, y1, x2, y2 = obj.bbox
                annotations.append(dict(id=len(annotations) + 1, image_id=image_id,
                    category_id=category_ids[obj.category_id], bbox=[x1, y1, x2-x1, y2-y1],
                    area=(x2-x1)*(y2-y1), iscrowd=obj.attributes.get("iscrowd", 0)))
        categories = [dict(id=category_ids[c.id], name=c.name) for c in dataset.categories]
        dump_json(output / "annotations.json", dict(images=rows, annotations=annotations, categories=categories))
        for split, ids in split_counts.items():
            subset = set(ids)
            dump_json(output / "annotations" / f"instances_{split}.json", dict(
                images=[i for i in rows if i["id"] in subset],
                annotations=[a for a in annotations if a["image_id"] in subset], categories=categories))
        warnings = ["Image/annotation IDs are regenerated. Revision metadata and confidence are stored in visionrefine.json; COCO area is bbox area.",
                    "annotations.json combines all splits; use annotations/instances_<split>.json for split-specific COCO files."]
        if any(set(obj.attributes) - {"iscrowd"} for image in dataset.images for obj in image.objects):
            warnings.append("Attributes unsupported by COCO Detection (e.g. VOC difficult/pose) are retained only in visionrefine.json.")
        return dict(image_paths=image_paths, category_mapping=[dict(label=c.name, id=category_ids[c.id]) for c in dataset.categories],
                    warnings=warnings)
