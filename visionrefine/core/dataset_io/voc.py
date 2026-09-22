"""Pascal VOC Detection: one-based inclusive XML <-> zero-based pixel edges."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from .common import clean_box, finish, read_image
from .models import Annotation, Category, Dataset, contained_path, safe_relative_path


def xml_read(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("VOC XML must not declare DTDs or entities")
    document = ET.fromstring(text)
    if document.tag != "annotation":
        raise ValueError("Expected VOC <annotation> root")
    return document


class VocDetection:
    def read(self, root: Path, source: Path | None, labels: list[str], split: str) -> Dataset:
        annotation_root = source or root / "Annotations"
        if not annotation_root.is_dir():
            raise ValueError("VOC requires an Annotations directory (dataset root contains JPEGImages)")
        dataset = Dataset()
        split_by_id = {}
        for name in ("train", "val", "test", "unspecified"):
            split_file = contained_path(root, f"ImageSets/Main/{name}.txt")
            if split_file.is_file():
                for entry in split_file.read_text(encoding="utf-8-sig").splitlines():
                    entry = entry.strip()
                    if not entry:
                        continue
                    safe_relative_path(entry)
                    if entry in split_by_id and split_by_id[entry] != name:
                        raise ValueError(f"VOC image belongs to multiple splits: {entry}")
                    split_by_id[entry] = name
        categories = {}

        def category(name):
            name = name.strip()
            if not name:
                raise ValueError("Empty VOC category name")
            if name not in categories:
                item = Category(id=len(categories), name=name)
                categories[name] = item
                dataset.categories.append(item)
            return categories[name]

        classes = contained_path(root, "classes.txt")
        if classes.is_file():
            for name in classes.read_text(encoding="utf-8-sig").splitlines():
                if name.strip():
                    category(name)
        seen = set()
        for path in sorted(annotation_root.rglob("*.xml")):
            location = path.relative_to(annotation_root).as_posix()
            try:
                contained_path(annotation_root, location)
                document = xml_read(path)
                image_id = path.relative_to(annotation_root).with_suffix("").as_posix()
                relative = "JPEGImages/" + safe_relative_path(document.findtext("filename", ""))
                record = read_image(root, relative, split_by_id.get(image_id, split), dataset.report)
                if record is None:
                    continue
                if relative in seen:
                    raise ValueError("Multiple XML files reference the same image")
                seen.add(relative)
                record.status = "imported_coarse"
                record.provenance = {"xml": location}
                size = document.find("size")
                if size is not None and (size.findtext("width"), size.findtext("height")) != (str(record.width), str(record.height)):
                    dataset.report.add("dimension_mismatch", location, "Using actual image dimensions")
                unsupported = {c.tag for c in document} - {"folder", "filename", "path", "source", "size", "segmented", "object"}
                if unsupported or document.findtext("segmented", "0") != "0":
                    dataset.report.add("unsupported_fields", location, "Segmentation/non-detection XML fields are not imported")
                for index, obj in enumerate(document.findall("object")):
                    try:
                        box = obj.find("bndbox")
                        if box is None:
                            raise ValueError("Object has no bndbox")
                        xmin, ymin, xmax, ymax = [float(box.findtext(k, "")) for k in ("xmin", "ymin", "xmax", "ymax")]
                        coords = clean_box([xmin - 1, ymin - 1, xmax, ymax], record, dataset.report, f"{location}:{index}")
                        attributes = {key: int(obj.findtext(key, "0")) for key in ("difficult", "truncated", "occluded")}
                        if any(v not in (0, 1) for v in attributes.values()):
                            raise ValueError("VOC flags must be 0 or 1")
                        attributes["pose"] = obj.findtext("pose", "Unspecified")
                        item = category(obj.findtext("name", ""))
                        record.objects.append(Annotation(id=f"voc-{index}", category_id=item.id, label=item.name,
                            bbox=coords, attributes=attributes, provenance={"xml": location, "object_index": index}))
                        omitted = {c.tag for c in obj} - {"name", "bndbox", "pose", "difficult", "truncated", "occluded"}
                        if omitted:
                            dataset.report.add("unsupported_fields", f"{location}:{index}", f"Omitted: {', '.join(sorted(omitted))}")
                    except (TypeError, ValueError, OverflowError) as exc:
                        dataset.report.skipped_objects += 1
                        dataset.report.add("invalid_annotation", f"{location}:{index}", str(exc), "error")
                dataset.images.append(record)
            except (OSError, ValueError, ET.ParseError) as exc:
                dataset.report.skipped_images += 1
                dataset.report.add("invalid_xml", location, str(exc), "error")
        if not categories:
            for name in labels:
                category(name)
        if not categories:
            raise ValueError("Empty VOC annotations require classes.txt or supplied labels")
        imported_ids = {i.provenance["xml"][:-4] for i in dataset.images}
        for image_id in sorted(set(split_by_id) - imported_ids):
            dataset.report.add("missing_annotation", image_id, "Split entry has no readable XML/image pair", "error")
        return finish(dataset)

    def write(self, dataset: Dataset, output: Path) -> dict:
        if any(any(ord(c) < 32 for c in category.name) for category in dataset.categories):
            raise ValueError("VOC class names cannot contain control characters or line breaks")
        (output / "Annotations").mkdir(parents=True, exist_ok=True)
        (output / "ImageSets/Main").mkdir(parents=True, exist_ok=True)
        image_paths, splits = {}, {}
        warnings = ["VOC uses one-based inclusive integer boxes; fractional edges are expanded to enclosing pixels.",
                    "VOC image IDs are stable hashes of internal paths. Confidence and revision metadata remain in visionrefine.json."]
        for image in dataset.images:
            image_id = "vr-" + hashlib.sha256(image.path.encode()).hexdigest()[:24]
            image_paths[image.path] = f"JPEGImages/{image_id}.jpg"
            splits.setdefault(image.split, []).append(image_id)
            document = ET.Element("annotation")

            def node(parent, name, value):
                ET.SubElement(parent, name).text = str(value)

            node(document, "folder", "JPEGImages")
            node(document, "filename", image_id + ".jpg")
            size = ET.SubElement(document, "size")
            for name, value in (("width", image.width), ("height", image.height), ("depth", 3)):
                node(size, name, value)
            node(document, "segmented", 0)
            for obj in image.objects:
                element = ET.SubElement(document, "object")
                node(element, "name", obj.label)
                node(element, "pose", obj.attributes.get("pose", "Unspecified"))
                for flag in ("difficult", "truncated", "occluded"):
                    value = obj.attributes.get(flag, 0)
                    if value not in (0, 1):
                        raise ValueError(f"Invalid VOC {flag} attribute")
                    node(element, flag, int(value))
                x1, y1, x2, y2 = obj.bbox
                box = ET.SubElement(element, "bndbox")
                for name, value in zip(("xmin", "ymin", "xmax", "ymax"),
                                       (math.floor(x1) + 1, math.floor(y1) + 1, math.ceil(x2), math.ceil(y2))):
                    node(box, name, value)
                if set(obj.attributes) - {"pose", "difficult", "truncated", "occluded"}:
                    warning = "Attributes unsupported by VOC (including iscrowd) are retained only in visionrefine.json."
                    if warning not in warnings:
                        warnings.append(warning)
            ET.indent(document)
            ET.ElementTree(document).write(output / "Annotations" / f"{image_id}.xml", encoding="utf-8", xml_declaration=True)
        for split, ids in splits.items():
            (output / "ImageSets/Main" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        trainval = splits.get("train", []) + splits.get("val", [])
        if trainval:
            (output / "ImageSets/Main/trainval.txt").write_text("\n".join(trainval) + "\n", encoding="utf-8")
        (output / "classes.txt").write_text("\n".join(c.name for c in dataset.categories) + "\n", encoding="utf-8")
        warnings.append("VOC packages require JPEG images. Non-JPEG inputs are converted when images are included; originals stay unchanged.")
        return dict(image_paths=image_paths, image_encoding="jpeg", warnings=warnings,
                    category_mapping=[dict(label=c.name, id=c.id) for c in dataset.categories])
