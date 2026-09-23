"""Detection-only interchange with CVAT, Label Studio and wkentaro Labelme."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse, unquote
from xml.etree import ElementTree as ET

from .common import clean_box, dump_json, finish, read_image
from .models import Annotation, Category, Dataset, contained_path, safe_relative_path


def xml_document(path, tag):
    text = path.read_text(encoding="utf-8-sig")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("DTDs and entities are not allowed")
    try:
        doc = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML: {exc}") from None
    if doc.tag != tag:
        raise ValueError(f"Expected <{tag}>")
    return doc


def category(dataset, name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Category name must be nonempty")
    name = name.strip()
    existing = next((c for c in dataset.categories if c.name == name), None)
    if existing:
        return existing
    result = Category(id=len(dataset.categories), name=name)
    dataset.categories.append(result)
    return result


def seed_categories(dataset, root, labels):
    path = contained_path(root, "classes.json")
    values = json.loads(path.read_text()) if path.is_file() else labels
    if not isinstance(values, list):
        raise ValueError("classes.json must be a list of category names")
    for name in values:
        category(dataset, name)


def add_box(dataset, record, name, coordinates, index, attributes=None, provenance=None):
    box = clean_box(coordinates, record, dataset.report, f"{record.path}:{index}")
    cat = category(dataset, name)
    record.objects.append(Annotation(id=f"import-{index}", category_id=cat.id, label=cat.name,
        bbox=box, attributes=attributes or {}, provenance=provenance or {}))


def invalid(dataset, location, exc):
    dataset.report.skipped_objects += 1
    dataset.report.add("unsupported_or_invalid_annotation", str(location), str(exc), "error")


def complete(dataset, labels=()):
    if not dataset.categories:
        for name in labels:
            category(dataset, name)
    if not dataset.categories:
        raise ValueError("No categories found; supply labels or classes.json for empty annotations")
    return finish(dataset)


def unique_images(dataset):
    if len({i.path for i in dataset.images}) != len(dataset.images):
        raise ValueError("Multiple records refer to the same image; select one annotation version explicitly")


class CvatDetection:
    def read(self, root, source, labels, split):
        source = source or root / "annotations.xml"
        doc = xml_document(source, "annotations")
        if doc.findtext("version") not in {"1.0", "1.1"}:
            raise ValueError("Supported CVAT XML versions: 1.0, 1.1")
        if doc.findall("track") or doc.findtext("meta/task/mode", "annotation") != "annotation":
            raise ValueError("CVAT video tracks are not supported by the image detection adapter")
        dataset = Dataset()
        for node in doc.findall("meta/task/labels/label"):
            category(dataset, node.findtext("name", ""))
        if not dataset.categories:
            seed_categories(dataset, root, labels)
        for node in doc.findall("image"):
            name = node.get("name", "")
            record = read_image(root, name, split, dataset.report)
            if record is None:
                continue
            record.status = "imported_coarse"
            if (node.get("width"), node.get("height")) != (str(record.width), str(record.height)):
                dataset.report.add("dimension_mismatch", name, "Using actual image dimensions")
            for index, shape in enumerate(node):
                try:
                    if shape.tag != "box" or float(shape.get("rotation", "0")) != 0:
                        raise ValueError("Only axis-aligned CVAT boxes are supported")
                    attrs = {n.get("name", ""): n.text or "" for n in shape.findall("attribute")}
                    if attrs.get("iscrowd") in {"0", "1"}:
                        attrs["iscrowd"] = int(attrs["iscrowd"])
                    attrs.update(occluded=int(shape.get("occluded", "0")), z_order=int(shape.get("z_order", "0")))
                    if attrs["occluded"] not in {0, 1}:
                        raise ValueError("occluded must be 0 or 1")
                    add_box(dataset, record, shape.get("label"), [float(shape.get(k, "")) for k in ("xtl", "ytl", "xbr", "ybr")],
                            index, attrs, {"cvat_image_id": node.get("id")})
                except (ValueError, TypeError) as exc:
                    invalid(dataset, name, exc)
            dataset.images.append(record)
        unique_images(dataset)
        return complete(dataset)

    def write(self, dataset, output):
        doc = ET.Element("annotations")
        ET.SubElement(doc, "version").text = "1.1"
        task = ET.SubElement(ET.SubElement(doc, "meta"), "task")
        for key, value in (("id", "0"), ("name", "VisionRefine export"), ("size", str(len(dataset.images))), ("mode", "annotation")):
            ET.SubElement(task, key).text = value
        catalog = ET.SubElement(task, "labels")
        for cat in dataset.categories:
            label = ET.SubElement(catalog, "label")
            ET.SubElement(label, "name").text = cat.name
            attrs = ET.SubElement(label, "attributes")
            names = sorted({k for i in dataset.images for obj in i.objects if obj.label == cat.name for k in obj.attributes if k not in {"occluded", "z_order"}})
            for name in names:
                attribute = ET.SubElement(attrs, "attribute")
                for key, value in (("name", name), ("mutable", "False"), ("input_type", "text"), ("default_value", ""), ("values", "")):
                    ET.SubElement(attribute, key).text = value
        paths = {}
        for index, image in enumerate(dataset.images):
            paths[image.path] = f"images/{image.path}"
            element = ET.SubElement(doc, "image", id=str(index), name=image.path, width=str(image.width), height=str(image.height))
            for obj in image.objects:
                occluded, z_order = obj.attributes.get("occluded", 0), obj.attributes.get("z_order", 0)
                if occluded not in (0, 1) or type(z_order) is not int:
                    raise ValueError("Invalid CVAT occluded/z_order attribute")
                box = ET.SubElement(element, "box", label=obj.label, occluded=str(int(occluded)), source="manual" if image.status in {"human_reviewed", "final_reviewed"} else "auto",
                                    xtl=str(obj.bbox[0]), ytl=str(obj.bbox[1]), xbr=str(obj.bbox[2]), ybr=str(obj.bbox[3]), z_order=str(z_order))
                for name, value in obj.attributes.items():
                    if name not in {"occluded", "z_order"}:
                        ET.SubElement(box, "attribute", name=name).text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        ET.indent(doc)
        ET.ElementTree(doc).write(output / "annotations.xml", encoding="utf-8", xml_declaration=True)
        return dict(image_paths=paths, category_mapping=[c.model_dump(include={"id", "name"}) for c in dataset.categories], warnings=[])

    def preflight(self, dataset, include_images):
        from .compatibility import issue
        count = sum(not isinstance(v, str) for i in dataset.images for o in i.objects for k, v in o.attributes.items() if k not in {"occluded", "z_order"})
        bad = sum(o.attributes.get("occluded", 0) not in (0, 1) or type(o.attributes.get("z_order", 0)) is not int for i in dataset.images for o in i.objects)
        return ([issue("attribute_text", "CVAT 自定义属性以文本保存，原始类型留在附加清单。", count)] if count else []) + ([issue("invalid_cvat_flags", "occluded/z_order 属性值不合法。", bad, "blocking")] if bad else [])


class LabelmeDetection:
    def preflight(self, dataset, include_images):
        from .compatibility import issue
        bad = 0
        for image in dataset.images:
            for obj in image.objects:
                flags = obj.attributes.get("flags", {})
                group = obj.attributes.get("group_id")
                bad += (not isinstance(flags, dict) or any(type(v) is not bool for v in flags.values())
                        or (group is not None and type(group) is not int)
                        or not isinstance(obj.attributes.get("description", ""), str))
        return [issue("invalid_labelme_attributes", "Labelme flags 必须为布尔字典，group_id 为整数或空，description 为文本。", bad, "blocking")] if bad else []

    def read(self, root, source, labels, split):
        source = source or root
        paths = [source] if source.is_file() else sorted(source.rglob("*.json"))
        dataset = Dataset()
        seed_categories(dataset, root, [])
        for path in paths:
            if path.name in {"classes.json", "export-report.json", "visionrefine.json"}:
                continue
            try:
                if not path.resolve().is_relative_to(source.resolve() if source.is_dir() else source.parent.resolve()):
                    raise ValueError("Labelme annotation symlink escapes input directory")
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
                    raise ValueError("Expected Labelme shapes array")
                image_name = data.get("imagePath", "")
                # Labelme normally resolves imagePath relative to its JSON, but root is authoritative.
                if not isinstance(image_name, str) or Path(image_name).is_absolute() or "\\" in image_name or ":" in image_name:
                    raise ValueError("Labelme imagePath must be a local relative path")
                adjacent = (path.parent / image_name).resolve()
                relative = adjacent.relative_to(root).as_posix() if adjacent.is_relative_to(root) and adjacent.is_file() else image_name
                record = read_image(root, relative, split, dataset.report)
                if record is None:
                    continue
                record.status = "imported_coarse"
                if data.get("imageData"):
                    dataset.report.add("embedded_image_ignored", relative, "Using local image file, not embedded imageData")
                for index, shape in enumerate(data["shapes"]):
                    try:
                        if not isinstance(shape, dict):
                            raise ValueError("Labelme shape must be an object")
                        if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) != 2:
                            raise ValueError("Only Labelme rectangle with two points is supported")
                        (x1, y1), (x2, y2) = shape["points"]
                        add_box(dataset, record, shape.get("label"), [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)], index,
                                {k: shape[k] for k in ("flags", "group_id", "description") if k in shape})
                    except (KeyError, TypeError, ValueError) as exc:
                        invalid(dataset, relative, exc)
                dataset.images.append(record)
            except (OSError, ValueError, TypeError) as exc:
                dataset.report.add("invalid_labelme", path.name, str(exc), "error")
                dataset.report.skipped_images += 1
        unique_images(dataset)
        return complete(dataset, labels)

    def write(self, dataset, output):
        paths = {}
        dump_json(output / "classes.json", [c.name for c in dataset.categories])
        for image in dataset.images:
            stem = hashlib.sha256(image.path.encode()).hexdigest()[:24]
            image_name = stem + Path(image.path).suffix.lower()
            paths[image.path] = f"images/{image_name}"
            shapes = [dict(label=obj.label, points=[obj.bbox[:2], obj.bbox[2:]], shape_type="rectangle",
                           flags=obj.attributes.get("flags", {}), group_id=obj.attributes.get("group_id"),
                           description=obj.attributes.get("description", "")) for obj in image.objects]
            dump_json(output / "images" / f"{stem}.json", dict(version="5.5.0", flags={}, shapes=shapes,
                      imagePath=image_name, imageData=None, imageWidth=image.width, imageHeight=image.height))
        return dict(image_paths=paths, category_mapping=[c.model_dump(include={"id", "name"}) for c in dataset.categories], warnings=[])


class LabelStudioDetection:
    def read(self, root, source, labels, split):
        source = source or root / "annotations.json"
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("Label Studio input must be a task JSON array")
        dataset = Dataset()
        seed_categories(dataset, root, [])
        for task in data:
            try:
                if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
                    raise ValueError("Label Studio task must contain a data object")
                relative = task["data"]["image"]
                if not isinstance(relative, str):
                    raise ValueError("Label Studio data.image must be a path string")
                if relative.startswith("/data/local-files/?"):
                    relative = parse_qs(urlparse(relative).query).get("d", [""])[0]
                elif urlparse(relative).scheme:
                    raise ValueError("Remote image URLs are not fetched; bind local relative data.image paths first")
                record = read_image(root, relative, split, dataset.report)
                if record is None:
                    continue
                if any(not isinstance(a, dict) for a in task.get("annotations", []) + task.get("predictions", [])):
                    raise ValueError("Label Studio versions must be objects")
                annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
                predictions = task.get("predictions", [])
                versions = annotations or predictions
                if versions and not isinstance(versions[0].get("result", []), list):
                    raise ValueError("Label Studio result must be an array")
                if len(versions) > 1:
                    raise ValueError("Multiple annotation/prediction versions: select one per task before import")
                record.status = "imported_coarse" if versions else "unreviewed"
                record.provenance["label_studio_origin"] = "annotation" if annotations else "prediction" if predictions else "unlabeled"
                for index, result in enumerate(versions[0].get("result", []) if versions else []):
                    try:
                        if not isinstance(result, dict) or not isinstance(result.get("value"), dict):
                            raise ValueError("Label Studio result must contain a value object")
                        value = result["value"]
                        if result.get("type") != "rectanglelabels" or float(value.get("rotation", 0)) != 0 or float(result.get("image_rotation", 0)) != 0:
                            raise ValueError("Only nonrotated RectangleLabels results are supported")
                        names = value["rectanglelabels"]
                        if not isinstance(names, list) or len(names) != 1:
                            raise ValueError("A box must have exactly one class")
                        x,y,w,h = [float(value[k]) for k in ("x", "y", "width", "height")]
                        add_box(dataset, record, names[0], [x*record.width/100, y*record.height/100, (x+w)*record.width/100, (y+h)*record.height/100],
                                index, provenance={"label_studio_result_id": result.get("id"), "label_studio_origin": record.provenance["label_studio_origin"]})
                    except (KeyError, TypeError, ValueError) as exc:
                        invalid(dataset, record.path, exc)
                dataset.images.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                dataset.report.add("invalid_label_studio_task", str(task.get("id", "task")) if isinstance(task, dict) else "task", str(exc), "error")
                dataset.report.skipped_images += 1
        unique_images(dataset)
        return complete(dataset, labels)

    def write(self, dataset, output):
        paths, tasks = {}, []
        for index, image in enumerate(dataset.images):
            relative = f"images/{image.path}"
            paths[image.path] = relative
            results = []
            for n, obj in enumerate(image.objects):
                x1,y1,x2,y2 = obj.bbox
                results.append(dict(id=f"box-{n}", from_name="label", to_name="image", type="rectanglelabels",
                    original_width=image.width, original_height=image.height, image_rotation=0,
                    value=dict(x=x1/image.width*100, y=y1/image.height*100, width=(x2-x1)/image.width*100,
                               height=(y2-y1)/image.height*100, rotation=0, rectanglelabels=[obj.label])))
            from urllib.parse import quote
            task = dict(id=index+1, data={"image": "/data/local-files/?d=" + quote(relative, safe="/")})
            if image.status in {"human_reviewed", "final_reviewed"}:
                task["annotations"] = [dict(id=index+1, result=results, was_cancelled=False, ground_truth=False)]
            else:
                task["predictions"] = [dict(model_version="VisionRefine-imported", result=results)]
            tasks.append(task)
        dump_json(output / "annotations.json", tasks)
        dump_json(output / "classes.json", [c.name for c in dataset.categories])
        view = ET.Element("View")
        ET.SubElement(view, "Image", name="image", value="$image")
        labels = ET.SubElement(view, "RectangleLabels", name="label", toName="image")
        for cat in dataset.categories:
            ET.SubElement(labels, "Label", value=cat.name)
        # Label Studio consumes this as an XML fragment / Unicode string; an
        # encoding declaration makes its official lxml-based config loader fail.
        ET.ElementTree(view).write(output / "label_config.xml", encoding="utf-8", xml_declaration=False)
        (output / "IMPORT.md").write_text("Enable Label Studio local file serving and set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT to this extracted directory. Apply label_config.xml, then import annotations.json. No absolute server paths or credentials are included.\n", encoding="utf-8")
        return dict(image_paths=paths, category_mapping=[c.model_dump(include={"id", "name"}) for c in dataset.categories],
                    warnings=["Label Studio requires enabling local file serving and applying label_config.xml; see IMPORT.md."])
