"""Independent-reader smoke tests. Run with isolated verifier dependencies.

python scripts/verify_format_samples.py /tmp/new-samples
Does NOT import VisionRefine readers. Uses Datumaro CVAT and official Labelme /
Label Studio converter / pycocotools loaders. This is not a target-server UI test.
"""
import argparse
import json
import os
import math
import tempfile
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()
    index = json.loads((args.samples / "index.json").read_text())
    root = lambda key: args.samples / index[key]["directory"]
    from datumaro import Dataset, AnnotationType
    cvat = Dataset.import_from(str(root("cvat_detection")), "cvat")
    assert len(cvat) == 2
    boxes = [a for image in cvat for a in image.annotations if a.type == AnnotationType.bbox]
    assert len(boxes) == 1
    assert boxes[0].get_bbox() == [10.25, 20.5, 30.5, 40.0]
    assert cvat.categories()[AnnotationType.label][boxes[0].label].name == "cell"

    from labelme.label_file import LabelFile
    labelme = [LabelFile(str(p)) for p in (root("labelme_detection") / "images").glob("*.json")]
    assert len(labelme) == 2 and sum(len(p.shapes) for p in labelme) == 1
    shape = next(p.shapes[0] for p in labelme if p.shapes)
    assert shape["shape_type"] == "rectangle" and shape["label"] == "cell"
    assert shape["points"] == [[10.25, 20.5], [40.75, 60.5]]
    assert all(p.imageData for p in labelme)

    ls = root("label_studio_detection")
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(ls.resolve())
    from label_studio_converter import Converter
    converter = Converter(str(ls / "label_config.xml"), str(ls))
    with tempfile.TemporaryDirectory(prefix="visionrefine-ls-check-") as temp:
        converter.convert_to_coco(str(ls / "annotations.json"), temp, is_dir=False)
        data = json.loads((Path(temp) / "result.json").read_text())
        assert len(data["images"]) == 2 and len(data["annotations"]) == 1
        box = data["annotations"][0]["bbox"]
        assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-8) for a, b in zip(box, [10.25, 20.5, 30.5, 40.0])), box

    from pycocotools.coco import COCO
    coco = COCO(str(root("coco_detection") / "annotations.json"))
    assert len(coco.imgs) == 2 and len(coco.anns) == 1
    print(json.dumps({"passed": ["cvat_detection", "labelme_detection", "label_studio_detection", "coco_detection"],
        "verifiers": {p: version(p) for p in ("datumaro", "labelme", "label-studio-converter", "pycocotools")}}, indent=2))


if __name__ == "__main__":
    main()
