"""Every supported detection import can feed every detection exporter."""
from types import SimpleNamespace

import pytest

from visionrefine.core.dataset_io import registry
from visionrefine.core.dataset_io.service import package_snapshot, prepare_import
from visionrefine.core.dataset_io.testing import detection_fixture, assert_detection_equivalent

FORMATS = [f["id"] for f in registry.capabilities() if f["can_export"] and "detection" in f["tasks"]]


def write_and_read(store, dataset, root, format_id):
    project = dict(id="matrix", task="detection", dataset_path=str(root))
    report = package_snapshot(store, project, dataset.model_copy(deep=True), format_id, "all_annotated", True,
                              dict(skipped_images=[], revisions=[]))
    folder = store.root / "matrix/exports" / report["export_id"]
    roots = {"coco_detection": folder / "images", "cvat_detection": folder / "images"}
    inputs = {"coco_detection": "annotations.json", "yolo_detection": "data.yaml", "cvat_detection": "annotations.xml",
              "label_studio_detection": "annotations.json", "labelme_detection": "images", "visionrefine": "dataset.json"}
    incoming_root = roots.get(format_id, folder)
    source = folder / inputs[format_id] if format_id in inputs else None
    restored = prepare_import(incoming_root, format_id, source, [], "detection", "val")
    mapping = {a: b.removeprefix("images/") if format_id in roots else b for a, b in report["image_paths"].items()}
    assert_detection_equivalent(dataset, restored, mapping, tolerance=1 if format_id == "voc_detection" else 1e-6,
                                compare_splits=True)
    assert not any(issue.severity == "error" for issue in restored.report.issues)
    return restored, incoming_root


@pytest.mark.parametrize("input_format", FORMATS)
@pytest.mark.parametrize("output_format", FORMATS)
def test_all_detection_format_pairs(tmp_path, input_format, output_format):
    original = detection_fixture(tmp_path / "input")
    store = SimpleNamespace(root=tmp_path / "projects")
    incoming, root = write_and_read(store, original, tmp_path / "input", input_format)
    write_and_read(store, incoming, root, output_format)


@pytest.mark.parametrize("format_id", ["cvat_detection", "voc_detection", "label_studio_detection"])
def test_xml_control_characters_block_preflight(tmp_path, format_id):
    dataset = detection_fixture(tmp_path / "input")
    dataset.categories[0].name = dataset.images[0].objects[0].label = "bad\x00label"
    issues = registry.get(format_id, "detection", "exporter").preflight(dataset, True)
    assert any(i["code"] == "invalid_xml_text" and i["severity"] == "blocking" for i in issues)
