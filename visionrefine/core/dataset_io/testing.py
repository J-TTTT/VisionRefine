"""Small synthetic fixtures for adapter authors; never reads a user's datasets."""
from pathlib import Path

from PIL import Image

from .models import Annotation, Category, Dataset, ImageRecord


def detection_fixture(root: Path) -> Dataset:
    """Create two distinct images, an unused class and an explicit empty review."""
    root.mkdir(parents=True, exist_ok=False)
    Image.new("RGB", (100, 80), "white").save(root / "测试.jpg")
    Image.new("RGB", (100, 80), "red").save(root / "empty.png")
    return Dataset(categories=[Category(id=0, name="cell"), Category(id=1, name="unused")], images=[
        ImageRecord(id="a", path="测试.jpg", width=100, height=80, split="val", status="human_reviewed",
            objects=[Annotation(id="box", category_id=0, label="cell", bbox=[10.25, 20.5, 40.75, 60.5],
                                source="human_reviewed")]),
        ImageRecord(id="b", path="empty.png", width=100, height=80, split="val", status="human_reviewed")])


def assert_detection_equivalent(expected, actual, path_mapping, *, tolerance=1e-6, compare_splits=False):
    """Compare semantic boxes per image, allowing declared path/ID remapping.

    Supply an explicit expected-path -> imported-path map, and raise tolerance only
    for a documented conversion (e.g. VOC's outward integer rounding).
    """
    assert {c.name for c in expected.categories} == {c.name for c in actual.categories}
    assert len(expected.images) == len(actual.images)
    indexed = {i.path: i for i in actual.images}
    assert set(path_mapping) == {i.path for i in expected.images}
    assert set(path_mapping.values()) == set(indexed)
    for image in expected.images:
        restored = indexed[path_mapping[image.path]]
        assert (image.width, image.height) == (restored.width, restored.height)
        if compare_splits:
            assert image.split == restored.split
        key = lambda obj: (obj.label, obj.bbox)
        before, after = sorted(image.objects, key=key), sorted(restored.objects, key=key)
        assert len(before) == len(after)
        for a, b in zip(before, after):
            assert a.label == b.label
            assert all(abs(x-y) <= tolerance for x, y in zip(a.bbox, b.bbox))
