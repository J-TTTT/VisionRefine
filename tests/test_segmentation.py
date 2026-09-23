import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

import visionrefine.server as server
from visionrefine.core.dataset_io.models import Annotation, Dataset, Category, ImageRecord
from visionrefine.core.dataset_io.service import select_snapshot
from visionrefine.core.segmentation import TileMask, mask_metrics
from visionrefine.core.store import ProjectStore


def polygon(**values):
    return dict(id="cell-1", kind="polygon", category_id=0, label="cell",
                polygons=[[[10, 20], [50, 20], [50, 60], [10, 60]]], **values)


def tile(points, x=0, y=0):
    data = [0] * (128 * 128)
    for px, py in points:
        data[py * 128 + px] = 1
    counts, bit, count = [], 0, 0
    for value in data:
        if value == bit:
            count += 1
        else:
            counts.append(count)
            bit, count = value, 1
    counts.append(count)
    return dict(x=x, y=y, counts=counts)


def test_polygon_derived_metrics_and_multiple_components():
    values = polygon(bbox=[0, 0, 1, 1], area=999)
    values["polygons"].append([[80, 20], [90, 20], [90, 30], [80, 30]])
    annotation = Annotation.model_validate(values)
    assert annotation.bbox == [10, 20, 90, 60]
    assert annotation.area == 1700
    restored = Annotation.model_validate_json(annotation.model_dump_json())
    assert restored == annotation


@pytest.mark.parametrize("parts", [
    [[[0, 0], [5, 5], [0, 5], [5, 0]]],  # crossing
    [[[0, 0], [5, 0], [3, 0], [3, 5], [0, 5]]],  # double back
    [[[0, 0], [1, 0], [2, 0]]],
    [[[0, 0], [5, 0], [5, 5], [0, 0]]],  # repeated closure
    [[[0, 0], [10, 0], [10, 10], [0, 10]], [[2, 2], [4, 2], [4, 4]]],
    [[[0, 0], [10, 0], [10, 10]], [[5, 0], [15, 0], [15, 10]]],
    [[[0, 0], [float("inf"), 1], [3, 5]]],
])
def test_invalid_polygons_rejected(parts):
    values = polygon()
    values["polygons"] = parts
    with pytest.raises(ValidationError):
        Annotation.model_validate(values)


def test_sparse_mask_hole_and_large_original_coordinates():
    # Ring with a one-pixel hole, plus a distant one-pixel component.
    pixels = [(x, y) for y in range(3) for x in range(3) if (x, y) != (1, 1)]
    values = dict(id="mask", kind="mask", category_id=0, label="cell", mask={"tiles": [
        tile(pixels, 128, 256), tile([(0, 0)], 99840, 199936),
    ]})
    annotation = Annotation.model_validate(values)
    assert annotation.bbox == [128, 256, 99841, 199937]
    assert annotation.area == 9
    assert Annotation.model_validate_json(annotation.model_dump_json()) == annotation
    # This validates a 20-billion-pixel image without allocating its bitmap.
    dataset = Dataset(task="instance_segmentation", categories=[Category(id=0, name="cell")],
                      images=[ImageRecord(id="huge", path="huge.png", width=100000, height=200000, objects=[annotation])])
    assert dataset.schema_version == "3.0"


@pytest.mark.parametrize("mask", [
    {"tiles": [dict(x=1, y=0, counts=[0, 16384])]},
    {"tiles": [dict(x=0, y=0, counts=[0, 16385])]},
    {"tiles": [dict(x=0, y=0, counts=[16384, 0])]},
    {"tiles": [dict(x=0, y=0, counts=[-1, 16385])]},
    {"tiles": [dict(x=0, y=0, counts=[0, 16384.0])]},
    {"tiles": [tile([(0, 0)]), tile([(1, 1)])]},
    {"tiles": [], "encoding": "coco-rle"},
])
def test_invalid_masks_rejected(mask):
    with pytest.raises(ValidationError):
        TileMask.model_validate(mask)


def test_mask_run_crossing_rows_metrics():
    mask = TileMask(tiles=[tile([(127, 0), (0, 1), (1, 1)])])
    assert mask_metrics(mask) == ([0, 0, 128, 2], 3)


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (640, 480), "white").save(root / "sample.png")
    monkeypatch.setattr(server, "store", ProjectStore(tmp_path / "workspace/projects"))
    client = TestClient(server.app)
    p = client.post("/api/projects", json=dict(name="Segmentation", task="instance_segmentation",
        dataset_path=str(root), labels=["cell", "defect"], model_max_side=1536)).json()
    assert client.post(f"/api/projects/{p['id']}/analyze").status_code == 200
    return client, p


def test_save_reload_history_empty_and_snapshot(project):
    client, p = project
    url = f"/api/projects/{p['id']}/annotations"
    objects = [polygon(), dict(id="mask", kind="mask", label="defect", mask={"tiles": [tile([(0, 0)], 128, 128)]})]
    first = client.put(url, json=dict(image="sample.png", objects=objects))
    assert first.status_code == 200, first.text
    first = first.json()
    assert first["annotation_schema_version"] == "2.0"
    assert client.get(url, params={"image": "sample.png"}).json() == first
    live = server.store.get(p["id"])
    snapshot, _ = select_snapshot(server.store, live, "reviewed", [])
    assert [o.kind for o in snapshot.images[0].objects] == ["polygon", "mask"]
    assert snapshot.images[0].objects[1].area == 1
    # AI suggestions cannot replace a saved human geometry.
    suggestion = server.suggestion_path(p["id"], "sample.png")
    suggestion.parent.mkdir()
    suggestion.write_text(json.dumps({"objects": [], "status": "ai_suggestion"}))
    assert client.get(url, params={"image": "sample.png"}).json() == first
    second = client.put(url, json=dict(image="sample.png", objects=[])).json()
    assert second["parent_revision_id"] == first["revision_id"]
    assert client.get(url, params={"image": "sample.png"}).json()["objects"] == []
    history = server.store.root / p["id"] / "revisions" / second["revision_id"] / "previous.json"
    assert json.loads(history.read_text()) == first


def test_save_rejects_outside_geometry_wrong_task_and_does_not_overwrite(project):
    client, p = project
    url = f"/api/projects/{p['id']}/annotations"
    good = client.put(url, json=dict(image="sample.png", objects=[polygon()])).json()
    bad = polygon()
    bad["polygons"][0][0] = [-1, 20]
    for value in [bad, {"label": "cell", "bbox": [1, 2, 3, 4]},
                  {"kind": "mask", "label": "cell", "mask": {"tiles": [tile([(127, 127)], 512, 384)]}},
                  {**polygon(), "mask": {"tiles": [tile([(0, 0)])]}}]:
        response = client.put(url, json=dict(image="sample.png", objects=[value]))
        assert response.status_code == 400, response.text
        assert client.get(url, params={"image": "sample.png"}).json() == good
    live = server.store.get(p["id"])
    live["task"] = "detection"
    server.store.save(live)
    assert client.put(url, json=dict(image="sample.png", objects=[polygon()])).status_code == 400
