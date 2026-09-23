"""COCO instance segmentation interchange, including holes and sparse masks."""
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import visionrefine.server as server
from visionrefine.core.dataset_io.coco_segmentation import (
    CocoSegmentation, coco_counts_from_mask, decode_coco_counts, encode_coco_counts,
    mask_from_coco_counts,
)
from visionrefine.core.dataset_io.models import Category, Dataset, ImageRecord, Annotation
from visionrefine.core.segmentation import TileMask
from visionrefine.core.store import ProjectStore


def counts_for(pixels):
    flat = pixels.ravel(order="F")
    boundaries = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, flat.size]
    counts = np.diff(boundaries).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return counts


def ring():
    pixels = np.zeros((260, 300), dtype=np.uint8)
    pixels[20:150, 30:160] = 1
    pixels[50:120, 60:130] = 0
    pixels[215:225, 260:275] = 1  # disconnected part of the same instance
    return pixels


def fixture(root):
    root.mkdir()
    Image.new("RGB", (300, 260), "white").save(root / "ring.png")
    Image.new("RGB", (300, 260), "white").save(root / "empty.png")
    pixels = ring()
    source = root / "instances.json"
    source.write_text(json.dumps({
        "images": [{"id": 7, "file_name": "ring.png", "width": 300, "height": 260},
                   {"id": 8, "file_name": "empty.png", "width": 300, "height": 260}],
        "categories": [{"id": 5, "name": "ring"}, {"id": 12, "name": "part"}],
        "annotations": [
            {"id": 100, "image_id": 7, "category_id": 5, "iscrowd": 1,
             "bbox": [0, 0, 1, 1], "area": 1,  # geometry is recalculated from RLE
             "segmentation": {"size": [260, 300], "counts": encode_coco_counts(counts_for(pixels))}},
            {"id": 101, "image_id": 7, "category_id": 12, "iscrowd": 0,
             "segmentation": [[170, 25, 230, 25, 230, 85, 170, 85]]},
        ],
    }), encoding="utf-8")
    return source, pixels


def test_sparse_coco_rle_preserves_holes_and_large_canvas():
    pixels = ring()
    counts = counts_for(pixels)
    assert decode_coco_counts(encode_coco_counts(counts), 300, 260) == counts
    mask = TileMask.model_validate(mask_from_coco_counts(counts, 300, 260))
    assert coco_counts_from_mask(mask, 300, 260) == counts
    # A canvas larger than two billion pixels still converts without a full bitmap.
    sparse = [1_500_020_000, 1, 999_979_999]
    native = TileMask.model_validate(mask_from_coco_counts(sparse, 50_000, 50_000))
    assert len(native.tiles) == 1
    assert coco_counts_from_mask(native, 50_000, 50_000) == sparse
    # COCO's 32-bit run format cannot represent this larger canvas.
    large_counts = [200_001, 1, 9_999_599_998, 1, 199_999]
    assert sum(large_counts) == 10_000_000_000  # beyond the COCO uint32 limit
    with pytest.raises(ValueError, match="32-bit"):
        decode_coco_counts(large_counts, 100_000, 100_000)


def test_official_coco_codec_interoperability():
    coco = pytest.importorskip("pycocotools.mask")
    pixels = ring()
    standard = coco.encode(np.asfortranarray(pixels))
    counts = decode_coco_counts(standard["counts"].decode("ascii"), 300, 260)
    assert counts == counts_for(pixels)
    native = TileMask.model_validate(mask_from_coco_counts(counts, 300, 260))
    encoded = {"size": [260, 300], "counts": encode_coco_counts(coco_counts_from_mask(native, 300, 260)).encode("ascii")}
    assert np.array_equal(coco.decode(encoded), pixels)
    assert encoded["counts"] == standard["counts"]


def test_coco_import_reports_bad_rle_without_inventing_boxes(tmp_path):
    source, _ = fixture(tmp_path / "source")
    data = json.loads(source.read_text())
    data["annotations"].append({"id": 102, "image_id": 7, "category_id": 5, "bbox": [0, 0, 10, 10],
                                 "segmentation": {"size": [260, 300], "counts": [1, 2]}})
    source.write_text(json.dumps(data))
    imported = CocoSegmentation().read(source.parent, source, [], "train")
    assert imported.report.skipped_objects == 1
    assert imported.report.object_count == 2
    assert {o.kind for o in imported.images[0].objects} == {"polygon", "mask"}
    assert imported.images[0].objects[0].area == int(ring().sum())
    assert imported.images[0].objects[0].attributes["iscrowd"] == 1
    assert imported.images[1].status == "imported_coarse" and not imported.images[1].objects


def test_uncompressed_rle_and_complex_polygons_import(tmp_path):
    source, _ = fixture(tmp_path / "source")
    data = json.loads(source.read_text())
    data["annotations"] = [
        {"id": 200, "image_id": 7, "category_id": 5,
         "segmentation": {"size": [260, 300], "counts": [0, 3, 300*260-3]}},
        {"id": 201, "image_id": 7, "category_id": 12,
         "segmentation": [[10, 10, 80, 10, 80, 80, 10, 80],
                          [60, 60, 100, 60, 100, 100, 60, 100]]},
    ]
    source.write_text(json.dumps(data))
    restored = CocoSegmentation().read(source.parent, source, [], "train")
    assert restored.report.object_count == 2
    assert all(obj.kind == "mask" for obj in restored.images[0].objects)
    assert restored.images[0].objects[0].area == 3
    assert "polygon_rasterized" in {issue.code for issue in restored.report.issues}


def test_oversized_mask_export_is_blocked_in_preflight():
    mask = mask_from_coco_counts([0, 1, 2_499_999_999], 50_000, 50_000)
    obj = Annotation(kind="mask", id="one", category_id=0, label="ring", mask=mask)
    dataset = Dataset(task="instance_segmentation", categories=[Category(id=0, name="ring")],
        images=[ImageRecord(id="huge", path="huge.png", width=100_000, height=100_000, objects=[obj])])
    issues = CocoSegmentation().preflight(dataset, False)
    assert issues[0]["code"] == "coco_rle_size_limit" and issues[0]["severity"] == "blocking"


def test_import_center_previews_and_commits_coco_segmentation(tmp_path, monkeypatch):
    source, _ = fixture(tmp_path / "source")
    monkeypatch.setattr(server, "store", ProjectStore(tmp_path / "workspace" / "projects"))
    client = TestClient(server.app)
    preview = client.post("/api/dataset-imports/preview", json={
        "name": "Imported COCO masks", "task": "instance_segmentation",
        "sources": [{"id": "batch1", "root": str(source.parent), "format": "coco_segmentation",
                     "annotation_path": str(source), "split": "train"}],
    })
    assert preview.status_code == 200, preview.text
    result = preview.json()
    assert result["summary"]["object_count"] == 2
    assert {category["name"] for category in result["categories"]} == {"ring", "part"}
    commit = client.post(f"/api/dataset-imports/{result['preview_id']}/commit")
    assert commit.status_code == 201, commit.text
    project = commit.json()
    manifest = client.get(f"/api/projects/{project['id']}/dataset").json()
    assert manifest["task"] == "instance_segmentation"
    assert {obj["kind"] for image in manifest["images"] for obj in image["objects"]} == {"mask", "polygon"}


def test_coco_import_export_roundtrip_and_export_center(tmp_path, monkeypatch):
    source, pixels = fixture(tmp_path / "source")
    monkeypatch.setattr(server, "store", ProjectStore(tmp_path / "workspace" / "projects"))
    client = TestClient(server.app)
    capabilities = client.get("/api/dataset-formats").json()
    spec = next(f for f in capabilities if f["id"] == "coco_segmentation")
    assert spec["can_import"] and spec["can_export"] and spec["tasks"] == ["instance_segmentation"]
    response = client.post("/api/projects", json={"name": "COCO segmentation", "task": "instance_segmentation",
        "dataset_path": str(source.parent), "dataset_format": "coco_segmentation", "annotation_path": str(source)})
    assert response.status_code == 201, response.text
    project_id = response.json()["id"]
    base = f"/api/projects/{project_id}"
    imported = client.get(base + "/annotations", params={"image": "ring.png"}).json()
    assert imported["status"] == "imported_coarse" and len(imported["objects"]) == 2
    assert client.put(base + "/annotations", json={"image": "ring.png", "objects": imported["objects"]}).status_code == 200
    assert client.put(base + "/annotations", json={"image": "empty.png", "objects": []}).status_code == 200
    preview = client.post(base + "/dataset/export/preview", json={
        "formats": ["coco_segmentation"], "include_images": True}).json()
    assert not preview["formats"][0]["blocked"]
    assert "孔洞及分离区域" in preview["formats"][0]["preserved"]
    required = [issue["code"] for issue in preview["formats"][0]["issues"] if issue["requires_ack"]]
    started = client.post(base + "/dataset/export/jobs", json={"preview_id": preview["preview_id"],
        "acknowledgements": {"coco_segmentation": required}})
    assert started.status_code == 202, started.text
    job_id = started.json()["job_id"]
    for _ in range(100):
        status = client.get(base + f"/dataset/export/jobs/{job_id}").json()
        if status["status"] not in {"queued", "running"}:
            break
        time.sleep(.05)
    assert status["status"] == "completed", status
    download = client.get(status["formats"]["coco_segmentation"]["report"]["download_url"])
    assert download.status_code == 200
    with ZipFile(io.BytesIO(download.content)) as archive:
        exported = json.loads(archive.read("annotations.json"))
        assert "images/ring.png" in archive.namelist()
        assert len(exported["images"]) == 2
        assert len(exported["annotations"]) == 2
        mask_row = next(a for a in exported["annotations"] if isinstance(a["segmentation"], dict))
        assert mask_row["area"] == int(pixels.sum())
        assert mask_row["iscrowd"] == 1
        assert decode_coco_counts(mask_row["segmentation"]["counts"], 300, 260) == counts_for(pixels)
        assert "annotations/instances_unspecified.json" in archive.namelist()
        destination = tmp_path / "exported"
        archive.extractall(destination)
    restored = CocoSegmentation().read(destination / "images", destination / "annotations.json", [], "val")
    assert restored.report.object_count == 2
    assert restored.images[0].objects[0].mask.model_dump() == imported["objects"][0]["mask"]
    assert restored.images[1].objects == []


@pytest.mark.parametrize("counts", [[0, 10], [1, -1, 78000], [0, 0, 78000], "bad!", "zzzzz"])
def test_invalid_coco_rle_is_rejected(counts):
    with pytest.raises(ValueError):
        decode_coco_counts(counts, 300, 260)
