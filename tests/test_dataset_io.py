import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

import visionrefine.server as server
from visionrefine.core.dataset_io import registry
from visionrefine.core.dataset_io.coco import CocoDetection
from visionrefine.core.dataset_io.models import Dataset, safe_relative_path
from visionrefine.core.dataset_io.service import load_dataset
from visionrefine.core.dataset_io.yolo import YoloDetection, label_path
from visionrefine.core.store import ProjectStore


def image(root, name, size=(100, 80)):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(target)
    return target


def json_file(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def coco_fixture(root):
    image(root, "nested/图像.jpg")
    image(root, "negative.png")
    return json_file(root / "input.json", {
        "images": [{"id": 11, "file_name": "nested/图像.jpg", "width": 100, "height": 80},
                   {"id": 12, "file_name": "negative.png", "width": 100, "height": 80}],
        "categories": [{"id": 7, "name": "自定义细胞"}, {"id": 91, "name": "defect"}],
        "annotations": [{"id": 70, "image_id": 11, "category_id": 7, "bbox": [10, 20, 30, 40], "iscrowd": 1}],
    })


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "store", ProjectStore(tmp_path / "workspace" / "projects"))
    return TestClient(server.app)


def create_import(client, root, source, format_id="coco_detection", split="train"):
    response = client.post("/api/projects", json={
        "name": "Dataset I/O", "task": "detection", "dataset_path": str(root),
        "annotation_path": str(source), "dataset_format": format_id, "split": split,
    })
    assert response.status_code == 201, response.text
    project = response.json()
    response = client.post(f"/api/projects/{project['id']}/analyze")
    assert response.status_code == 200, response.text
    return response.json()


def unpack(client, project_id, payload, destination):
    response = client.post(f"/api/projects/{project_id}/dataset/export", json=payload)
    assert response.status_code == 201, response.text
    report = response.json()
    download = client.get(report["download_url"])
    assert download.status_code == 200
    with ZipFile(io.BytesIO(download.content)) as archive:
        assert "export-report.json" in archive.namelist()
        assert "visionrefine.json" in archive.namelist()
        assert all(not p.startswith("/") and ".." not in Path(p).parts for p in archive.namelist())
        archive.extractall(destination)
    return report


def test_cross_format_review_restart_and_roundtrip(client, tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    source = coco_fixture(root)
    original = source.read_bytes()
    project = create_import(client, root, source)
    base = f"/api/projects/{project['id']}"
    loaded = client.get(base + "/annotations", params={"image": "nested/图像.jpg"}).json()
    assert loaded["status"] == "imported_coarse"
    assert loaded["objects"][0]["bbox"] == [10, 20, 40, 60]
    assert project["labels"] == ["自定义细胞", "defect"]
    loaded["objects"][0].update(label="defect", bbox=[12, 15, 60, 70])
    saved = client.put(base + "/annotations", json={"image": "nested/图像.jpg", "objects": loaded["objects"]})
    assert saved.status_code == 200
    assert client.put(base + "/annotations", json={"image": "negative.png", "objects": []}).status_code == 200
    # Recreate store/client; no runtime job dictionary is needed to restore annotations.
    monkeypatch.setattr(server, "store", ProjectStore(server.store.root))
    client = TestClient(server.app)
    assert client.get(base + "/annotations", params={"image": "negative.png"}).json()["status"] == "human_reviewed"
    yolo_dir = tmp_path / "yolo-export"
    report = unpack(client, project["id"], {"format": "yolo_detection", "include_images": True}, yolo_dir)
    assert (report["image_count"], report["object_count"]) == (2, 1)
    yolo = YoloDetection().read(yolo_dir, yolo_dir / "data.yaml", [], "unspecified")
    assert len(yolo.images) == 2
    obj = next(i.objects[0] for i in yolo.images if i.objects)
    assert obj.label == "defect"
    assert obj.bbox == pytest.approx([12, 15, 60, 70])
    assert all(i.split == "train" for i in yolo.images)
    second = create_import(client, yolo_dir, yolo_dir / "data.yaml", "yolo_detection")
    coco_dir = tmp_path / "coco-export"
    unpack(client, second["id"], {"format": "coco_detection", "policy": "all_annotated", "include_images": True}, coco_dir)
    again = CocoDetection().read(coco_dir / "images", coco_dir / "annotations/instances_train.json", [], "train")
    obj = next(i.objects[0] for i in again.images if i.objects)
    assert obj.label == "defect"
    assert obj.bbox == pytest.approx([12, 15, 60, 70])
    assert len(again.images) == 2
    assert source.read_bytes() == original
    assert load_dataset(server.store, project).images[0].objects[0].label == "自定义细胞"


def test_revision_precedence_and_export_defaults(client, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    project = create_import(client, root, coco_fixture(root))
    base = f"/api/projects/{project['id']}"
    assert client.post(base + "/dataset/export", json={"format": "coco_detection"}).status_code == 400
    suggestion = server.suggestion_path(project["id"], "nested/图像.jpg")
    suggestion.parent.mkdir(parents=True)
    json_file(suggestion, {"revision_id": "ai-test", "objects": [{"label": "defect", "bbox": [1, 2, 3, 4]}]})
    assert client.get(base + "/annotations", params={"image": "nested/图像.jpg"}).json()["status"] == "ai_suggestion"
    report = unpack(client, project["id"], {"format": "coco_detection", "policy": "reviewed_or_ai"}, tmp_path / "ai")
    assert report["image_count"] == 1
    assert len(report["skipped_images"]) == 1
    client.put(base + "/annotations", json={"image": "nested/图像.jpg", "objects": []})
    report = unpack(client, project["id"], {"format": "coco_detection"}, tmp_path / "human")
    assert (report["image_count"], report["object_count"]) == (1, 0)
    assert report["revisions"][0]["status"] == "human_reviewed"


def test_coco_import_report_is_partial_and_ids_preserved(tmp_path):
    source = coco_fixture(tmp_path)
    data = json.loads(source.read_text())
    data["categories"].append({"id": 99, "name": "defect"})
    data["images"].append({"id": 13, "file_name": "missing.jpg"})
    data["annotations"] += [
        {"id": 71, "image_id": 11, "category_id": 99, "bbox": [-10, -10, 30, 30], "segmentation": [[0, 0, 1, 1, 2, 2]]},
        {"id": 72, "image_id": 11, "category_id": 999, "bbox": [0, 0, 1, 1]},
        {"id": 73, "image_id": 11, "category_id": 7, "bbox": [1, 1, -2, 3]},
        {"id": 70, "image_id": 11, "category_id": 7, "bbox": [1, 1, 2, 3]},
    ]
    json_file(source, data)
    dataset = CocoDetection().read(tmp_path, source, [], "val")
    assert dataset.report.skipped_images == 1
    assert dataset.report.skipped_objects == 3
    assert dataset.report.object_count == 2
    assert {"duplicate_category_name", "clipped_bbox", "unsupported_fields"} <= {i.code for i in dataset.report.issues}
    assert dataset.images[0].objects[1].bbox == [0, 0, 20, 20]
    output = tmp_path / "export"
    output.mkdir()
    CocoDetection().write(dataset, output)
    exported = json.loads((output / "annotations.json").read_text())
    assert [c["id"] for c in exported["categories"]] == [7, 91]
    assert exported["annotations"][0]["iscrowd"] == 1


@pytest.mark.parametrize("row", ["0 .5 .5 0 .4", "0 nan .5 .2 .4", "7 .5 .5 .2 .4", "0 .1 .1 .2 .2 .3 .3", "0 1.5 .5 .2 .4"])
def test_yolo_rejects_invalid_rows_with_report(tmp_path, row):
    image(tmp_path, "images/train/a.jpg")
    (tmp_path / "labels/train").mkdir(parents=True)
    (tmp_path / "labels/train/a.txt").write_text(row + "\n0 .5 .5 .2 .4\n")
    source = tmp_path / "data.yaml"
    source.write_text("train: images/train\nnames: [custom]\n")
    dataset = YoloDetection().read(tmp_path, source, [], "unspecified")
    assert dataset.report.skipped_objects == 1
    assert dataset.report.object_count == 1


def test_yolo_split_lists_empty_images_and_category_dictionary(tmp_path):
    image(tmp_path, "images/train/a.jpg")
    image(tmp_path, "images/val/空.png")
    (tmp_path / "labels/train").mkdir(parents=True)
    (tmp_path / "labels/train/a.txt").write_text("0 .5 .5 .2 .4\n")
    (tmp_path / "train.txt").write_text("./images/train/a.jpg\n")
    source = tmp_path / "data.yaml"
    source.write_text("path: .\ntrain: train.txt\nval: [images/val]\nnames: {0: custom}\n")
    dataset = YoloDetection().read(tmp_path, source, [], "unspecified")
    assert [i.split for i in dataset.images] == ["train", "val"]
    assert dataset.report.empty_images == 1
    assert "missing_label_file" in {i.code for i in dataset.report.issues}


def test_more_than_200_images_and_images_only_export(client, tmp_path):
    root = tmp_path / "images"
    for index in range(205):
        image(root, f"{index:03}.png", (10, 10))
    project = client.post("/api/projects", json={"name": "205", "task": "detection", "dataset_path": str(root), "labels": ["custom"]}).json()
    base = f"/api/projects/{project['id']}"
    analysis = client.post(base + "/analyze").json()["analysis"]
    assert not analysis["truncated"]
    assert len(client.get(base + "/images").json()) == 205
    client.put(base + "/annotations", json={"image": "204.png", "objects": [{"label": "custom", "bbox": [1, 1, 5, 5]}]})
    report = unpack(client, project["id"], {"format": "coco_detection"}, tmp_path / "out")
    assert report["image_count"] == 1
    assert report["object_count"] == 1
    assert len(report["skipped_images"]) == 204


def test_split_filter_and_no_image_copy_by_default(client, tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    project = create_import(client, root, coco_fixture(root), split="test")
    base = f"/api/projects/{project['id']}"
    assert client.post(base + "/dataset/export", json={"format": "coco_detection", "policy": "all_annotated", "splits": ["val"]}).status_code == 400
    output = tmp_path / "out"
    report = unpack(client, project["id"], {"format": "coco_detection", "policy": "all_annotated", "splits": ["test"]}, output)
    assert report["image_count"] == 2
    assert not (output / "images").exists()
    assert (output / "annotations/instances_test.json").exists()


@pytest.mark.parametrize("path", ["../outside.jpg", "/absolute.jpg", "dir/../../out.jpg", "C:\\img.jpg", "", ".", "a\nb.jpg"])
def test_relative_paths_cannot_escape(path):
    with pytest.raises(ValueError):
        safe_relative_path(path)


def test_import_rejects_symlink_escape(client, tmp_path):
    outside = image(tmp_path, "outside.png")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked.png").symlink_to(outside)
    source = json_file(root / "input.json", {"images": [{"id": 1, "file_name": "linked.png"}], "categories": [{"id": 1, "name": "x"}], "annotations": []})
    response = client.post("/api/projects", json={"name": "bad", "task": "detection", "dataset_path": str(root), "annotation_path": str(source), "dataset_format": "coco_detection"})
    assert response.status_code == 400
    assert server.store.list() == []


def test_unsupported_task_or_format_fails_explicitly(client, tmp_path):
    response = client.post("/api/projects", json={"name": "bad", "task": "instance_segmentation", "dataset_path": str(tmp_path), "dataset_format": "coco_detection"})
    assert response.status_code == 400
    with pytest.raises(ValueError):
        registry.get("cvat", "detection", "exporter")
    formats = client.get("/api/dataset-formats").json()
    assert {f["id"] for f in formats if f["can_export"]} == {"coco_detection", "yolo_detection", "voc_detection", "cvat_detection", "label_studio_detection", "labelme_detection", "visionrefine"}


def test_import_cannot_replace_human_or_imported_versions(client, tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    source = coco_fixture(root)
    project = create_import(client, root, source)
    response = client.post(f"/api/projects/{project['id']}/dataset/import", json={"format": "coco_detection", "annotation_path": str(source)})
    assert response.status_code == 409


def test_yolo_same_stem_export_does_not_overwrite(tmp_path):
    image(tmp_path, "same.jpg")
    image(tmp_path, "same.png")
    source = json_file(tmp_path / "in.json", {"images": [{"id": 1, "file_name": "same.jpg", "width": 100, "height": 80}, {"id": 2, "file_name": "same.png", "width": 100, "height": 80}], "categories": [{"id": 1, "name": "x"}], "annotations": []})
    dataset = CocoDetection().read(tmp_path, source, [], "train")
    output = tmp_path / "out"
    output.mkdir()
    report = YoloDetection().write(dataset, output)
    assert len(set(report["image_paths"].values())) == 2
    assert len(list((output / "labels/train").glob("*.txt"))) == 2


def test_yolo_download_not_executed_and_unsafe_yaml_rejected(tmp_path):
    image(tmp_path, "images/train/a.jpg")
    source = tmp_path / "data.yaml"
    source.write_text("train: images/train\nnames: [x]\ndownload: echo should-never-run\n")
    dataset = YoloDetection().read(tmp_path, source, [], "unspecified")
    assert "download_ignored" in {i.code for i in dataset.report.issues}
    source.write_text("!!python/object/apply:os.system ['echo unsafe']")
    with pytest.raises(ValueError):
        YoloDetection().read(tmp_path, source, [], "unspecified")


def test_yolo_reexport_images_components_keep_label_association(client, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    image(root, "images/train/nested/images/a.jpg")
    # The last images component is replaced, following YOLO conventions.
    labels = root / "images/train/nested/labels/a.txt"
    labels.parent.mkdir(parents=True)
    labels.write_text("0 .5 .5 .2 .4\n")
    source = root / "data.yaml"
    source.write_text("train: images/train\nnames: [custom]\n")
    project = create_import(client, root, source, "yolo_detection")
    output = tmp_path / "out"
    report = unpack(client, project["id"], {"format": "yolo_detection", "policy": "all_annotated", "include_images": True}, output)
    destination = next(iter(report["image_paths"].values()))
    assert label_path(output, destination).is_file()
    again = YoloDetection().read(output, output / "data.yaml", [], "unspecified")
    assert again.report.object_count == 1
    assert again.images[0].objects[0].bbox == pytest.approx([40, 24, 60, 56])


def test_save_fractional_box_and_revision_parent(client, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    project = create_import(client, root, coco_fixture(root))
    response = client.put(f"/api/projects/{project['id']}/annotations", json={"image": "nested/图像.jpg", "objects": [{"label": "defect", "bbox": [1, 2, 1.25, 2.5]}]})
    assert response.status_code == 200
    assert response.json()["parent_revision_id"] == project["dataset_revision"]
    assert response.json()["objects"][0]["bbox"] == [1, 2, 1.25, 2.5]


def test_attach_annotations_to_untouched_images_project(client, tmp_path):
    source = coco_fixture(tmp_path)
    project = client.post("/api/projects", json={"name": "raw", "task": "detection", "dataset_path": str(tmp_path), "labels": ["initial"]}).json()
    base = f"/api/projects/{project['id']}"
    assert client.post(base + "/analyze").status_code == 200
    response = client.post(base + "/dataset/import", json={"format": "coco_detection", "annotation_path": str(source)})
    assert response.status_code == 200, response.text
    assert response.json()["labels"] == ["自定义细胞", "defect"]
    assert client.get(base + "/annotations", params={"image": "nested/图像.jpg"}).json()["status"] == "imported_coarse"


def test_yolo_conflicting_splits_fail(tmp_path):
    image(tmp_path, "images/a.jpg")
    source = tmp_path / "data.yaml"
    source.write_text("train: images\nval: images\nnames: [x]\n")
    with pytest.raises(ValueError, match="multiple splits"):
        YoloDetection().read(tmp_path, source, [], "unspecified")


def test_yolo_label_collision_on_import_fails(tmp_path):
    image(tmp_path, "images/a.jpg")
    image(tmp_path, "images/a.png")
    source = tmp_path / "data.yaml"
    source.write_text("train: images\nnames: [x]\n")
    with pytest.raises(ValueError, match="Ambiguous label"):
        YoloDetection().read(tmp_path, source, [], "unspecified")
