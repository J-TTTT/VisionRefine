import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import visionrefine.server as server
from visionrefine.core.dataset_io.models import Dataset
from visionrefine.core.dataset_io.service import load_dataset
from visionrefine.core.dataset_io.voc import VocDetection
from visionrefine.core.dataset_io.yolo import YoloDetection
from visionrefine.core.store import ProjectStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "store", ProjectStore(tmp_path / "workspace/projects"))
    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path / "workspace")
    return TestClient(server.app)


def source(root, source_id="a", color="white", label="person", split="train"):
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color).save(root / "same.png")
    (root / "data.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "same.png", "width": 100, "height": 80}],
        "categories": [{"id": 17, "name": label}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 17, "bbox": [10, 10, 20, 30]}],
    }))
    return dict(id=source_id, root=str(root), format="coco_detection", annotation_path=str(root / "data.json"), split=split)


def preview(client, sources, **kwargs):
    response = client.post("/api/dataset-imports/preview", json=dict(sources=sources, **kwargs))
    assert response.status_code == 200, response.text
    return response.json()


def commit(client, plan):
    response = client.post(f"/api/dataset-imports/{plan['preview_id']}/commit")
    assert response.status_code == 201, response.text
    return response.json()


def test_multisource_mapping_preview_and_all_image_routes(client, tmp_path):
    a = source(tmp_path / "train", label="pedestrian")
    b = source(tmp_path / "val", "b", "red", "person", "val")
    a["category_mapping"] = {"pedestrian": "person"}
    plan = preview(client, [a, b], name="Merged")
    assert client.get("/api/projects").json() == []
    assert plan["summary"]["splits"] == dict(train=1, val=1, test=0, unspecified=0)
    assert [c["name"] for c in plan["categories"]] == ["person"]
    project = commit(client, plan)
    assert commit(client, plan)["id"] == project["id"]
    base = f"/api/projects/{project['id']}"
    analyzed = client.post(base + "/analyze").json()
    assert {i["path"] for i in analyzed["analysis"]["images"]} == {"a/same.png", "b/same.png"}
    for path in ("a/same.png", "b/same.png"):
        assert client.get(base + "/preview/" + path).status_code == 200
        assert client.get(base + "/thumbnail/" + path).status_code == 200
        assert client.get(base + "/crop/" + path, params=dict(x=0, y=0, width=64, height=64)).status_code == 200
        annotation = client.get(base + "/annotations", params={"image": path}).json()
        assert annotation["objects"][0]["label"] == "person"
    assert client.get(base + "/preview/same.png").status_code == 404
    assert len(client.get(base + "/dataset/history").json()) == 1
    # Legacy schema remains readable.
    data = client.get(base + "/dataset").json()
    assert data["schema_version"] == "2.0"
    assert len(data["sources"]) == 2
    assert data["images"][0]["provenance"]["source_id"] == 1
    assert data["images"][0]["objects"][0]["provenance"]["source_category_id"] == 17


def test_no_implicit_category_synonyms(client, tmp_path):
    plan = preview(client, [source(tmp_path / "a", label="pedestrian"), source(tmp_path / "b", "b", "red")])
    assert [c["name"] for c in plan["categories"]] == ["pedestrian", "person"]


def test_preview_and_commit_do_not_hash_image_or_annotation_files(client, tmp_path, monkeypatch):
    import visionrefine.core.dataset_io.service as service

    a = source(tmp_path / "a")

    def unexpected_digest(path):
        raise AssertionError(f"Import unexpectedly hashed {path}")

    monkeypatch.setattr(service, "file_digest", unexpected_digest)
    project = commit(client, preview(client, [a]))
    dataset = load_dataset(server.store, project)
    assert dataset.images[0].sha256 is None
    assert dataset.provenance["source_sha256"] is None


def test_multisource_ai_paths_and_manifest_survive_worker_save(client, tmp_path, monkeypatch):
    a, b = source(tmp_path / "a"), source(tmp_path / "b", "b", "red")
    project = commit(client, preview(client, [a, b]))
    base = f"/api/projects/{project['id']}"
    client.post(base + "/analyze")
    response = client.put(base + "/adapter", json=dict(base_url="http://127.0.0.1:8000/v1", model="test"))
    assert response.status_code == 200
    seen = []

    def pilot(path, *args):
        seen.append(path)
        return dict(crop_jpeg=b"test", objects=[], summary="empty")

    def initial(path, *args, **kwargs):
        seen.append(path)
        return dict(tile_count=1, errors=[], objects=[])

    monkeypatch.setattr(server, "run_detection_pilot", pilot)
    monkeypatch.setattr(server, "run_tiled_detection", initial)
    assert client.post(base + "/pilot").status_code == 200
    monkeypatch.setattr(server, "initial_detection_jobs", {"test": {}})
    server._run_initial_detection_job("test", project["id"], "b/same.png")
    assert server.initial_detection_jobs["test"]["status"] == "completed"
    assert seen == [Path(a["root"]) / "same.png", Path(b["root"]) / "same.png"]
    assert client.get(base).json()["dataset_revision"] == project["dataset_revision"]
    assert client.get(base + "/annotations", params={"image": "b/same.png"}).json()["status"] == "ai_suggestion"


def test_append_preserves_human_empty_ai_and_old_category_ids(client, tmp_path):
    first = source(tmp_path / "a")
    project = commit(client, preview(client, [first]))
    base = f"/api/projects/{project['id']}"
    # Preview before saving: a new human revision must not make the preview destructive.
    second = source(tmp_path / "a", "a", "white", "car", "val")
    third = source(tmp_path / "c", "c", "blue", "car", "test")
    plan = preview(client, [second, third], project_id=project["id"], duplicate_policy="update_coarse")
    saved = client.put(base + "/annotations", json={"image": "a/same.png", "objects": []}).json()
    ai = server.suggestion_path(project["id"], "a/same.png")
    ai.parent.mkdir(parents=True)
    ai.write_text(json.dumps(dict(revision_id="ai-old", objects=[])))
    human_bytes = server.annotation_path(project["id"], "a/same.png").read_bytes()
    ai_bytes = ai.read_bytes()
    updated = commit(client, plan)
    assert updated["labels"] == ["person", "car"]
    manifest = load_dataset(server.store, updated)
    assert manifest.images[0].path == "a/same.png"
    assert manifest.images[0].split == "train"
    assert manifest.images[0].objects[0].label == "car"
    assert manifest.images[0].objects[0].category_id == 1
    assert server.annotation_path(project["id"], "a/same.png").read_bytes() == human_bytes
    assert ai.read_bytes() == ai_bytes
    actual = client.get(base + "/annotations", params={"image": "a/same.png"}).json()
    assert actual["revision_id"] == saved["revision_id"] and actual["objects"] == []
    history = client.get(base + "/dataset/history").json()
    assert [r["operation"]["kind"] for r in history] == ["append", "create"]
    assert history[0]["parent_revision_id"] == project["dataset_revision"]
    report = client.post(base + "/dataset/export", json={"format": "coco_detection"}).json()
    assert report["object_count"] == 0 and report["image_count"] == 1
    # An old commit request is idempotent even after later appends.
    assert commit(client, plan)["dataset_revision"] == updated["dataset_revision"]


@pytest.mark.parametrize("policy,count,allowed", [("keep_existing", 0, True), ("update_coarse", 1, True), ("error", 0, False)])
def test_duplicate_policy_and_per_image_resolution(client, tmp_path, policy, count, allowed):
    a = source(tmp_path / "a")
    project = commit(client, preview(client, [a]))
    plan = preview(client, [a], project_id=project["id"], duplicate_policy=policy)
    assert plan["summary"]["image_count"] == 1
    assert plan["summary"]["updated_coarse"] == count
    assert plan["commit_allowed"] == allowed
    if not allowed:
        response = client.post(f"/api/dataset-imports/{plan['preview_id']}/commit")
        assert response.status_code == 400
        assert [p["id"] for p in client.get("/api/projects").json()] == [project["id"]]
        plan = preview(client, [a], project_id=project["id"], duplicate_policy="error",
                       conflict_resolutions={"a/same.png": "keep_existing"})
    commit(client, plan)


@pytest.mark.parametrize("change", ["image", "annotations", "new_file", "removed"])
def test_user_manages_changed_inputs_but_missing_images_cannot_commit(client, tmp_path, change):
    a = source(tmp_path / "a")
    plan = preview(client, [a])
    root = Path(a["root"])
    if change == "image":
        Image.new("RGB", (100, 80), "red").save(root / "same.png")
    elif change == "annotations":
        (root / "data.json").write_text("{}")
    elif change == "new_file":
        (root / "new.xml").write_text("<annotation/>")
    else:
        (root / "same.png").unlink()
    response = client.post(f"/api/dataset-imports/{plan['preview_id']}/commit")
    assert response.status_code == (409 if change == "removed" else 201), response.text
    assert bool(client.get("/api/projects").json()) == (change != "removed")


def test_concurrent_previews_and_changed_source_identity(client, tmp_path):
    a = source(tmp_path / "a")
    project = commit(client, preview(client, [a]))
    b = source(tmp_path / "b", "b", "red")
    plan1 = preview(client, [b], project_id=project["id"])
    plan2 = preview(client, [b], project_id=project["id"])
    commit(client, plan1)
    response = client.post(f"/api/dataset-imports/{plan2['preview_id']}/commit")
    assert response.status_code == 409
    b["id"] = "a"
    response = client.post("/api/dataset-imports/preview", json=dict(project_id=project["id"], sources=[b]))
    assert response.status_code == 400
    assert "Source ID" in response.text


def test_legacy_append_preserves_annotation_keys_and_raw_images_not_negative(client, tmp_path):
    a = source(tmp_path / "a")
    response = client.post("/api/projects", json=dict(name="legacy", task="detection", dataset_path=a["root"],
                           dataset_format=a["format"], annotation_path=a["annotation_path"]))
    project = response.json()
    base = f"/api/projects/{project['id']}"
    client.post(base + "/analyze")
    client.put(base + "/annotations", json=dict(image="same.png", objects=[]))
    b = source(tmp_path / "raw", "raw", "blue")
    b.update(format="images", annotation_path=None, labels=["newclass"], split="test")
    updated = commit(client, preview(client, [b], project_id=project["id"]))
    manifest = load_dataset(server.store, updated)
    assert [i.path for i in manifest.images] == ["same.png", "raw/same.png"]
    assert manifest.images[0].source_id == "legacy"
    assert manifest.categories[0].source_id == 17
    assert client.get(base + "/annotations", params={"image": "same.png"}).json()["status"] == "human_reviewed"
    report = client.post(base + "/dataset/export", json={"format": "yolo_detection", "policy": "all_annotated"}).json()
    assert report["image_count"] == 1
    assert report["skipped_images"] == [{"image": "raw/same.png", "status": "unreviewed"}]


def voc_fixture(root, box="<xmin>1</xmin><ymin>1</ymin><xmax>1</xmax><ymax>1</ymax>"):
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()
    (root / "ImageSets/Main").mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(root / "JPEGImages/test.jpg")
    (root / "ImageSets/Main/val.txt").write_text("test\n")
    (root / "Annotations/test.xml").write_text(f"""<annotation><filename>test.jpg</filename>
      <size><width>100</width><height>80</height></size><object><name>cell</name><difficult>1</difficult>
      <truncated>1</truncated><pose>Left</pose><bndbox>{box}</bndbox></object></annotation>""")
    return root / "Annotations/test.xml"


def test_voc_one_pixel_flags_splits_and_roundtrip(tmp_path):
    root = tmp_path / "voc"
    voc_fixture(root)
    dataset = VocDetection().read(root, None, [], "unspecified")
    record = dataset.images[0]
    assert record.split == "val"
    assert record.objects[0].bbox == [0, 0, 1, 1]
    assert record.objects[0].attributes["difficult"] == 1
    out = tmp_path / "out"
    mapping = VocDetection().write(dataset, out)
    target = out / mapping["image_paths"][record.path]
    target.parent.mkdir()
    target.write_bytes((root / record.path).read_bytes())
    again = VocDetection().read(out, None, [], "unspecified")
    assert again.images[0].objects[0].bbox == [0, 0, 1, 1]
    assert again.images[0].objects[0].attributes == record.objects[0].attributes


@pytest.mark.parametrize("bad", ["<!DOCTYPE annotation [<!ENTITY x 'bad'>]><annotation/>", "<broken", "<annotation><filename>../../secret.jpg</filename></annotation>"])
def test_voc_rejects_unsafe_xml_and_paths_with_report(tmp_path, bad):
    root = tmp_path / "voc"
    voc_fixture(root)
    (root / "Annotations/invalid.xml").write_text(bad)
    dataset = VocDetection().read(root, None, [], "unspecified")
    assert dataset.report.skipped_images == 1
    assert dataset.report.issues[0].severity == "error"
    assert len(dataset.images) == 1


def test_voc_empty_image_classes_and_split_conflicts(tmp_path):
    root = tmp_path / "voc"
    xml = voc_fixture(root)
    xml.write_text("<annotation><filename>test.jpg</filename></annotation>")
    (root / "classes.txt").write_text("cell\nunused\n")
    dataset = VocDetection().read(root, None, [], "unspecified")
    assert dataset.report.empty_images == 1
    assert [c.name for c in dataset.categories] == ["cell", "unused"]
    (root / "ImageSets/Main/train.txt").write_text("test\n")
    with pytest.raises(ValueError, match="multiple splits"):
        VocDetection().read(root, None, [], "unspecified")


def test_multisource_coco_voc_jpeg_yolo_cross_export(client, tmp_path):
    a, b = source(tmp_path / "a"), source(tmp_path / "b", "b", "red", split="val")
    project = commit(client, preview(client, [a, b]))
    base = f"/api/projects/{project['id']}"
    response = client.post(base + "/dataset/export", json={"format": "voc_detection", "policy": "all_annotated", "include_images": True})
    assert response.status_code == 201, response.text
    report = response.json()
    out = tmp_path / "voc"
    with ZipFile(io.BytesIO(client.get(report["download_url"]).content)) as archive:
        archive.extractall(out)
    for path in report["image_paths"].values():
        with Image.open(out / path) as image:
            assert image.format == "JPEG"
    dataset = VocDetection().read(out, None, [], "unspecified")
    assert [i.split for i in dataset.images].count("val") == 1
    assert all(i.objects[0].bbox == [10, 10, 30, 40] for i in dataset.images)
    voc_project = client.post("/api/projects", json=dict(name="VOC", task="detection", dataset_path=str(out),
                            dataset_format="voc_detection", annotation_path=str(out / "Annotations"))).json()
    response = client.post(f"/api/projects/{voc_project['id']}/dataset/export", json={"format": "yolo_detection", "policy": "all_annotated", "include_images": True})
    assert response.status_code == 201, response.text
    yolo_root = tmp_path / "yolo"
    with ZipFile(io.BytesIO(client.get(response.json()["download_url"]).content)) as archive:
        archive.extractall(yolo_root)
    yolo = YoloDetection().read(yolo_root, yolo_root / "data.yaml", [], "unspecified")
    assert len(yolo.images) == 2
    assert all(i.objects[0].bbox == pytest.approx([10, 10, 30, 40]) for i in yolo.images)


def test_manifest_v1_is_readable():
    assert Dataset.model_validate({"schema_version": "1.0"}).schema_version == "1.0"


def test_image_only_category_change_invalidates_append_preview(client, tmp_path):
    a = source(tmp_path / "a")
    a.update(format="images", annotation_path=None, labels=["person"])
    project = commit(client, preview(client, [a]))
    assert not project["labels_locked"]
    b = source(tmp_path / "b", "b", "red")
    plan = preview(client, [b], project_id=project["id"])
    response = client.put(f"/api/projects/{project['id']}/labels", json={"labels": ["car"]})
    assert response.status_code == 200
    assert client.post(f"/api/dataset-imports/{plan['preview_id']}/commit").status_code == 409


def test_equal_file_bytes_from_different_sources_are_not_matched(client, tmp_path):
    a = source(tmp_path / "a")
    project = commit(client, preview(client, [a]))
    raw = dict(id="raw", root=a["root"], format="images", labels=["person"])
    plan = preview(client, [raw], project_id=project["id"], duplicate_policy="update_coarse")
    assert plan["conflicts"] == []
    assert plan["summary"]["image_count"] == 2
    assert plan["summary"]["object_count"] == 1
    commit(client, plan)
    Image.new("RGB", (100, 80), "blue").save(Path(a["root"]) / "same.png")
    raw["id"] = "raw2"
    response = client.post("/api/dataset-imports/preview", json=dict(project_id=project["id"], sources=[raw]))
    assert response.status_code == 200, response.text
    assert response.json()["summary"]["image_count"] == 3


@pytest.mark.parametrize("mapping", [{"unknown": "person"}, {"person": "  "}])
def test_invalid_mapping_never_creates_a_project(client, tmp_path, mapping):
    a = source(tmp_path / "a")
    a["category_mapping"] = mapping
    response = client.post("/api/dataset-imports/preview", json={"sources": [a]})
    assert response.status_code in {400, 422}
    assert client.get("/api/projects").json() == []


def test_voc_fractional_edges_expanded_and_invalid_flags_reported(tmp_path):
    root = tmp_path / "voc"
    xml = voc_fixture(root)
    dataset = VocDetection().read(root, None, [], "train")
    dataset.images[0].objects[0].bbox = [0.5, 0.25, 1.2, 2.1]
    out = tmp_path / "out"
    report = VocDetection().write(dataset, out)
    text = next((out / "Annotations").glob("*.xml")).read_text()
    assert "<xmin>1</xmin>" in text and "<xmax>2</xmax>" in text
    assert "<ymax>3</ymax>" in text
    assert any("fractional" in w for w in report["warnings"])
    xml.write_text(xml.read_text().replace("<difficult>1</difficult>", "<difficult>3</difficult>"))
    dataset = VocDetection().read(root, None, ["cell"], "train")
    assert dataset.report.skipped_objects == 1 and dataset.report.empty_images == 1
