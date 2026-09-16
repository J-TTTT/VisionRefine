from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import visionrefine.server as server
from visionrefine.core.store import ProjectStore


def test_human_annotations_are_separate_and_label_bounded(tmp_path: Path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (640, 480), "white").save(images / "sample.jpg")
    store = ProjectStore(tmp_path / "workspace" / "projects")
    monkeypatch.setattr(server, "store", store)
    project = store.create({
        "name": "Annotation test", "task": "detection",
        "labels": ["person", "vehicle"], "dataset_path": str(images),
        "model_max_side": 1536,
    })
    client = TestClient(server.app)
    response = client.put(f"/api/projects/{project['id']}/annotations", json={
        "image": "sample.jpg",
        "objects": [{"label": "person", "bbox": [10, 20, 50, 90]}],
    })
    assert response.status_code == 200
    assert response.json()["objects"][0]["source"] == "human_reviewed"
    assert (images / "sample.jpg").exists()
    reloaded = client.get(f"/api/projects/{project['id']}/annotations", params={"image": "sample.jpg"})
    assert reloaded.status_code == 200
    assert reloaded.json() == response.json()

    invalid = client.put(f"/api/projects/{project['id']}/annotations", json={
        "image": "sample.jpg",
        "objects": [{"label": "unknown", "bbox": [10, 20, 50, 90]}],
    })
    assert invalid.status_code == 400


def test_ai_suggestion_is_loaded_until_human_review_exists(tmp_path: Path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (640, 480), "white").save(images / "sample.jpg")
    store = ProjectStore(tmp_path / "workspace" / "projects")
    monkeypatch.setattr(server, "store", store)
    project = store.create({
        "name": "Suggestion test", "task": "detection", "labels": ["person"],
        "dataset_path": str(images), "model_max_side": 1536,
    })
    path = server.suggestion_path(project["id"], "sample.jpg")
    path.parent.mkdir(parents=True)
    path.write_text('{"revision_id":"initial-test","objects":[{"label":"person","bbox":[1,2,3,4]}]}')
    client = TestClient(server.app)
    response = client.get(f"/api/projects/{project['id']}/annotations", params={"image": "sample.jpg"})
    assert response.status_code == 200
    assert response.json()["status"] == "ai_suggestion"
    assert response.json()["objects"][0]["bbox"] == [1, 2, 3, 4]


def test_locked_label_schema_cannot_be_changed(tmp_path: Path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64), "white").save(images / "sample.jpg")
    store = ProjectStore(tmp_path / "workspace" / "projects")
    monkeypatch.setattr(server, "store", store)
    project = store.create({
        "name": "Locked labels", "task": "detection", "labels": ["person"],
        "labels_locked": True, "dataset_path": str(images), "model_max_side": 1536,
    })
    response = TestClient(server.app).put(f"/api/projects/{project['id']}/labels", json={"labels": ["car"]})
    assert response.status_code == 409


def test_detection_project_rejects_arbitrary_labels(tmp_path: Path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    store = ProjectStore(tmp_path / "workspace" / "projects")
    monkeypatch.setattr(server, "store", store)
    response = TestClient(server.app).post("/api/projects", json={
        "name": "Invalid detection label",
        "task": "detection",
        "labels": ["invented-object"],
        "dataset_path": str(images),
        "model_max_side": 1536,
    })
    assert response.status_code == 400
