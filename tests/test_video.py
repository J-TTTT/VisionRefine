"""Manual video workflow contracts; media decoding is tested separately."""
import copy
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from visionrefine.core.store import ProjectStore
from visionrefine.core.dataset_io.service import load_dataset
from visionrefine.core.video import media
from visionrefine.core.video.api import create_video_router
from visionrefine.core.video.models import ProjectInput
from visionrefine.core.video.service import VideoService, atomic_json, ensure_no_active_jobs, VideoConflict


@pytest.fixture
def video_env(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / "workspace" / "projects")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"test video identity")

    def prepare(path, cache, progress=None, cancel=None):
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "preview.mp4").write_bytes(b"preview")
        (cache / "preview.webm").write_bytes(b"webm preview")
        Image.new("RGB", (64, 32), "red").save(cache / "poster.jpg")
        stat = path.stat()
        return dict(cache_version=media.CACHE_VERSION, width=64, height=32, duration=1.0, frame_count=4, fps=4, has_audio=False,
                    timestamps=[0.0, 0.1, 0.4, 0.8], source_pts=[0, 1, 4, 8], time_base=[1, 10],
                    source_signature={"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rotation=0,
                    preview_path=str(cache / "preview.mp4"), preview_webm_path=str(cache / "preview.webm"), poster_path=str(cache / "poster.jpg"))

    def frame(path, cache, metadata, frame_index, cancel=None):
        target = cache / f"frame-{frame_index}.jpg"
        Image.new("RGB", (64, 32), (frame_index * 40, 0, 0)).save(target)
        return target

    monkeypatch.setattr(media, "prepare_video", prepare)
    monkeypatch.setattr(media, "frame_image", frame)
    app = FastAPI()
    app.include_router(create_video_router(lambda: store))
    client = TestClient(app)
    yield client, store, source
    # Do not leave workers using a temporary workspace past fixture cleanup.
    for project in store.list():
        if project["task"] == "video":
            for job in VideoService(store).jobs(project["id"]):
                if job["status"] in {"queued", "running", "cancelling"}:
                    VideoService(store).cancel_job(project["id"], job["id"])
                    wait_job(client, project["id"], job["id"])
    client.close()


def wait_job(client, pid, jid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        jobs = client.get(f"/api/video/projects/{pid}/jobs").json()
        job = next(job for job in jobs if job["id"] == jid)
        if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Video worker did not finish")


def imported(video_env, labels=None, split="unspecified"):
    client, store, source = video_env
    response = client.post("/api/video/projects", json={"name": "Video test", "paths": [str(source)],
                            "labels": labels or [], "split": split})
    assert response.status_code == 201, response.text
    result = response.json()
    pid = result["project"]["id"]
    job = wait_job(client, pid, result["job"]["id"])
    assert job["status"] == "completed", job
    detail = client.get(f"/api/video/projects/{pid}").json()
    return pid, detail["videos"][0]["id"], detail


def annotation_url(pid, vid):
    return f"/api/video/projects/{pid}/videos/{vid}/annotations"


def test_temporal_events_captions_revisions_and_labels(video_env):
    client, store, source = video_env
    pid, vid, detail = imported(video_env, ["走动"])
    label_id = detail["labels"][0]["id"]
    url = annotation_url(pid, vid)
    doc = client.get(url).json()
    doc["events"] = [
        {"id": "e1", "kind": "interval", "start": 0, "end": 0.6, "label_id": label_id, "text": "", "reviewed": True},
        {"id": "e2", "kind": "interval", "start": 0.1, "end": 0.8, "text": "与第一个事件重叠"},
        {"id": "e3", "kind": "point", "start": 0.4, "text": "瞬时事件"},
    ]
    doc["captions"] = [{"id": "c1", "text": "整段视觉描述", "basis": "visual"},
                       {"id": "c2", "start": 0.4, "end": 1, "text": "片段描述", "basis": "audiovisual"}]
    saved = client.put(url, json=doc)
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == 1
    assert saved.json()["review"] == {"events": "unreviewed", "captions": "unreviewed"}
    assert client.put(url, json=doc).status_code == 409
    assert client.get(url).json() == saved.json()
    assert len(list((store.root / pid / "video/revisions" / vid).glob("*.json"))) == 1
    label = detail["labels"][0]
    label["name"] = "行走"
    assert client.put(f"/api/video/projects/{pid}/labels", json={"labels": [label]}).status_code == 200
    assert client.get(url).json()["events"][0]["label_id"] == label_id
    assert client.put(f"/api/video/projects/{pid}/labels", json={"labels": []}).status_code == 409
    assert source.exists()


@pytest.mark.parametrize("event", [
    {"id": "e", "start": 0.1, "end": 1.1, "text": "beyond duration"},
    {"id": "e", "start": 0.5, "end": 0.2, "text": "reversed"},
    {"id": "e", "start": 0.1, "end": 0.2, "label_id": "missing"},
    {"id": "e", "start": 0.1, "end": 0.2},
    {"id": "e", "kind": "point", "start": 0.1, "end": 0.2, "text": "point has duration"},
])
def test_invalid_annotations_never_change_revision(video_env, event):
    client, _, _ = video_env
    pid, vid, _ = imported(video_env)
    url = annotation_url(pid, vid)
    response = client.put(url, json={"revision": 0, "events": [event]})
    assert response.status_code in {400, 422}
    assert client.get(url).json()["revision"] == 0


def test_empty_review_and_native_roundtrip_resets_trust(video_env):
    client, store, _ = video_env
    pid, vid, _ = imported(video_env)
    url = annotation_url(pid, vid)
    doc = {"revision": 0, "events": [], "captions": [{"id": "c", "text": "description", "reviewed": True}],
           "review": {"events": "reviewed", "captions": "reviewed"}}
    assert client.put(url, json=doc).status_code == 200
    response = client.get(f"/api/video/projects/{pid}/export?policy=all")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    bundle = response.json()
    assert str(store.root) not in response.text
    assert bundle["videos"][0]["annotations"]["review"]["events"] == "reviewed"
    assert bundle["videos"][0]["timestamps"] == [0, 0.1, 0.4, 0.8]
    restored = client.post(f"/api/video/projects/{pid}/annotations/import", json=bundle)
    assert restored.status_code == 200, restored.text
    current = client.get(url).json()
    assert current["revision"] == 2
    assert current["captions"][0]["text"] == "description"
    assert not current["captions"][0]["reviewed"]
    assert current["review"]["events"] == "unreviewed"
    assert client.post(f"/api/video/projects/{pid}/annotations/import", json=bundle).status_code == 409
    assert client.post(f"/api/video/projects/{pid}/annotations/import", json={"bundle": bundle, "expected_revisions": {vid: 2}}).status_code == 200


def test_import_validates_all_sources_before_changes(video_env):
    client, _, _ = video_env
    pid, vid, _ = imported(video_env)
    bundle = client.get(f"/api/video/projects/{pid}/export").json()
    bundle["labels"] = [{"id": "new", "name": "new", "color": "#112233"}]
    bundle["videos"].append({**copy.deepcopy(bundle["videos"][0]), "sha256": "not-the-source"})
    result = client.post(f"/api/video/projects/{pid}/annotations/import", json=bundle)
    assert result.status_code == 400
    assert client.get(annotation_url(pid, vid)).json()["revision"] == 0
    assert client.get(f"/api/video/projects/{pid}/labels").json()["labels"] == []


def test_exact_vfr_sampling_extraction_dedup_and_independent_files(video_env):
    client, store, _ = video_env
    pid, vid, _ = imported(video_env, split="val")
    base = f"/api/video/projects/{pid}/videos/{vid}"
    sample = client.post(f"{base}/extract/preview", json={"mode": "interval", "interval": 0.2}).json()
    assert [row["frame_index"] for row in sample["frames"]] == [0, 1, 2, 3]
    assert client.get(f"{base}/frames/4").status_code == 400
    assert client.get(f"{base}/frames/-1").status_code == 400
    payload = {"frame_indices": [1, 3, 1], "task": "detection", "labels": ["person"]}
    job = client.post(f"{base}/extract", json=payload).json()
    job = wait_job(client, pid, job["id"])
    assert job["status"] == "completed", job
    child = store.get(job["result"]["project_id"])
    dataset = load_dataset(store, child)
    assert len(dataset.images) == 2
    assert child["video_source"]["project_id"] == pid
    assert child["dataset_format"] != "images"
    assert dataset.images[0].provenance["video"]["frame_index"] == 1
    assert dataset.images[0].provenance["video"]["timestamp"] == 0.1
    assert all(image.split == "val" for image in dataset.images)
    assert all(Path(child["dataset_path"]).is_relative_to(store.root / child["id"]) for _ in dataset.images)
    again = client.post(f"{base}/extract", json=payload).json()
    again = wait_job(client, pid, again["id"])
    assert again["result"]["project_id"] == child["id"]
    assert again["result"]["added"] == []
    assert again["result"]["skipped"] == [1, 3]
    # Removing video metadata cannot remove the extracted training images.
    store.delete(pid)
    assert all((Path(child["dataset_path"]) / image.path).is_file() for image in dataset.images)


def test_job_cancel_retry_recovery_and_delete_guard(video_env, monkeypatch):
    client, store, source = video_env
    original = media.prepare_video
    entered = threading.Event()

    def waiting(path, cache, progress=None, cancel=None):
        entered.set()
        while not cancel():
            time.sleep(0.01)
        raise media.VideoCancelledError("cancelled")

    monkeypatch.setattr(media, "prepare_video", waiting)
    created = client.post("/api/video/projects", json={"name": "cancel", "paths": [str(source)]}).json()
    pid, jid = created["project"]["id"], created["job"]["id"]
    assert entered.wait(2)
    with pytest.raises(VideoConflict):
        ensure_no_active_jobs(store, pid)
    assert client.post(f"/api/video/projects/{pid}/jobs/{jid}/cancel").status_code == 200
    assert wait_job(client, pid, jid)["status"] == "cancelled"
    monkeypatch.setattr(media, "prepare_video", original)
    retry = client.post(f"/api/video/projects/{pid}/jobs/{jid}/retry").json()
    assert retry["id"] != jid
    assert wait_job(client, pid, retry["id"])["status"] == "completed"
    job_path = store.root / pid / "video/jobs/job-crash.json"
    atomic_json(job_path, {"id": "job-crash", "status": "running", "kind": "import", "payload": {"paths": [str(source)]}, "created_at": "2020"})
    recovered = VideoService(ProjectStore(store.root)).jobs(pid)
    assert next(job for job in recovered if job["id"] == "job-crash")["status"] == "interrupted"
    retried = client.post(f"/api/video/projects/{pid}/jobs/job-crash/retry").json()
    finished = wait_job(client, pid, retried["id"])
    assert finished["status"] == "completed"
    assert len(finished["result"]["skipped"]) == 1


def test_changed_source_blocks_frame_extraction(video_env):
    client, _, source = video_env
    pid, vid, _ = imported(video_env)
    source.write_bytes(b"changed source")
    assert client.get(f"/api/video/projects/{pid}/videos/{vid}/frames/0").status_code == 409


def test_sampling_respects_selected_range_and_explicit_review(video_env):
    client, _, _ = video_env
    pid, vid, _ = imported(video_env)
    base = f"/api/video/projects/{pid}/videos/{vid}"
    response = client.post(f"{base}/extract/preview", json={"mode": "count", "start": 0.11, "end": 0.39, "count": 3})
    assert response.json() == {"frames": [], "count": 0}
    response = client.post(f"{base}/extract/preview", json={"mode": "count", "start": 0.11, "end": 0.41, "count": 3})
    assert response.json() == {"frames": [{"frame_index": 2, "timestamp": 0.4}], "count": 1}
    response = client.put(f"{base}/annotations", json={"events": [{"id": "e", "kind": "point", "start": 0.4, "text": "draft"}],
                                                            "review": {"events": "reviewed"}})
    assert response.status_code == 400
    empty = client.put(f"{base}/annotations", json={"events": [], "review": {"events": "reviewed"}})
    assert empty.status_code == 200
    bundle = client.get(f"/api/video/projects/{pid}/export?policy=reviewed").json()
    assert bundle["videos"][0]["annotations"]["events"] == []
    assert bundle["videos"][0]["annotations"]["review"] == {"events": "reviewed", "captions": "unreviewed"}


def test_cancelled_extraction_has_no_published_frames_or_lost_provenance(video_env):
    from visionrefine.core.video.service import Cancelled
    _, store, _ = video_env
    pid, vid, _ = imported(video_env)
    service = VideoService(store)
    def cancel_after_first_frame(progress):
        raise Cancelled()
    with pytest.raises(Cancelled):
        service._extract(pid, {"video_id": vid, "task": "detection", "labels": ["person"], "frame_indices": [0, 1]},
                         lambda: None, cancel_after_first_frame)
    child_id = store.get(pid)["video_derived_projects"]["detection"]
    child = store.get(child_id)
    assert child["dataset_format"] == "visionrefine"
    assert load_dataset(store, child).images == []
    assert list(Path(child["dataset_path"]).glob("*.jpg")) == []
    assert list((store.root / child_id).glob(".video-extract-*")) == []
    result = service._extract(pid, {"video_id": vid, "task": "detection", "labels": ["person"], "frame_indices": [0, 1]},
                              lambda: None, lambda _: None)
    assert result["project_id"] == child_id
    assert all("video" in record.provenance for record in load_dataset(store, store.get(child_id)).images)


def test_matching_fingerprint_can_relink_a_moved_source(video_env):
    client, _, source = video_env
    pid, vid, _ = imported(video_env)
    moved = source.with_name("moved.mp4")
    source.rename(moved)
    job = client.post(f"/api/video/projects/{pid}/import", json={"paths": [str(moved)]}).json()
    assert wait_job(client, pid, job["id"])["status"] == "completed"
    detail = client.get(f"/api/video/projects/{pid}").json()
    assert len(detail["videos"]) == 1
    assert detail["videos"][0]["id"] == vid
    assert detail["videos"][0]["source_path"] == str(moved)
    assert client.get(f"/api/video/projects/{pid}/videos/{vid}/frames/0").status_code == 200


@pytest.mark.parametrize("mutation", [
    lambda bundle: bundle.update(videos="wrong"),
    lambda bundle: bundle["videos"][0].update(annotations="wrong"),
    lambda bundle: bundle["videos"][0]["annotations"].update(events=["wrong"]),
    lambda bundle: bundle["videos"][0]["annotations"].update(captions=[{}]),
    lambda bundle: bundle.update(labels=[{"id": "bad", "name": "label", "color": "red"}]),
])
def test_malformed_native_bundle_returns_validation_error(video_env, mutation):
    client, _, _ = video_env
    pid, vid, _ = imported(video_env)
    bundle = client.get(f"/api/video/projects/{pid}/export").json()
    mutation(bundle)
    response = client.post(f"/api/video/projects/{pid}/annotations/import", json=bundle)
    assert response.status_code == 400, response.text
    assert client.get(annotation_url(pid, vid)).json()["revision"] == 0


def test_preview_formats_and_missing_cache_reimport_preserve_annotations(video_env):
    client, store, source = video_env
    pid, vid, _ = imported(video_env)
    base = f"/api/video/projects/{pid}/videos/{vid}"
    mp4 = client.get(f"{base}/media")
    webm = client.get(f"{base}/media?format=webm")
    assert mp4.status_code == webm.status_code == 200
    assert mp4.headers["content-type"] == "video/mp4"
    assert webm.headers["content-type"] == "video/webm"
    assert client.get(f"{base}/media?format=invalid").status_code == 422
    document = client.put(f"{base}/annotations", json={"events": [{"id": "e", "kind": "point", "start": 0.1, "text": "keep"}]}).json()
    (store.root / pid / "video/media" / vid / "preview.webm").unlink()
    job = client.post(f"/api/video/projects/{pid}/import", json={"paths": [str(source)]}).json()
    result = wait_job(client, pid, job["id"])
    assert result["status"] == "completed"
    assert result["result"]["video_ids"] == [vid]
    assert client.get(f"{base}/media?format=webm").status_code == 200
    assert client.get(f"{base}/annotations").json() == document
