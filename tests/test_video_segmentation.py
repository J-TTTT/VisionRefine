"""Sparse video segmentation, explicit review coverage and portable lineage."""
import copy
import json

import pytest

from test_video import video_env, imported, wait_job
from visionrefine.core.dataset_io.service import load_dataset
from visionrefine.core.video.service import VideoService


def polygon():
    return {"kind": "polygon", "polygons": [[[2, 2], [20, 2], [20, 20], [2, 20]]], "mask": None}


def ring():
    pixels = [0] * (128 * 128)
    for y in range(2, 8):
        for x in range(2, 8):
            if x in (2, 7) or y in (2, 7):
                pixels[y * 128 + x] = 1
    counts, bit, count = [], 0, 0
    for value in pixels:
        if value == bit:
            count += 1
        else:
            counts.append(count)
            bit, count = value, 1
    counts.append(count)
    return {"kind": "mask", "polygons": None, "mask": {
        "encoding": "tile-rle-row-v1", "tile_size": 128, "tiles": [{"x": 0, "y": 0, "counts": counts}]}}


def track(id="person-1", frames=(0, 3), reviewed=False):
    return {"id": id, "name": "人物一", "label_id": "object-person", "start_frame": 0, "end_frame": 3,
            "keyframes": [{"frame_index": frame, "visibility": "visible", "geometry": polygon(), "reviewed": reviewed}
                          for frame in frames], "visibility_ranges": [], "reviewed": False}


@pytest.fixture
def environment(video_env):
    client, store, _ = video_env
    pid, vid, _ = imported(video_env)
    assert client.put(f"/api/video/projects/{pid}/object-labels", json={"labels": [
        {"id": "object-person", "name": "person", "color": "#22aabb"}]}).status_code == 200
    return client, store, pid, vid, f"/api/video/projects/{pid}/videos/{vid}"


def save(client, base, tracks, **fields):
    current = client.get(f"{base}/segmentation").json()
    response = client.put(f"{base}/segmentation", json={**current, "tracks": tracks, **fields})
    assert response.status_code == 200, response.text
    return response.json()


def test_sparse_masks_roundtrip_assets_and_separate_revisions(environment):
    client, store, pid, vid, base = environment
    shape = track()
    shape["keyframes"][0]["geometry"] = ring()
    saved = save(client, base, [shape])
    assert saved["revision"] == 1
    assert [key["frame_index"] for key in saved["tracks"][0]["keyframes"]] == [0, 3]
    assert saved["tracks"][0]["keyframes"][0]["geometry"] == ring()
    assert client.get(f"{base}/segmentation").json() == saved
    assert client.get(f"{base}/annotations").json()["revision"] == 0
    assert len(list((store.root / pid / "video/mask_assets").glob("*.json.gz"))) == 1
    snapshot = next((store.root / pid / "video/segmentation_revisions" / vid).glob("*.json"))
    assert '"mask_ref"' in snapshot.read_text()
    assert '"counts"' not in snapshot.read_text()
    assert client.put(f"{base}/segmentation", json={**saved, "revision": 0}).status_code == 409
    assert client.put(f"{base}/annotations", json={"captions": [{"id": "caption", "text": "description"}]}).status_code == 200
    assert client.get(f"{base}/segmentation").json() == saved
    reloaded = VideoService(type(store)(store.root))
    assert reloaded.segmentation(pid, vid) == saved
    assert reloaded.annotations(pid, vid)["captions"][0]["text"] == "description"


@pytest.mark.parametrize("mutation", [
    lambda row: row["keyframes"][0].update(geometry={"kind": "polygon", "polygons": [[[0, 0], [65, 0], [10, 10]]]}),
    lambda row: row["keyframes"][0].update(geometry={"kind": "polygon", "polygons": [[[0, 0], [10, 10], [0, 10], [10, 0]]]}),
    lambda row: row["keyframes"][0].update(geometry=None),
    lambda row: row["keyframes"][0].update(visibility="occluded"),
    lambda row: row["keyframes"].append(copy.deepcopy(row["keyframes"][0])),
    lambda row: row["visibility_ranges"].append({"start_frame": 0, "end_frame": 1, "visibility": "outside"}),
    lambda row: row.update(end_frame=4),
    lambda row: row.update(label_id="missing"),
])
def test_invalid_shapes_or_conflicting_states_never_persist(environment, mutation):
    client, _, _, _, base = environment
    row = track()
    mutation(row)
    response = client.put(f"{base}/segmentation", json={"tracks": [row]})
    assert response.status_code in (400, 422), response.text
    assert client.get(f"{base}/segmentation").json()["revision"] == 0


def test_review_requires_every_frame_not_just_endpoint_keyframes(environment):
    client, _, _, _, base = environment
    row = track(reviewed=True)
    assert client.put(f"{base}/segmentation", json={"tracks": [{**row, "reviewed": True}]}).status_code == 400
    assert client.put(f"{base}/segmentation", json={"tracks": [row], "reviewed_ranges": [{"start_frame": 0, "end_frame": 3}]}).status_code == 400
    row["visibility_ranges"] = [{"start_frame": 1, "end_frame": 2, "visibility": "occluded", "reviewed": True}]
    row["reviewed"] = True
    result = save(client, base, [row], reviewed_ranges=[{"start_frame": 0, "end_frame": 3}])
    assert result["tracks"][0]["reviewed"]
    assert result["reviewed_ranges"] == [{"start_frame": 0, "end_frame": 3}]


def test_object_labels_independent_and_referenced_deletion_rejected(environment):
    client, _, pid, _, base = environment
    save(client, base, [track()])
    assert client.put(f"/api/video/projects/{pid}/object-labels", json={"labels": []}).status_code == 409
    assert client.get(f"/api/video/projects/{pid}/labels").json()["labels"] == []
    renamed = client.put(f"/api/video/projects/{pid}/object-labels", json={"labels": [
        {"id": "object-person", "name": "human", "color": "#221199"}]})
    assert renamed.status_code == 200
    response = client.put(f"{base}/annotations", json={"events": [
        {"id": "event", "kind": "point", "start": 0, "text": "enter", "track_ids": ["person-1"]}]})
    assert response.status_code == 200
    current = client.get(f"{base}/segmentation").json()
    assert client.put(f"{base}/segmentation", json={**current, "tracks": []}).status_code == 409
    temporal = client.get(f"{base}/annotations").json()
    temporal["events"][0]["track_ids"] = ["missing"]
    assert client.put(f"{base}/annotations", json=temporal).status_code == 400


def test_split_merge_delete_ranges_keep_event_refs_atomic(environment):
    client, _, _, _, base = environment
    row = track(frames=(0, 3))
    row["visibility_ranges"] = [{"start_frame": 1, "end_frame": 2, "visibility": "occluded", "reviewed": True}]
    saved = save(client, base, [row])
    events = [
        {"id": "ending", "kind": "interval", "start": 0, "end": 0.4, "text": "before", "track_ids": ["person-1"]},
        {"id": "starting", "kind": "interval", "start": 0.4, "end": 0.8, "text": "after", "track_ids": ["person-1"]},
        {"id": "spanning", "kind": "interval", "start": 0, "end": 0.8, "text": "both", "track_ids": ["person-1"]},
        {"id": "point", "kind": "point", "start": 0.4, "text": "at boundary", "track_ids": ["person-1"]},
    ]
    temporal = client.put(f"{base}/annotations", json={"events": events}).json()
    split = client.post(f"{base}/segmentation/operations", json={"revision": saved["revision"],
        "annotations_revision": temporal["revision"], "operation": "split", "track_id": "person-1", "frame_index": 2, "new_track_id": "person-2"})
    assert split.status_code == 200, split.text
    response = split.json()
    refs = {event["id"]: event["track_ids"] for event in response["annotations"]["events"]}
    assert refs == {"ending": ["person-1"], "starting": ["person-2"], "spanning": ["person-1", "person-2"], "point": ["person-2"]}
    assert response["segmentation"]["tracks"][0]["visibility_ranges"][0]["end_frame"] == 1
    assert response["segmentation"]["tracks"][1]["visibility_ranges"][0]["start_frame"] == 2
    assert client.get(f"{base}/annotations").json() == response["annotations"]
    merge = client.post(f"{base}/segmentation/operations", json={"revision": response["segmentation"]["revision"],
        "annotations_revision": response["annotations"]["revision"], "operation": "merge", "track_id": "person-1", "other_track_id": "person-2"})
    assert merge.status_code == 200, merge.text
    response = merge.json()
    assert all(event["track_ids"] == ["person-1"] for event in response["annotations"]["events"])
    assert len(response["segmentation"]["tracks"]) == 1
    deletion = client.post(f"{base}/segmentation/operations", json={"revision": response["segmentation"]["revision"],
        "operation": "delete_range", "track_id": "person-1", "start_frame": 1, "end_frame": 2})
    assert deletion.status_code == 200
    assert deletion.json()["segmentation"]["tracks"][0]["visibility_ranges"] == []
    assert [key["frame_index"] for key in deletion.json()["segmentation"]["tracks"][0]["keyframes"]] == [0, 3]


def test_conflicting_merge_and_stale_event_revision_rejected(environment):
    client, _, _, _, base = environment
    saved = save(client, base, [track(), track(id="other")])
    response = client.post(f"{base}/segmentation/operations", json={"revision": saved["revision"],
        "operation": "merge", "track_id": "person-1", "other_track_id": "other"})
    assert response.status_code == 400
    response = client.post(f"{base}/segmentation/operations", json={"revision": saved["revision"], "annotations_revision": 20,
        "operation": "split", "track_id": "person-1", "frame_index": 2})
    assert response.status_code == 409
    assert client.get(f"{base}/segmentation").json() == saved


def test_native_v2_masks_links_review_reset_and_prevalidation(environment):
    client, store, pid, vid, base = environment
    row = track(frames=(0,), reviewed=True)
    row["keyframes"][0]["geometry"] = ring()
    save(client, base, [row], reviewed_ranges=[{"start_frame": 0, "end_frame": 0}])
    assert client.put(f"{base}/annotations", json={"events": [{"id": "event", "kind": "point", "start": 0,
        "text": "test", "track_ids": ["person-1"]}]}).status_code == 200
    bundle = client.get(f"/api/video/projects/{pid}/export").json()
    assert bundle["schema_version"] == "2.0"
    assert str(store.root) not in json.dumps(bundle)
    assert bundle["videos"][0]["segmentation"]["tracks"][0]["keyframes"][0]["geometry"] == ring()
    invalid = copy.deepcopy(bundle)
    invalid["videos"][0]["segmentation"]["tracks"][0]["label_id"] = "unknown"
    assert client.post(f"/api/video/projects/{pid}/annotations/import", json=invalid).status_code == 400
    assert client.get(f"{base}/segmentation").json()["revision"] == 1
    result = client.post(f"/api/video/projects/{pid}/annotations/import", json=bundle)
    assert result.status_code == 200, result.text
    restored = client.get(f"{base}/segmentation").json()
    assert restored["revision"] == 2
    assert restored["reviewed_ranges"] == []
    assert not restored["tracks"][0]["keyframes"][0]["reviewed"]
    assert restored["tracks"][0]["keyframes"][0]["geometry"] == ring()
    assert client.get(f"{base}/annotations").json()["events"][0]["track_ids"] == ["person-1"]
    assert client.post(f"/api/video/projects/{pid}/annotations/import", json=bundle).status_code == 409
    explicit = {"bundle": bundle, "expected_revisions": {vid: 2}, "expected_segmentation_revisions": {vid: 2}}
    assert client.post(f"/api/video/projects/{pid}/annotations/import", json=explicit).status_code == 200


def test_mask_extraction_snapshots_unreviewed_preserves_existing_human(environment):
    client, store, pid, vid, base = environment
    row = track(frames=(0,), reviewed=True)
    row["keyframes"][0]["geometry"] = ring()
    save(client, base, [row])
    job = client.post(f"{base}/extract", json={"frame_indices": [0, 1], "task": "instance_segmentation",
        "labels": ["person"], "include_segmentation": True}).json()
    job = wait_job(client, pid, job["id"])
    assert job["status"] == "completed", job
    child = store.get(job["result"]["project_id"])
    dataset = load_dataset(store, child)
    first, empty = dataset.images
    assert first.status == "imported_coarse"
    assert first.objects[0].mask.model_dump() == ring()["mask"]
    assert first.objects[0].id == "person-1"
    assert first.objects[0].source == "imported_coarse"
    assert first.provenance["video"]["segmentation_revision"] == 1
    assert not empty.objects and empty.status == "unreviewed"
    # An existing human edit remains a separate immutable higher-priority document.
    from visionrefine.core.dataset_io.service import document_path, effective_annotation
    path = document_path(store, child["id"], first.path, "annotations")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"image": first.path, "objects": [], "revision_id": "human"}))
    changed = copy.deepcopy(row)
    changed["keyframes"][0]["geometry"] = polygon()
    save(client, base, [changed])
    repeated = client.post(f"{base}/extract", json={"frame_indices": [0], "task": "instance_segmentation",
        "labels": ["person"], "include_segmentation": True}).json()
    repeated = wait_job(client, pid, repeated["id"])
    assert repeated["result"]["skipped"] == [0]
    assert effective_annotation(store, child, first.path)["objects"] == []
    assert load_dataset(store, store.get(child["id"])).images[0].objects[0].kind == "mask"


def test_geometry_preview_never_saves_a_keyframe(environment):
    pytest.importorskip("cv2")
    client, _, _, _, base = environment
    response = client.post(f"{base}/segmentation/preview", json={"frame_index": 0, "geometry": polygon(),
        "operation": "shrink", "radius": 1, "protect_holes": True})
    assert response.status_code == 200, response.text
    assert response.json()["geometry"]["kind"] == "mask"
    assert client.get(f"{base}/segmentation").json()["revision"] == 0


def test_failed_atomic_publication_keeps_both_document_heads(environment, monkeypatch):
    import visionrefine.core.video.service as service_module
    client, _, _, _, base = environment
    saved = save(client, base, [track()])
    temporal = client.put(f"{base}/annotations", json={"events": [{"id": "event", "kind": "interval", "start": 0,
        "end": 0.8, "text": "both", "track_ids": ["person-1"]}]}).json()
    original = service_module.atomic_json
    def fail_state(path, value):
        if path.parent.name == "state":
            raise OSError("simulated interrupted publication")
        return original(path, value)
    monkeypatch.setattr(service_module, "atomic_json", fail_state)
    response = client.post(f"{base}/segmentation/operations", json={"revision": saved["revision"], "operation": "split",
        "track_id": "person-1", "frame_index": 2, "new_track_id": "person-2"})
    assert response.status_code == 400
    assert client.get(f"{base}/segmentation").json() == saved
    assert client.get(f"{base}/annotations").json() == temporal


def test_extraction_uses_frozen_shapes_when_video_is_edited_before_worker(environment, monkeypatch):
    from visionrefine.core.video.models import ExtractInput
    client, store, pid, vid, base = environment
    row = track(frames=(0,))
    row["keyframes"][0]["geometry"] = ring()
    save(client, base, [row])
    service = VideoService(store)
    captured = {}
    def capture_job(project_id, kind, payload):
        captured.update(payload)
        return {"status": "queued"}
    monkeypatch.setattr(service, "start_job", capture_job)
    service.extract(pid, vid, ExtractInput(frame_indices=[0], labels=["person"], task="instance_segmentation", include_segmentation=True))
    row["keyframes"][0]["geometry"] = polygon()
    save(client, base, [row])
    result = service._extract(pid, captured, lambda: None, lambda progress: None)
    dataset = load_dataset(store, store.get(result["project_id"]))
    assert dataset.images[0].objects[0].kind == "mask"
    assert dataset.images[0].objects[0].mask.model_dump() == ring()["mask"]
    assert dataset.images[0].provenance["video"]["segmentation_revision"] == 1


@pytest.mark.parametrize("catalog", ["labels", "object_labels"])
def test_native_merge_catalog_capacity_rejects_without_partial_writes(environment, catalog):
    client, store, pid, vid, base = environment
    endpoint = "labels" if catalog == "labels" else "object-labels"
    existing = [{"id": f"label-{i}", "name": f"label {i}", "color": "#112233"} for i in range(2000)]
    assert client.put(f"/api/video/projects/{pid}/{endpoint}", json={"labels": existing}).status_code == 200
    bundle = client.get(f"/api/video/projects/{pid}/export").json()
    bundle[catalog] = [{"id": "new-label", "name": "new imported label", "color": "#112233"}]
    # Also stage a change in the other catalog, verifying no writes occur before both unions validate.
    other = "object_labels" if catalog == "labels" else "labels"
    bundle[other] = [{"id": "other-new", "name": "other imported label", "color": "#112233"}]
    before_project = store.get(pid)
    before_temporal = client.get(f"{base}/annotations").json()
    before_segmentation = client.get(f"{base}/segmentation").json()
    response = client.post(f"/api/video/projects/{pid}/annotations/import", json=bundle)
    assert response.status_code == 400, response.text
    assert store.get(pid) == before_project
    assert client.get(f"{base}/annotations").json() == before_temporal
    assert client.get(f"{base}/segmentation").json() == before_segmentation
