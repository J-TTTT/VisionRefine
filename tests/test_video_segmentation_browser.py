"""Real browser acceptance for object identity, exact frames and editable masks."""
import json
import os
import socket
import threading
import time
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VISIONREFINE_BROWSER_TESTS") != "1", reason="Opt-in browser acceptance"
)


@pytest.fixture
def video_workspace(tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    playwright = pytest.importorskip("playwright.sync_api")
    import uvicorn
    import visionrefine.server as server
    from visionrefine.core.store import ProjectStore

    source = tmp_path / "moving-object.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for index in range(24):
            pixels = np.full((240, 320, 3), [30, 60, 45], dtype=np.uint8)
            pixels[50:160, 30 + index * 3:130 + index * 3] = [220, 140, 60]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 12)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    store = ProjectStore(tmp_path / "workspace" / "projects")
    monkeypatch.setattr(server, "store", store)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    app_server = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port,
                                             loop="asyncio", log_level="error"))
    thread = threading.Thread(target=lambda: app_server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not app_server.started and time.monotonic() < deadline:
            time.sleep(.05)
        assert app_server.started
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1660, "height": 1200})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            base = f"http://127.0.0.1:{port}"
            response = page.request.post(base + "/api/video/projects", data={
                "name": "视频分割验收", "paths": [str(source)], "labels": ["移动"]})
            assert response.status == 201, response.text()
            pid = response.json()["project"]["id"]
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                project = page.request.get(base + f"/api/video/projects/{pid}").json()
                if project["videos"]:
                    break
                page.wait_for_timeout(100)
            assert project["videos"], project
            vid = project["videos"][0]["id"]
            project_url = base + f"/api/video/projects/{pid}"
            labels = page.request.put(project_url + "/object-labels", data={"labels": [
                {"id": "object-label", "name": "方块", "color": "#f4b266"}]})
            assert labels.ok, labels.text()
            page.goto(base + f"/video?project={pid}&video={vid}")
            playwright.expect(page.locator("#workspace")).to_be_visible()
            yield page, playwright, base, project_url, vid, store
            assert errors == []
            browser.close()
    finally:
        app_server.should_exit = True
        thread.join(timeout=10)
        listener.close()


def save(page, playwright):
    if page.locator("#save").is_enabled():
        page.locator("#save").click()
    playwright.expect(page.locator("#saveState")).to_contain_text("已保存")


def draw_polygon(page, points):
    canvas = page.locator("#segmentationCanvas")
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    # Zoom is relative to the fitted original 320 x 240 image.
    zoom = min(box["width"] / 320, box["height"] / 240) * float(page.locator("#segZoom").inner_text().rstrip("%")) / 100
    left = box["x"] + (box["width"] - 320 * zoom) / 2
    top = box["y"] + (box["height"] - 240 * zoom) / 2
    for x, y in points:
        page.mouse.click(left + x * zoom, top + y * zoom)
    page.locator("#segFinish").click()
    return left, top, zoom


def mask_at(geometry, x, y):
    tile_size = geometry["mask"]["tile_size"]
    for tile in geometry["mask"]["tiles"]:
        if tile["x"] <= x < tile["x"] + tile_size and tile["y"] <= y < tile["y"] + tile_size:
            offset = (y - tile["y"]) * tile_size + x - tile["x"]
            cursor = 0
            for index, run in enumerate(tile["counts"]):
                cursor += run
                if cursor > offset:
                    return index % 2
    return 0


def test_video_shapes_occlusion_and_exchange(video_workspace):
    page, playwright, base, project_url, vid, store = video_workspace
    video_url = project_url + f"/videos/{vid}"

    # Regression: native paused seeks must replace the previous exact-frame image.
    playwright.expect(page.locator("#exactFrame")).to_be_visible()
    page.evaluate("document.getElementById('player').currentTime = 1")
    playwright.expect(page.locator("#frameReadout")).to_have_text("帧 13 / 24")
    playwright.expect(page.locator("#exactFrame")).to_have_attribute("src", f"/api/video/projects/{project_url.rsplit('/', 1)[-1]}/videos/{vid}/frames/12")
    page.evaluate("document.getElementById('player').currentTime = 0")
    playwright.expect(page.locator("#frameReadout")).to_have_text("帧 1 / 24")

    page.locator("#tabSegmentation").click()
    page.locator("#segAddTrack").click()
    page.locator("#segNewName").fill("方块 01")
    page.locator("#segNewEnd").fill("6")
    page.locator('#segNewTrackForm button[type="submit"]').click()
    playwright.expect(page.locator("#segPolygon")).to_be_enabled()
    left, top, zoom = draw_polygon(page, [(35, 55), (125, 55), (125, 155), (35, 155)])
    playwright.expect(page.locator("#segFrameReviewed")).to_be_enabled()

    # A real eraser click creates a hole and converts the polygon into a tile mask.
    page.locator("#segErase").click()
    page.locator("#segRadius").fill("9")
    page.locator("#segRadius").blur()
    page.locator("#segmentationCanvas").scroll_into_view_if_needed()
    box = page.locator("#segmentationCanvas").bounding_box()
    left = box["x"] + (box["width"] - 320 * zoom) / 2
    top = box["y"] + (box["height"] - 240 * zoom) / 2
    page.mouse.click(left + 80 * zoom, top + 105 * zoom)
    page.locator("#segFrameReviewed").check()
    save(page, playwright)
    saved = page.request.get(video_url + "/segmentation").json()
    track = saved["tracks"][0]
    track_id = track["id"]
    geometry = track["keyframes"][0]["geometry"]
    assert geometry["kind"] == "mask"
    assert mask_at(geometry, 80, 105) == 0
    assert mask_at(geometry, 50, 105) == 1
    assert track["keyframes"][0]["reviewed"]
    assert not track["reviewed"]

    # A lone keyframe must not mark the six-frame object as complete.
    page.locator("#segReviewTrack").click()
    assert "已审核" not in page.locator("#segReviewTrack").inner_text()
    assert not page.request.get(video_url + "/segmentation").json()["tracks"][0]["reviewed"]
    page.reload()
    playwright.expect(page.locator("#workspace")).to_be_visible()
    page.locator("#tabSegmentation").click()
    page.locator(f'[data-seg-track="{track_id}"]').click()
    playwright.expect(page.locator("#segFrameReviewed")).to_be_checked()
    assert page.request.get(video_url + "/segmentation").json()["tracks"][0]["keyframes"][0]["geometry"] == geometry

    # Switching tabs keeps the same frame editable. Refinement stays a preview until applied.
    page.locator("#tabEvents").click()
    page.locator("#tabSegmentation").click()
    playwright.expect(page.locator("#segBrush")).to_be_enabled()
    page.get_by_text("轮廓调整", exact=True).click()
    page.locator("#segRefineOperation").select_option("expand")
    page.locator("#segRefineRadius").fill("2")
    page.locator("#segPreviewRefine").click()
    playwright.expect(page.locator("#segApplyRefine")).to_be_enabled()
    playwright.expect(page.locator("#segBrush")).to_be_disabled()
    page.locator("#segCancelRefine").click()
    playwright.expect(page.locator("#segBrush")).to_be_enabled()
    playwright.expect(page.locator("#segFrameReviewed")).to_be_checked()
    if not page.locator("#segPreviewRefine").is_visible():
        page.get_by_text("轮廓调整", exact=True).click()
    page.locator("#segRefineOperation").select_option("expand")
    page.locator("#segRefineRadius").fill("2")
    page.locator("#segPreviewRefine").click()
    playwright.expect(page.locator("#segApplyRefine")).to_be_enabled()
    page.locator("#segApplyRefine").click()
    playwright.expect(page.locator("#segUndo")).to_be_enabled()
    page.locator("#segUndo").click()
    page.locator("#segFrameReviewed").check()
    save(page, playwright)
    assert page.request.get(video_url + "/segmentation").json()["tracks"][0]["keyframes"][0]["geometry"] == geometry

    # Frames 2–5 are explicitly occluded; frame 6 resumes the same identity.
    page.get_by_text("遮挡 / 离开区间", exact=True).click()
    page.locator("#segStateStart").fill("2")
    page.locator("#segStateEnd").fill("5")
    page.locator("#segStateReviewed").check()
    page.locator("#segApplyStateRange").click()
    page.evaluate("document.getElementById('player').currentTime = 5 / 12")
    playwright.expect(page.locator("#frameReadout")).to_have_text("帧 6 / 24")
    playwright.expect(page.locator("#segPolygon")).to_be_enabled()
    page.locator("#segPolygon").click()
    draw_polygon(page, [(50, 55), (140, 55), (140, 155), (50, 155)])
    page.locator("#segFrameReviewed").check()
    page.once("dialog", lambda dialog: dialog.accept())
    page.locator("#segReviewTrack").click()
    save(page, playwright)
    completed = page.request.get(video_url + "/segmentation").json()
    assert len(completed["tracks"]) == 1
    assert completed["tracks"][0]["id"] == track_id
    assert completed["tracks"][0]["reviewed"]
    assert [item["frame_index"] for item in completed["tracks"][0]["keyframes"]] == [0, 5]
    assert completed["tracks"][0]["visibility_ranges"] == [
        {"start_frame": 1, "end_frame": 4, "visibility": "occluded", "reviewed": True}]

    # The native bundle carries exact masks and resets review on import.
    page.locator("#transfer").click()
    with page.expect_download() as transfer:
        page.locator("#export").click()
    exported = Path(transfer.value.path()).read_bytes()
    bundle = json.loads(exported)
    assert bundle["schema_version"] == "2.0"
    assert bundle["videos"][0]["segmentation"]["tracks"][0]["keyframes"][0]["geometry"] == geometry
    page.locator("#annotationsFile").set_input_files({"name": "tracks.json", "mimeType": "application/json", "buffer": exported})
    page.once("dialog", lambda dialog: dialog.accept())
    page.locator("#importAnnotations").click()
    playwright.expect(page.locator("#transferDialog")).not_to_be_visible()
    playwright.expect(page.locator("#workspace")).to_be_visible()
    restored = page.request.get(video_url + "/segmentation").json()
    assert restored["tracks"][0]["id"] == track_id
    assert not restored["tracks"][0]["reviewed"]
    assert not any(key["reviewed"] for key in restored["tracks"][0]["keyframes"])

    # UI extraction hands the exact sparse mask to an independent image dataset.
    page.evaluate("document.getElementById('player').currentTime = 0")
    playwright.expect(page.locator("#frameReadout")).to_have_text("帧 1 / 24")
    page.locator("#tabFrames").click()
    page.locator("#sampleMode").select_option("current")
    page.locator('#sampleForm button[type="submit"]').click()
    playwright.expect(page.locator("#sampleFrames input[type=checkbox]")).to_have_count(1)
    page.locator("#extractTask").select_option("instance_segmentation")
    page.locator("#extractLabels").fill("方块")
    page.locator("#includeSegmentation").check()
    page.locator('#extractForm button[type="submit"]').click()
    deadline = time.monotonic() + 20
    extraction = None
    while time.monotonic() < deadline:
        jobs = page.request.get(project_url + "/jobs").json()
        extraction = next((job for job in jobs if job["kind"] == "extract"), None)
        if extraction and extraction["status"] in {"completed", "failed"}:
            break
        page.wait_for_timeout(100)
    assert extraction and extraction["status"] == "completed", extraction
    from visionrefine.core.dataset_io.service import load_dataset
    child = store.get(extraction["result"]["project_id"])
    dataset = load_dataset(store, child)
    image = dataset.images[0]
    assert image.status == "imported_coarse"
    assert image.objects[0].provenance["video"]["track_id"] == track_id
    assert image.objects[0].mask.model_dump() == geometry["mask"]


def test_video_track_operations_and_conflict_preserve_edits(video_workspace):
    page, playwright, base, project_url, vid, store = video_workspace
    video_url = project_url + f"/videos/{vid}"
    page.locator("#tabSegmentation").click()
    page.locator("#segAddTrack").click()
    page.locator("#segNewName").fill("待拆分对象")
    page.locator("#segNewEnd").fill("6")
    page.locator('#segNewTrackForm button[type="submit"]').click()
    playwright.expect(page.locator("#segPolygon")).to_be_enabled()
    draw_polygon(page, [(35, 55), (125, 55), (125, 155), (35, 155)])
    save(page, playwright)
    original = page.request.get(video_url + "/segmentation").json()["tracks"][0]["id"]

    page.locator("#tabEvents").click()
    page.locator("#addRecord").click()
    page.locator("#recordStart").fill("0")
    page.locator("#recordEnd").fill("0.5")
    page.locator("#eventTracks").select_option([original])
    save(page, playwright)
    assert page.request.get(video_url + "/annotations").json()["events"][0]["track_ids"] == [original]

    page.locator("#tabSegmentation").click()
    page.evaluate("document.getElementById('player').currentTime = 0.25")
    playwright.expect(page.locator("#frameReadout")).to_have_text("帧 4 / 24")
    playwright.expect(page.locator("#segPolygon")).to_be_enabled()
    page.get_by_text("轨迹拆分 / 合并 / 删除", exact=True).click()
    page.once("dialog", lambda dialog: dialog.accept())
    page.locator("#segSplitTrack").click()
    playwright.expect(page.locator("#segTrackCount")).to_have_text("2")
    split = page.request.get(video_url + "/segmentation").json()
    added = next(track["id"] for track in split["tracks"] if track["id"] != original)
    assert set(page.request.get(video_url + "/annotations").json()["events"][0]["track_ids"]) == {original, added}
    page.locator(f'[data-seg-track="{original}"]').click()
    if not page.locator("#segMergeOther").is_visible():
        page.get_by_text("轨迹拆分 / 合并 / 删除", exact=True).click()
    page.locator("#segMergeOther").select_option(added)
    page.once("dialog", lambda dialog: dialog.accept())
    page.locator("#segMergeTrack").click()
    playwright.expect(page.locator("#segTrackCount")).to_have_text("1")
    merged = page.request.get(video_url + "/segmentation").json()
    assert merged["tracks"][0]["id"] == original
    assert merged["tracks"][0]["end_frame"] == 5
    assert page.request.get(video_url + "/annotations").json()["events"][0]["track_ids"] == [original]

    # A second writer must not erase the first window's unsaved edits or be overwritten.
    merged["tracks"][0]["name"] = "其他窗口保存的名称"
    assert page.request.put(video_url + "/segmentation", data=merged).ok
    page.locator("#segTrackName").fill("本窗口尚未保存的名称")
    page.locator("#segTrackName").blur()
    with page.expect_response(lambda response: response.url == video_url + "/segmentation" and response.request.method == "PUT") as save_response:
        page.locator("#save").click()
    assert save_response.value.status == 409
    assert page.locator("#segTrackName").input_value() == "本窗口尚未保存的名称"
    assert page.request.get(video_url + "/segmentation").json()["tracks"][0]["name"] == "其他窗口保存的名称"
