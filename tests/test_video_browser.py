"""Opt-in acceptance with a real video, browser input, and durable API state."""
import os
import json
import socket
import threading
import time
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VISIONREFINE_BROWSER_TESTS") != "1", reason="Opt-in browser acceptance"
)


def test_video_events_captions_and_frame_handoff(tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    playwright = pytest.importorskip("playwright.sync_api")
    import uvicorn
    import visionrefine.server as server
    from visionrefine.core.store import ProjectStore

    source = tmp_path / "sample.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for index in range(36):
            pixels = np.full((240, 320, 3), [30, 60, 45], dtype=np.uint8)
            pixels[70:140, 20 + index * 4:60 + index * 4] = [220, 140, 60]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 12)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    monkeypatch.setattr(server, "store", ProjectStore(tmp_path / "workspace" / "projects"))
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
            page = browser.new_page(viewport={"width": 1560, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            base = f"http://127.0.0.1:{port}"
            page.goto(base + "/video")
            page.locator("#newProject").click()
            page.locator('#projectForm [name="name"]').fill("视频第一版验收")
            page.locator('#projectForm [name="paths"]').fill(str(source))
            page.locator('#projectForm [name="labels"]').fill("移动,接触")
            page.locator('#projectForm [name="split"]').select_option("train")
            page.locator('#projectForm button[type="submit"]').click()
            playwright.expect(page.locator("#workspace")).to_be_visible(timeout=30000)
            playwright.expect(page.locator("#videoCount")).to_have_text("1")
            page.locator("#play").click()
            page.wait_for_function("document.getElementById('player').currentTime > 0.1 && !document.getElementById('player').paused")
            page.locator("#play").click()
            page.locator("#nextFrame").click()
            playwright.expect(page.locator("#exactFrame")).to_be_visible()
            page.wait_for_function("document.getElementById('exactFrame').naturalWidth > 0")

            # Exercise both kinds of event and free text, then reload the document.
            page.locator("#addRecord").click()
            page.locator("#recordStart").fill("0.25")
            page.locator("#recordEnd").fill("1.5")
            page.locator("#recordText").fill("橙色方块向右移动")
            page.locator("#recordText").blur()
            page.locator("#save").click()
            playwright.expect(page.locator("#saveState")).to_contain_text("已保存")
            page.locator("#addRecord").click()
            page.locator("#eventKind").select_option("point")
            page.locator("#eventLabel").select_option("")
            page.locator("#recordStart").fill("1.75")
            page.locator("#recordText").fill("经过参考位置")
            page.locator("#recordText").blur()
            page.locator("#save").click()
            playwright.expect(page.locator("#saveState")).to_contain_text("已保存")

            page.locator("#tabCaptions").click()
            page.locator("#addRecord").click()
            page.locator("#recordText").fill("深绿色背景上，一个橙色方块由左向右移动。")
            page.locator("#recordText").blur()
            page.locator("#save").click()
            playwright.expect(page.locator("#saveState")).to_contain_text("已保存")

            # Editing again after saving must update the newly persisted document.
            page.locator("#recordText").fill("深绿色背景上，一个橙色方块由左向右移动。已复核。")
            page.locator("#save").click()
            playwright.expect(page.locator("#saveState")).to_contain_text("已保存")
            page.locator("#addRecord").click()
            page.locator("#captionScope").select_option("segment")
            page.locator("#recordStart").fill("0.5")
            page.locator("#recordEnd").fill("2")
            page.locator("#recordText").fill("方块继续向右运动。")
            page.locator("#save").click()
            playwright.expect(page.locator("#saveState")).to_contain_text("已保存")

            projects = page.request.get(base + "/api/video/projects").json()
            project_id = projects[0]["id"]
            bundle = page.request.get(base + f"/api/video/projects/{project_id}").json()
            video_id = bundle["videos"][0]["id"]
            annotations_url = base + f"/api/video/projects/{project_id}/videos/{video_id}/annotations"
            saved = page.request.get(annotations_url).json()
            assert len(saved["events"]) == 2
            assert saved["events"][0]["start"] == .25
            assert saved["events"][0]["end"] == 1.5
            assert saved["events"][1]["kind"] == "point"
            assert saved["events"][1]["label_id"] is None
            assert saved["captions"][0]["text"].startswith("深绿色背景")
            assert saved["captions"][0]["text"].endswith("已复核。")
            assert saved["captions"][1]["start"] == .5
            assert saved["review"] == {"events": "unreviewed", "captions": "unreviewed"}
            page.reload()
            playwright.expect(page.locator("#workspace")).to_be_visible()
            playwright.expect(page.locator("#eventCount")).to_have_text("2")
            playwright.expect(page.locator("#captionCount")).to_have_text("2")

            page.locator("#manageLabels").click()
            page.locator('input[data-label-name="0"]').fill("移动动作")
            page.locator("#labelsFile").set_input_files({"name": "labels.txt", "mimeType": "text/plain", "buffer": "移动动作\n停留".encode()})
            page.locator("#applyLabelImport").click()
            page.locator('#labelsForm button[type="submit"]').click()
            playwright.expect(page.locator("#labelsDialog")).not_to_be_visible()
            labels = page.request.get(base + f"/api/video/projects/{project_id}/labels").json()["labels"]
            assert [label["name"] for label in labels] == ["移动动作", "接触", "停留"]
            assert labels[0]["id"] == saved["events"][0]["label_id"]
            page.locator("#tabEvents").click()
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator("#reviewTask").click()
            playwright.expect(page.locator("#saveState")).to_contain_text("已保存")
            reviewed = page.request.get(annotations_url).json()
            assert reviewed["review"] == {"events": "reviewed", "captions": "unreviewed"}

            page.locator("#transfer").click()
            with page.expect_download() as transfer:
                page.locator("#export").click()
            exported = Path(transfer.value.path()).read_bytes()
            assert json.loads(exported)["videos"][0]["annotations"]["review"]["events"] == "reviewed"
            page.locator("#annotationsFile").set_input_files({"name": "roundtrip.json", "mimeType": "application/json", "buffer": exported})
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator("#importAnnotations").click()
            playwright.expect(page.locator("#transferDialog")).not_to_be_visible()
            playwright.expect(page.locator("#workspace")).to_be_visible()
            restored = page.request.get(annotations_url).json()
            assert restored["review"]["events"] == "unreviewed"
            assert len(restored["events"]) == 2

            page.locator("#tabFrames").click()
            page.locator("#sampleMode").select_option("count")
            page.locator("#sampleCount").fill("3")
            page.locator('#sampleForm button[type="submit"]').click()
            playwright.expect(page.locator("#sampleFrames input[type=checkbox]")).to_have_count(3)
            page.locator("#extractLabels").fill("方块")
            page.locator('#extractForm button[type="submit"]').click()
            deadline = time.monotonic() + 30
            derived_id = None
            while time.monotonic() < deadline:
                jobs = page.request.get(base + f"/api/video/projects/{project_id}/jobs").json()
                extraction = next((job for job in jobs if job.get("kind") == "extract"), None)
                if extraction and extraction["status"] == "completed":
                    derived_id = extraction["result"]["project_id"]
                    break
                page.wait_for_timeout(100)
            assert derived_id, jobs
            page.goto(base + f"/?project={derived_id}")
            playwright.expect(page.locator("#annotationWorkspace")).to_be_visible()
            page.wait_for_function("typeof editor !== 'undefined' && editor.image && editor.thumbnail.naturalWidth > 0")
            playwright.expect(page.locator("#videoFrameOrigin a")).to_be_visible()
            page.locator("#videoFrameOrigin a").click()
            playwright.expect(page.locator("#workspace")).to_be_visible()
            assert f"video={video_id}" in page.url
            assert errors == []
            browser.close()
    finally:
        app_server.should_exit = True
        thread.join(timeout=10)
        listener.close()
