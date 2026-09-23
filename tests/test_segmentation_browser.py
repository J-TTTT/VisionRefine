"""Opt-in real pointer/keyboard acceptance for instance segmentation."""
import os
import socket
import threading
import time

import pytest
from PIL import Image
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(os.environ.get("VISIONREFINE_BROWSER_TESTS") != "1", reason="Opt-in browser acceptance")


def test_polygon_mask_edit_save_reload(tmp_path, monkeypatch):
    playwright = pytest.importorskip("playwright.sync_api")
    import uvicorn
    import visionrefine.server as server
    from visionrefine.core.store import ProjectStore

    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (640, 480), "#e4e8e4").save(root / "sample.png")
    Image.new("RGB", (640, 480), "#eeeeee").save(root / "second.png")
    monkeypatch.setattr(server, "store", ProjectStore(tmp_path / "workspace/projects"))
    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path / "workspace")
    client = TestClient(server.app)
    project = client.post("/api/projects", json=dict(name="分割浏览器验收", task="instance_segmentation",
        dataset_path=str(root), labels=["cell", "defect"], model_max_side=1536)).json()
    assert client.post(f"/api/projects/{project['id']}/analyze").status_code == 200
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    app_server = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, loop="asyncio", log_level="error"))
    thread = threading.Thread(target=lambda: app_server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not app_server.started and time.monotonic() < deadline:
            time.sleep(.05)
        assert app_server.started
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}")

            def open_editor():
                page.get_by_role("button", name="分割浏览器验收 实例分割").click()
                page.locator("#openWorkspace").click()
                page.wait_for_function("editor.thumbnail.naturalWidth > 0 && editor.image === 'sample.png'")
                page.locator("#detailCanvas").scroll_into_view_if_needed()

            def screen(x, y):
                return page.evaluate("""([x,y]) => {
                  const c=$('detailCanvas'), r=c.getBoundingClientRect(), f=canvasFit(c,editor.view.width,editor.view.height);
                  return [r.left+(f.x+(x-editor.view.x)*f.scale)*r.width/c.width,
                          r.top+(f.y+(y-editor.view.y)*f.scale)*r.height/c.height];
                }""", [x, y])

            def click(x, y):
                page.mouse.click(*screen(x, y))

            def drag(a, b):
                page.mouse.move(*screen(*a))
                page.mouse.down()
                page.mouse.move(*screen(*b), steps=5)
                page.mouse.up()

            def objects():
                return page.evaluate("JSON.parse(JSON.stringify(editor.objects))")

            def mask_at(obj, x, y):
                for tile in obj["mask"]["tiles"]:
                    if tile["x"] <= x < tile["x"] + 128 and tile["y"] <= y < tile["y"] + 128:
                        target = (y-tile["y"])*128+x-tile["x"]
                        at = 0
                        for i, n in enumerate(tile["counts"]):
                            if at <= target < at+n:
                                return i % 2
                            at += n
                return 0

            open_editor()
            playwright.expect(page.locator("#segmentationTools")).to_be_visible()
            playwright.expect(page.locator("#runInitialDetection")).to_be_hidden()
            # Draw with real pointer input, close with Enter.
            for point in [(64, 120), (240, 120), (240, 280), (64, 280)]:
                click(*point)
            page.keyboard.press("Enter")
            assert len(objects()) == 1
            assert objects()[0]["kind"] == "polygon"
            drag((64, 120), (80, 100))
            assert objects()[0]["polygons"][0][0] == pytest.approx([80, 100], abs=.8)
            page.keyboard.down("Shift")
            click(160, 110)
            page.keyboard.up("Shift")
            assert len(objects()[0]["polygons"][0]) == 5
            page.locator("#deleteVertex").click()
            assert len(objects()[0]["polygons"][0]) == 4
            page.locator("#appendContour").click()
            for point in [(330, 120), (380, 120), (380, 170), (330, 170)]:
                click(*point)
            page.keyboard.press("Enter")
            assert len(objects()[0]["polygons"]) == 2
            page.locator("#saveAnnotations").click()
            playwright.expect(page.locator("#workspaceStatus")).to_contain_text("已保存 1 个实例")
            saved_polygon = objects()
            page.reload()
            open_editor()
            assert objects() == saved_polygon
            # Vertex edits after a zoom still use original-image coordinates.
            page.locator("#editBoxTool").click()
            click(150, 240)
            page.mouse.move(*screen(200, 220))
            page.mouse.wheel(0, -200)
            page.wait_for_timeout(300)
            drag((80, 100), (90, 110))
            assert objects()[0]["polygons"][0][0] == pytest.approx([90, 110], abs=.8)
            page.locator("#segUndo").click()
            assert objects() == saved_polygon
            # Mask conversion, holes, and undo/redo preserve independent components.
            page.locator("#brushRadius").fill("16")
            page.locator("#eraseMask").click()
            click(150, 200)
            assert objects()[0]["kind"] == "mask"
            assert mask_at(objects()[0], 150, 200) == 0
            assert mask_at(objects()[0], 150, 240) == 1
            assert mask_at(objects()[0], 350, 145) == 1
            mask = objects()
            page.locator("#segUndo").click()
            assert objects() == saved_polygon
            page.locator("#segRedo").click()
            assert objects() == mask
            page.locator("#brushMask").click()
            click(150, 200)
            assert mask_at(objects()[0], 150, 200) == 1
            page.locator("#segUndo").click()
            assert objects() == mask
            # A new brush instance can cross tile boundaries and remain separate.
            page.locator("#workspaceLabel").select_option("defect")
            page.locator("#newMask").click()
            drag((450, 350), (540, 350))
            assert len(objects()) == 2
            assert objects()[1]["label"] == "defect"
            assert mask_at(objects()[1], 511, 350) == 1
            assert mask_at(objects()[1], 512, 350) == 1
            page.locator("#saveAnnotations").click()
            playwright.expect(page.locator("#workspaceStatus")).to_contain_text("已保存 2 个实例")
            saved_masks = objects()
            page.reload()
            open_editor()
            assert objects() == saved_masks
            # Operations remain previews until applied and are undoable.
            page.locator("#editBoxTool").click()
            click(150, 240)
            page.locator("#segEditRadius").fill("3")
            page.locator("#segExpand").click()
            playwright.expect(page.locator("#segOperationPreview")).to_be_visible()
            assert objects() == saved_masks
            page.locator("#segCancelPreview").click()
            assert objects() == saved_masks
            page.locator("#segExpand").click()
            playwright.expect(page.locator("#segApplyPreview")).to_be_enabled()
            page.locator("#segApplyPreview").click()
            assert objects()[0]["area"] > saved_masks[0]["area"]
            page.locator("#segUndo").click()
            assert objects() == saved_masks
            # Multipart mask splits by connectivity and keeps its pixels.
            page.locator("#segSplit").click()
            playwright.expect(page.locator("#segOperationPreview")).to_be_visible()
            page.locator("#segApplyPreview").click()
            assert len(objects()) == 3
            page.locator("#segUndo").click()
            assert objects() == saved_masks
            # Ctrl-click multi-selection enables sparse union.
            page.keyboard.down("Control")
            click(150, 240)
            click(480, 350)
            page.keyboard.up("Control")
            playwright.expect(page.locator("#segMerge")).to_be_enabled()
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator("#segMerge").click()
            playwright.expect(page.locator("#segOperationPreview")).to_be_visible()
            page.locator("#segApplyPreview").click()
            assert len(objects()) == 1
            page.locator("#segUndo").click()
            assert objects() == saved_masks
            # Manual line splits the connected brush capsule without losing area.
            page.locator("#editBoxTool").click()
            click(480, 350)
            page.locator("#segCutTool").click()
            drag((500, 320), (500, 380))
            page.locator("#segSplit").click()
            playwright.expect(page.locator("#segOperationPreview")).to_be_visible()
            page.locator("#segApplyPreview").click()
            assert len(objects()) == 3
            assert sum(o["area"] for o in objects()) == sum(o["area"] for o in saved_masks)
            page.locator("#segUndo").click()
            assert objects() == saved_masks
            # Unfinished polygons cannot be saved, and image switching asks before discarding.
            page.locator("#drawBoxTool").click()
            click(50, 350)
            click(100, 350)
            page.locator("#saveAnnotations").click()
            playwright.expect(page.locator("#workspaceStatus")).to_contain_text("请先完成")
            page.once("dialog", lambda dialog: dialog.dismiss())
            page.locator("#workspaceImage").select_option("second.png")
            assert page.locator("#workspaceImage").input_value() == "sample.png"
            page.locator("#cancelContour").click()
            # Relabel is an undoable change and an erased pixel does not select a mask.
            page.locator("#editBoxTool").click()
            click(150, 200)
            assert page.evaluate("editor.selected") == -1
            click(150, 240)
            assert page.evaluate("editor.selected") == 0
            label_point = page.evaluate("""() => {const b=selectedLabelRect(), c=$('detailCanvas'),r=c.getBoundingClientRect();return [r.left+(b.x+b.width/2)*r.width/c.width,r.top+(b.y+b.height/2)*r.height/c.height];}""")
            page.mouse.click(*label_point)
            page.locator("#boxLabelOptions").get_by_role("option", name="defect", exact=True).click()
            assert objects()[0]["label"] == "defect"
            page.locator("#segUndo").click()
            assert objects()[0]["label"] == "cell"
            page.screenshot(path=str(tmp_path / "segmentation-editor.png"), full_page=True)
            assert errors == []
            browser.close()
    finally:
        app_server.should_exit = True
        thread.join(timeout=10)
