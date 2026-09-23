"""Optional real-browser acceptance: VISIONREFINE_BROWSER_TESTS=1 pytest -q ..."""
import json
import os
import socket
import threading
import time
from zipfile import ZipFile

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(os.environ.get("VISIONREFINE_BROWSER_TESTS") != "1", reason="Opt-in browser acceptance")


def test_import_edit_export_in_browser(tmp_path, monkeypatch):
    playwright = pytest.importorskip("playwright.sync_api")
    import uvicorn
    import visionrefine.server as server
    from visionrefine.core.store import ProjectStore

    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (640, 480), "white").save(root / "sample.jpg")
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 7, "name": "cell"}, {"id": 8, "name": "defect"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [64, 100, 128, 100]}],
    }))
    monkeypatch.setattr(server, "store", ProjectStore(tmp_path / "workspace/projects"))
    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path / "workspace")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    app_server = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, loop="asyncio", log_level="error"))
    thread = threading.Thread(target=lambda: app_server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not app_server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert app_server.started
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1400, "height": 1100})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{port}")
                page.wait_for_function("datasetFormats.length === 8")
                def confirm_export():
                    playwright.expect(page.locator("#exportPreflight")).to_be_visible()
                    for checkbox in page.locator("#exportPreflightFormats input[type=checkbox]").all():
                        checkbox.check()
                    playwright.expect(page.locator("#confirmExport")).to_be_enabled()
                    page.locator("#confirmExport").click()
                    playwright.expect(page.locator("#exportDownload")).to_be_visible(timeout=15000)
                page.locator('[name="name"]').fill("Browser Dataset")
                page.locator("#datasetFormat").select_option("coco_detection")
                playwright.expect(page.locator("#labelsField")).to_be_hidden()
                page.locator('[name="dataset_path"]').fill(str(root))
                page.locator("#annotationSource").fill(str(source))
                page.locator("#importSplit").select_option("val")
                page.locator('#projectForm button[type="submit"]').click()
                playwright.expect(page.locator("#commitDatasetImport")).to_be_enabled()
                playwright.expect(page.locator("#importPreviewSummary")).to_contain_text("新增 1 张")
                page.locator("#commitDatasetImport").click()
                playwright.expect(page.locator("#pageTitle")).to_have_text("Browser Dataset")
                playwright.expect(page.locator("#createView")).to_be_hidden()
                playwright.expect(page.locator("#datasetIOSummary")).to_contain_text("1 张图像")
                page.locator("#showImportReport").click()
                playwright.expect(page.locator("#importReport")).to_contain_text('"object_count": 1')
                page.locator("#openWorkspace").click()
                page.wait_for_function("editor.objects.length === 1 && editor.thumbnail.naturalWidth > 0")
                assert page.evaluate("editor.annotationStatus") == "imported_coarse"
                # Select the imported box through actual pointer events.
                point = page.evaluate("""() => {
                    const c = $('detailCanvas'), f = canvasFit(c, editor.view.width, editor.view.height);
                    const b = editor.objects[0].bbox;
                    return {x:(f.x + ((b[0]+b[2])/2-editor.view.x)*f.scale)*c.clientWidth/c.width,
                            y:(f.y + ((b[1]+b[3])/2-editor.view.y)*f.scale)*c.clientHeight/c.height};
                }""")
                page.locator("#detailCanvas").click(position=point)
                assert page.evaluate("editor.selected") == 0
                point = page.evaluate("""() => {
                    const c = $('detailCanvas'), r = selectedLabelRect();
                    return {x:(r.x+r.width/2)*c.clientWidth/c.width,
                            y:(r.y+r.height/2)*c.clientHeight/c.height};
                }""")
                page.locator("#detailCanvas").click(position=point)
                page.locator("#boxLabelOptions button", has_text="defect").click()
                assert page.evaluate("editor.objects[0].label") == "defect"
                page.locator("#saveAnnotations").click()
                playwright.expect(page.locator("#workspaceStatus")).to_contain_text("已保存 1 个框")
                page.locator("#exportFormat").select_option("yolo_detection")
                page.locator("#exportImages").select_option("true")
                page.locator("#exportDataset").click()
                confirm_export()
                playwright.expect(page.locator("#exportDownload")).to_be_visible()
                with page.expect_download() as download:
                    page.locator("#exportDownload").click()
                package = tmp_path / "download.zip"
                download.value.save_as(package)
                with ZipFile(package) as archive:
                    report = json.loads(archive.read("export-report.json"))
                    assert report["format"] == "yolo_detection"
                    assert report["image_count"] == 1
                    assert archive.read("labels/val/source1/sample.txt").decode().startswith("1 ")
                # Reload the app and load the saved human version through UI.
                page.reload()
                page.locator(".project-item", has_text="Browser Dataset").click()
                page.locator("#openWorkspace").click()
                page.wait_for_function("editor.annotationStatus === 'human_reviewed'")
                assert page.evaluate("editor.objects[0].label") == "defect"
                # Append a duplicate coarse version and a differently named new source,
                # explicitly map its alias, then prove the old manual edit survives.
                second_root = tmp_path / "second"
                second_root.mkdir()
                Image.new("RGB", (640, 480), "red").save(second_root / "sample.jpg")
                second_source = second_root / "annotations.json"
                second_data = json.loads(source.read_text())
                second_data["categories"] = [{"id": 7, "name": "cell_alias"}]
                second_source.write_text(json.dumps(second_data))
                page.locator("#appendDataset").click()
                first_row = page.locator(".import-source").nth(0)
                first_row.locator('[data-field="id"]').fill("dup")
                first_row.locator('[data-field="format"]').select_option("coco_detection")
                first_row.locator('[data-field="root"]').fill(str(root))
                first_row.locator('[data-field="annotation_path"]').fill(str(source))
                page.locator("#addImportSource").click()
                second_row = page.locator(".import-source").nth(1)
                second_row.locator('[data-field="id"]').fill("new")
                second_row.locator('[data-field="format"]').select_option("coco_detection")
                second_row.locator('[data-field="root"]').fill(str(second_root))
                second_row.locator('[data-field="annotation_path"]').fill(str(second_source))
                second_row.locator('[data-field="split"]').select_option("train")
                page.locator("#previewDatasetImport").click()
                playwright.expect(page.locator("#commitDatasetImport")).to_be_enabled()
                playwright.expect(page.locator("#importPreviewSummary")).to_contain_text("跳过重复 1 张")
                page.get_by_label("new: cell_alias 目标类别", exact=True).fill("cell")
                playwright.expect(page.locator("#commitDatasetImport")).to_be_disabled()
                page.locator("#importConflicts select").select_option("update_coarse")
                page.locator("#previewDatasetImport").click()
                playwright.expect(page.locator("#commitDatasetImport")).to_be_enabled()
                playwright.expect(page.locator("#importPreviewSummary")).to_contain_text("更新粗标注 1 张")
                playwright.expect(page.locator("#importPreviewSummary")).to_contain_text("2 类")
                page.locator("#commitDatasetImport").click()
                playwright.expect(page.locator("#datasetImportDialog")).not_to_be_visible()
                playwright.expect(page.locator("#statImages")).to_have_text("2")
                page.locator("#showDatasetHistory").click()
                playwright.expect(page.locator("#datasetHistory > details")).to_have_count(2)
                page.locator("#openWorkspace").click()
                page.wait_for_function("editor.annotationStatus === 'human_reviewed'")
                assert page.evaluate("editor.objects[0].label") == "defect"
                page.locator("#exportFormat").select_option("voc_detection")
                page.locator("#exportPolicy").select_option("all_annotated")
                page.locator("#exportDataset").click()
                confirm_export()
                playwright.expect(page.locator("#exportDownload")).to_be_visible()
                with page.expect_download() as download:
                    page.locator("#exportDownload").click()
                package = tmp_path / "voc.zip"
                download.value.save_as(package)
                with ZipFile(package) as archive:
                    report = json.loads(archive.read("export-report.json"))
                    assert report["format"] == "voc_detection" and report["image_count"] == 2
                    assert len([p for p in archive.namelist() if p.startswith("Annotations/")]) == 2
                # Registry-generated multi-format controls and persistent presets.
                page.locator(".format-list summary").click()
                for format_id in ("cvat_detection", "label_studio_detection", "labelme_detection", "visionrefine"):
                    page.locator(f'[data-export-format="{format_id}"]').check()
                page.locator("#exportSplit").select_option(["train", "val"])
                page.locator("#exportImages").select_option("true")
                page.locator("#exportPresetName").fill("跨工具检测")
                page.locator("#saveExportPreset").click()
                playwright.expect(page.locator("#exportStatus")).to_contain_text("预设已保存")
                page.reload()
                page.locator(".project-item", has_text="Browser Dataset").click()
                page.locator("#exportPreset").select_option(label="跨工具检测")
                assert page.evaluate("exportOptions().splits") == ["train", "val"]
                page.locator("#exportDataset").click()
                playwright.expect(page.locator("#confirmExport")).to_be_disabled()
                confirm_export()
                with page.expect_download() as download:
                    page.locator("#exportDownload").click()
                package = tmp_path / "batch.zip"
                download.value.save_as(package)
                with ZipFile(package) as archive:
                    assert len([n for n in archive.namelist() if n.endswith(".zip")]) == 5
                    assert json.loads(archive.read("batch-report.json"))["status"] == "completed"
                    native = tmp_path / "native.zip"
                    native.write_bytes(archive.read("visionrefine.zip"))
                # Default-off trust checkbox is a real UI affordance, not just an API flag.
                page.locator("#newProject").click()
                page.locator('[name="name"]').fill("Native coarse")
                page.locator("#datasetFormat").select_option("visionrefine")
                page.locator('[name="dataset_path"]').fill(str(tmp_path / "unused"))
                page.locator("#annotationSource").fill(str(native))
                page.locator('#projectForm button[type="submit"]').click()
                playwright.expect(page.locator('[data-field="trust_reviewed"]')).to_be_visible()
                playwright.expect(page.locator('[data-field="trust_reviewed"]')).not_to_be_checked()
                page.wait_for_function("!importSession.busy")
                assert page.locator("#commitDatasetImport").is_enabled(), page.locator("#datasetImportStatus").inner_text()
                playwright.expect(page.locator("#commitDatasetImport")).to_be_enabled()
                page.locator("#commitDatasetImport").click()
                playwright.expect(page.locator("#pageTitle")).to_have_text("Native coarse")
                page.locator("#openWorkspace").click()
                page.wait_for_function("editor.annotationStatus === 'imported_coarse'")
                assert not errors
            finally:
                browser.close()
    finally:
        app_server.should_exit = True
        thread.join(timeout=10)
        listener.close()
