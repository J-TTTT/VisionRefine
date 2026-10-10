import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import visionrefine.server as server
from scripts.benchmark_project_browsing import synthetic_project
from visionrefine.core.catalog import dataset_index
from visionrefine.core.dataset_io import service
from visionrefine.core.store import ProjectStore


@pytest.fixture
def large_project(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / "workspace/projects")
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path / "workspace")
    project = synthetic_project(store, tmp_path / "images", 10003)
    return TestClient(server.app), project


def test_bounded_pages_and_single_image_requests_survive_restart(large_project, monkeypatch):
    client, project = large_project
    base = f"/api/projects/{project['id']}"
    monkeypatch.setattr(server, "store", ProjectStore(server.store.root))
    read_text = Path.read_text

    def no_manifest_read(path, *args, **kwargs):
        assert not (path.parent.name == "datasets" and path.suffix == ".json"), "Browsing parsed the full manifest"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_manifest_read)
    for url in ("/api/projects", base):
        response = client.get(url)
        assert response.status_code == 200
        assert len(response.content) < 6000
    assert "images" not in client.get(base).json()["analysis"]
    paths = []
    for offset in range(0, 10003, 200):
        response = client.get(base + "/images", params=dict(offset=offset, limit=200))
        page = response.json()
        assert page["total"] == 10003
        assert len(page["items"]) <= 200
        paths.extend(row["path"] for row in page["items"])
    assert paths == [f"synthetic/{i:06}.png" for i in range(10003)]
    assert len(client.get(base + "/images").json()["items"]) == 50
    assert not client.get(base + "/images", params={"offset": 20000}).json()["items"]
    selected = paths[-1]
    annotation = client.get(base + "/annotations", params={"image": selected}).json()
    assert annotation["status"] == "imported_coarse" and len(annotation["objects"]) == 1
    assert client.get(base + "/thumbnail/" + selected).status_code == 200
    assert client.get(base + "/crop/" + selected, params=dict(x=0, y=0, width=64, height=64)).status_code == 200
    saved = client.put(base + "/annotations", json=dict(image=selected, objects=[]))
    assert saved.status_code == 200
    assert saved.json()["parent_revision_id"] == project["dataset_revision"]
    reloaded = client.get(base + "/annotations", params={"image": selected}).json()
    assert reloaded["status"] == "human_reviewed" and reloaded["objects"] == []
    assert client.get(base + "/dataset/report").json()["image_count"] == 10003
    assert client.get(base + "/annotations", params={"image": "unknown.png"}).status_code == 404
    assert client.get(base + "/annotations", params={"image": "../sample.png"}).status_code == 404


def test_search_paging_validation_and_new_revision(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / "workspace/projects")
    monkeypatch.setattr(server, "store", store)
    project = synthetic_project(store, tmp_path / "images", 105)
    client = TestClient(server.app)
    base = f"/api/projects/{project['id']}"
    page = client.get(base + "/images", params=dict(q="SYNTHETIC/0000", split="val", limit=20, offset=40)).json()
    assert page["total"] == 50 and len(page["items"]) == 10
    assert page["items"][0]["path"] == "synthetic/000081.png"
    for params in ({"offset": -1}, {"limit": 0}, {"limit": 201}, {"limit": "x"}, {"q": "x" * 257}, {"split": "invalid"}):
        assert client.get(base + "/images", params=params).status_code == 422
    assert client.get(base + "/images", params={"q": "%_not_a_wildcard"}).json()["total"] == 0
    old_page = client.get(base + "/images").json()
    dataset = service.load_dataset(store, project)
    dataset.images.append(dataset.images[0].model_copy(update={"id": "new.png", "path": "new.png"}))
    dataset.report.image_count += 1
    service.persist_import(store, project, dataset, hash_images=False)
    assert client.get(base + "/images").json()["total"] == 0
    assert client.post(base + "/analyze").status_code == 200
    new_page = client.get(base + "/images").json()
    assert new_page["total"] == 106 and new_page["revision"] != old_page["revision"]
    assert client.get(base + "/annotations", params={"image": "new.png"}).json()["objects"]


def test_legacy_analysis_migrates_once_and_missing_dataset_index_rebuilds(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / "workspace/projects")
    monkeypatch.setattr(server, "store", store)
    project = synthetic_project(store, tmp_path / "images", 3)
    client = TestClient(server.app)
    base = f"/api/projects/{project['id']}"
    page = client.get(base + "/images").json()
    project["analysis"].pop("image_index")
    project["analysis"]["images"] = page["items"]
    project_path = store.root / project["id"] / "project.json"
    project_path.write_text(json.dumps(project))
    dataset_index(store, project).unlink()
    assert client.get("/api/projects").status_code == 200
    migrated = client.get(base).json()
    assert "images" not in migrated["analysis"]
    assert client.get(base + "/images").json()["items"] == page["items"]
    assert client.get(base + "/annotations", params={"image": "synthetic/000002.png"}).status_code == 200
    assert dataset_index(store, project).exists()
    assert store.get(project["id"])["analysis"]["image_index"] == migrated["analysis"]["image_index"]
    (store.root / project["id"] / "analysis" / f"{migrated['analysis']['image_index']}.sqlite3").unlink()
    assert client.get(base + "/images").status_code == 409
    assert client.post(base + "/analyze").status_code == 200
    assert client.get(base + "/images").json()["total"] == 3


@pytest.mark.skipif(os.environ.get("VISIONREFINE_BROWSER_TESTS") != "1", reason="Opt-in browser acceptance")
def test_large_project_browser(large_project, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    import uvicorn

    _, project = large_project
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
            errors, image_requests = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: image_requests.append(request.url) if "/images?" in request.url else None)
            page.goto(f"http://127.0.0.1:{port}")
            page.get_by_role("button", name=f"{project['name']} 实例分割").click()
            playwright.expect(page.locator("#imageRows tr")).to_have_count(50)
            playwright.expect(page.locator("#projectImageCount")).to_contain_text("10,003")
            assert page.evaluate("current.analysis.images === undefined")
            page.locator("#projectImagePage").fill("201")
            page.locator("#projectImageJump button").click()
            playwright.expect(page.locator("#imageRows tr")).to_have_count(3)
            playwright.expect(page.locator("#imageRows tr").last).to_contain_text("010002.png")

            # Deliberately hold an old search until a newer request has rendered.
            held = []
            page.route("**/images?**", lambda route: held.append(route) if "q=slow" in route.request.url else route.continue_())
            page.locator("#projectImageQuery").fill("slow")
            page.locator("#projectImageSearch button").click()
            page.wait_for_timeout(100)
            assert held
            page.locator("#projectImageQuery").fill("010002")
            page.locator("#projectImageSearch button").click()
            playwright.expect(page.locator("#imageRows tr")).to_have_count(1)
            held[0].fulfill(json=dict(items=[], total=0, offset=0, limit=50))
            page.wait_for_timeout(100)
            playwright.expect(page.locator("#imageRows tr")).to_have_count(1)
            page.unroute("**/images?**")

            page.locator("#openWorkspace").click()
            page.wait_for_function("editor.image === 'synthetic/000000.png' && !editor.loading")
            playwright.expect(page.locator("#workspaceImage option")).to_have_count(50)
            page.locator("#saveAnnotations").click()
            playwright.expect(page.locator("#workspaceStatus")).to_contain_text("已保存 1 个实例")
            # A real pointer starts an unsaved polygon draft; refusing a page
            # change must preserve both the draft and the active image.
            page.locator("#drawBoxTool").click()
            point = page.evaluate("""() => {
                const c=$('detailCanvas'), f=canvasFit(c,editor.view.width,editor.view.height);
                return {x:(f.x+90*f.scale)*c.clientWidth/c.width, y:(f.y+90*f.scale)*c.clientHeight/c.height};
            }""")
            page.locator("#detailCanvas").click(position=point)
            assert page.evaluate("Segmentation.dirty()")
            page.once("dialog", lambda dialog: dialog.dismiss())
            page.locator("#workspaceImageNext").click()
            assert page.evaluate("editor.image") == "synthetic/000000.png"
            assert page.evaluate("Segmentation.dirty()")
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator("#workspaceImageNext").click()
            page.wait_for_function("editor.image === 'synthetic/000050.png' && !editor.loading")
            page.locator("#workspaceImagePrev").click()
            page.wait_for_function("editor.image === 'synthetic/000000.png' && !editor.loading")
            assert page.evaluate("editor.annotationStatus") == "human_reviewed"
            page.locator("#workspaceImagePage").fill("201")
            page.locator("#workspaceImageJump button").click()
            page.wait_for_function("editor.image === 'synthetic/010000.png' && !editor.loading")
            playwright.expect(page.locator("#workspaceImage option")).to_have_count(3)
            page.locator("#workspaceImage").select_option("synthetic/010002.png")
            page.wait_for_function("editor.image === 'synthetic/010002.png' && !editor.loading")
            page.locator("#workspaceImageQuery").fill("no-match")
            page.locator("#workspaceImageSearch button").click()
            playwright.expect(page.locator("#workspaceImageStatus")).to_contain_text("没有匹配图像")
            assert page.evaluate("editor.image") == "synthetic/010002.png"
            page.locator("#workspaceImageQuery").fill("010002")
            page.locator("#workspaceImageSearch button").click()
            playwright.expect(page.locator("#workspaceImage option")).to_have_count(1)
            # Failed pages and failed annotation loads keep the current canvas
            # and can be retried using the same visible controls.
            page.route("**/images?**", lambda route: route.fulfill(status=503, json={"detail": "temporary failure"}))
            page.locator("#workspaceImageSearch button").click()
            playwright.expect(page.locator("#workspaceImageStatus")).to_contain_text("加载失败")
            assert page.evaluate("editor.image") == "synthetic/010002.png"
            page.unroute("**/images?**")
            page.route("**/annotations?**", lambda route: route.fulfill(status=503, json={"detail": "temporary failure"}))
            page.locator("#workspaceImageQuery").fill("000000")
            page.locator("#workspaceImageSearch button").click()
            playwright.expect(page.locator("#workspaceStatus")).to_contain_text("加载失败")
            assert page.evaluate("editor.image") == "synthetic/010002.png"
            page.unroute("**/annotations?**")
            page.locator("#workspaceImageSearch button").click()
            page.wait_for_function("editor.image === 'synthetic/000000.png' && !editor.loading")
            assert page.evaluate("editor.annotationStatus") == "human_reviewed"
            assert all("limit=50" in url for url in image_requests)
            assert not errors
            page.screenshot(path=str(tmp_path / "large-project.png"), full_page=True)
            browser.close()
    finally:
        app_server.should_exit = True
        thread.join(timeout=10)
        listener.close()
