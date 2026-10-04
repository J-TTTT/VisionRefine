"""Opt-in pointer acceptance for the reusable exact-frame segmentation editor."""
import base64
import io
import os
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(
    os.environ.get("VISIONREFINE_BROWSER_TESTS") != "1", reason="Opt-in browser acceptance"
)


@pytest.fixture
def canvas_page():
    playwright = pytest.importorskip("playwright.sync_api")
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), "#cdd5df").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 800}, device_scale_factor=2)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_content('<style>canvas{width:640px;height:480px;display:block}</style>'
                         '<canvas id="canvas"></canvas><input id="text">')
        source = Path(__file__).resolve().parents[1] / "visionrefine/static/video-segmentation-canvas.js"
        page.add_script_tag(path=str(source))
        page.evaluate("""url => {
          window.changes=[]; window.states=[]; window.selections=[];
          window.editor=VideoSegmentationCanvas.create({canvas:document.getElementById('canvas'),
            onChange:(geometry,meta)=>changes.push({geometry,meta}),
            onStatus:state=>states.push(state),onSelection:id=>selections.push(id)});
          window.frame={imageUrl:url,width:640,height:480,key:'video:object:0'};
        }""", url)
        assert page.evaluate("editor.load(frame)")
        yield page
        assert not errors
        browser.close()


def _point(page, x, y):
    return page.evaluate("""([x,y]) => {
      const r=document.getElementById('canvas').getBoundingClientRect(),v=editor.getState().view;
      return [r.left+v.x+x*v.scale,r.top+v.y+y*v.scale];
    }""", [x, y])


def _click(page, x, y):
    page.mouse.click(*_point(page, x, y))


def _drag(page, a, b):
    page.mouse.move(*_point(page, *a))
    page.mouse.down()
    page.mouse.move(*_point(page, *b), steps=5)
    page.mouse.up()


def _geometry(page):
    return page.evaluate("editor.getGeometry()")


def _mask_at(geometry, x, y):
    for tile in geometry["mask"]["tiles"]:
        if tile["x"] <= x < tile["x"] + 128 and tile["y"] <= y < tile["y"] + 128:
            target = (y-tile["y"])*128+x-tile["x"]
            offset = 0
            for i, n in enumerate(tile["counts"]):
                if offset <= target < offset+n:
                    return i % 2
                offset += n
    return 0


def test_video_canvas_polygons_masks_history_and_drafts(canvas_page):
    page = canvas_page
    assert page.evaluate("editor.getState().ready && editor.canSave()")
    assert page.locator("#canvas").evaluate("c => c.width") == 1280
    for p in [(64, 100), (240, 100), (240, 280), (64, 280)]:
        _click(page, *p)
    assert page.evaluate("editor.isDraftDirty() && !editor.canSave()")
    assert not page.evaluate("editor.load({...frame,key:'video:object:1'})")
    assert page.evaluate("editor.getState().key") == "video:object:0"
    assert not page.evaluate("editor.setEnabled(false)")
    page.keyboard.press("Enter")
    assert _geometry(page)["kind"] == "polygon"
    assert page.evaluate("editor.canSave() && !editor.isDraftDirty()")
    assert page.evaluate("changes.at(-1).meta.key") == "video:object:0"

    _drag(page, (64, 100), (80, 100))
    assert _geometry(page)["polygons"][0][0] == pytest.approx([80, 100], abs=.5)
    page.keyboard.down("Shift")
    _click(page, 160, 100)
    page.keyboard.up("Shift")
    assert len(_geometry(page)["polygons"][0]) == 5
    assert page.evaluate("editor.deleteVertex()")
    assert len(_geometry(page)["polygons"][0]) == 4
    assert page.evaluate("editor.appendContour()")
    for p in [(350, 120), (400, 120), (400, 170), (350, 170)]:
        _click(page, *p)
    page.keyboard.press("Enter")
    assert len(_geometry(page)["polygons"]) == 2
    polygon = _geometry(page)
    _click(page, 350, 120)
    assert page.evaluate("editor.deleteContour()")
    assert len(_geometry(page)["polygons"]) == 1
    assert page.evaluate("editor.undo()")
    assert _geometry(page) == polygon

    # Transforms preserve original coordinates at devicePixelRatio=2 and after zoom.
    page.mouse.move(*_point(page, 200, 200))
    page.mouse.wheel(0, -240)
    page.wait_for_function("editor.getState().zoom > 1")
    _drag(page, (80, 100), (90, 110))
    assert _geometry(page)["polygons"][0][0] == pytest.approx([90, 110], abs=.8)
    page.keyboard.press("Control+z")
    assert _geometry(page) == polygon
    page.evaluate("editor.fit();editor.setRadius(16);editor.setTool('erase')")
    _click(page, 150, 200)
    mask = _geometry(page)
    assert mask["kind"] == "mask"
    assert _mask_at(mask, 150, 200) == 0
    assert _mask_at(mask, 150, 240) == 1
    assert _mask_at(mask, 370, 140) == 1
    from visionrefine.core.segmentation import TileMask
    TileMask.model_validate(mask["mask"])
    page.keyboard.press("Control+z")
    assert _geometry(page) == polygon
    page.keyboard.press("Control+Shift+z")
    assert _geometry(page) == mask
    page.keyboard.press("b")
    _click(page, 150, 200)
    assert _mask_at(_geometry(page), 150, 200) == 1
    page.evaluate("editor.undo()")
    assert _geometry(page) == mask

    # Frame/object history is isolated and empty erasure produces null geometry.
    assert page.evaluate("editor.load({...frame,key:'video:object:1'})")
    assert not page.evaluate("editor.undo()")
    assert _geometry(page) is None
    page.evaluate("editor.setTool('brush');editor.setRadius(6)")
    _drag(page, (510, 300), (515, 300))
    assert _mask_at(_geometry(page), 511, 300) == 1
    assert _mask_at(_geometry(page), 512, 300) == 1
    page.evaluate("editor.setTool('erase');editor.setRadius(30)")
    _click(page, 512, 300)
    assert _geometry(page) is None
    assert page.evaluate("editor.undo()")
    assert _geometry(page)["kind"] == "mask"

    # Keyboard shortcuts remain local to the focused canvas.
    before = _geometry(page)
    page.locator("#text").fill("N B E")
    page.keyboard.press("Control+z")
    assert _geometry(page) == before
    assert page.evaluate("editor.setEnabled(false)")
    _click(page, 80, 80)
    assert _geometry(page) == before


def test_video_canvas_invalid_geometry_async_frames_and_cancelled_strokes(canvas_page):
    page = canvas_page
    # Self intersection remains an explicit draft, with no saved callback.
    for p in [(100, 100), (250, 250), (250, 100), (100, 250)]:
        _click(page, *p)
    page.keyboard.press("Enter")
    assert page.evaluate("editor.isDraftDirty()")
    assert _geometry(page) is None
    assert page.evaluate("changes.length") == 0
    page.keyboard.press("Escape")
    assert not page.evaluate("editor.isDraftDirty()")
    assert not page.evaluate("editor.setGeometry({kind:'polygon',polygons:[[[0,0],[641,0],[0,30]]]})")
    assert _geometry(page) is None

    # pointercancel rolls back the whole stroke instead of accepting half a mask.
    page.evaluate("editor.setTool('brush');editor.setRadius(12)")
    page.mouse.move(*_point(page, 150, 150))
    page.mouse.down()
    page.mouse.move(*_point(page, 200, 150))
    assert page.evaluate("editor.isDraftDirty()")
    page.keyboard.press("Escape")
    page.mouse.up()
    assert _geometry(page) is None
    assert page.evaluate("changes.length") == 0
    assert page.evaluate("editor.canSave()")

    # A loaded image with unexpected dimensions cannot be edited.
    assert not page.evaluate("editor.load({...frame,width:320,key:'video:object:2'})")
    assert not page.evaluate("editor.getState().ready")
    _click(page, 100, 100)
    assert page.evaluate("changes.length") == 0

    # A late response for an earlier frame never overwrites the current frame.
    page.route("https://frames.test/slow.png", lambda route: route.fulfill(
        status=200, content_type="image/png", body=base64.b64decode(
            page.evaluate("frame.imageUrl.split(',')[1]")
        )))
    result = page.evaluate("""async () => {
      const first=editor.load({...frame,imageUrl:'https://frames.test/slow.png',key:'old'});
      const second=editor.load({...frame,key:'new'});
      return await Promise.all([first,second]);
    }""")
    assert result == [False, True]
    assert page.evaluate("editor.getState().key") == "new"
    assert page.evaluate("editor.getState().ready")

    # Silent refinement preview can restore then commit with a reversible history.
    original = {"kind": "polygon", "polygons": [[[30, 30], [130, 30], [130, 130], [30, 130]]]}
    refined = {"kind": "polygon", "polygons": [[[31, 31], [129, 31], [129, 129], [31, 129]]]}
    page.evaluate("g => editor.setGeometry(g)", original)
    page.evaluate("g => editor.setGeometry(g)", refined)
    assert page.evaluate("changes.length") == 0
    page.evaluate("g => editor.setGeometry(g)", original)
    page.evaluate("g => editor.setGeometry(g,{silent:false})", refined)
    assert _geometry(page) == refined
    assert page.evaluate("editor.undo()")
    assert _geometry(page) == original
