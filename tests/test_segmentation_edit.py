"""Pixel-meaningful tests of contour editing, holes, and topology."""
import json
from pathlib import Path

import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from visionrefine.core.segmentation_edit import EditOptions, _encode_sparse, _roi_mask, preview_edit


def annotation(name, pixels, shape, *, x0=0, y0=0, label="ring"):
    return dict(id=name, kind="mask", label=label, category_id=0,
                mask=_encode_sparse(pixels.astype(np.uint8), x0, y0, shape))


def request(operation, image, objects, selected, **kwargs):
    return preview_edit(EditOptions(image="test.png", objects=objects, selected_ids=selected, operation=operation, **kwargs),
                        image, Image.open(image).size, ["ring", "other"])


def decoded(object_, shape):
    from visionrefine.core.dataset_io.models import Annotation
    annotation = Annotation.model_validate(object_)
    return _roi_mask(annotation, (0, 0, *shape))


def test_ring_smooth_shrink_expand_and_hole_guard(tmp_path):
    size = (256, 256)
    source = tmp_path / "test.png"
    Image.new("RGB", size, "white").save(source)
    yy, xx = np.mgrid[:256, :256]
    rr = (xx-128)**2 + (yy-128)**2
    pixels = ((rr <= 70**2) & (rr >= 30**2)).astype(np.uint8)
    pixels[128, 196:202] = 1  # a protruding jagged edge
    original = annotation("ring", pixels, size)
    smooth = request("smooth", source, [original], ["ring"], radius=3)
    assert smooth["report"]["after_holes"] == 1
    assert smooth["report"]["pixel_changes"] > 0
    assert decoded(smooth["objects"][0], size)[128, 201] == 0
    shrink = request("shrink", source, [original], ["ring"], radius=3)
    expand = request("expand", source, [original], ["ring"], radius=3)
    assert shrink["report"]["after_area"] < pixels.sum() < expand["report"]["after_area"]
    assert shrink["report"]["after_holes"] == expand["report"]["after_holes"] == 1
    with pytest.raises(ValueError, match="孔洞"):
        request("expand", source, [original], ["ring"], radius=35)
    unprotected = request("expand", source, [original], ["ring"], radius=35, preserve_holes=False)
    assert unprotected["report"]["after_holes"] == 0


def test_snap_improves_boundary_on_high_contrast_ring(tmp_path):
    size = (240, 240)
    yy, xx = np.mgrid[:240, :240]
    rr = (xx-120)**2 + (yy-120)**2
    truth = ((rr <= 70**2) & (rr >= 31**2)).astype(np.uint8)
    initial = ((rr <= 64**2) & (rr >= 25**2)).astype(np.uint8)
    rgb = np.full((240, 240, 3), 220, np.uint8)
    rgb[truth > 0] = (45, 110, 160)
    image = tmp_path / "test.png"
    Image.fromarray(rgb).save(image)
    before = annotation("ring", initial, size)
    result = request("snap", image, [before], ["ring"], distance=10)
    after = decoded(result["objects"][0], size)
    def iou(values):
        return np.count_nonzero(values & truth) / np.count_nonzero(values | truth)
    assert iou(after) > iou(initial) + .01
    assert result["report"]["after_holes"] == 1
    assert result["report"]["edge_score_after"] >= result["report"]["edge_score_before"]


def test_sparse_merge_distant_parts_and_mixed_class(tmp_path):
    size = (100_000, 100_000)
    # No source pixels are needed for a sparse union; only metadata is read.
    image = tmp_path / "test.png"
    Image.new("RGB", (1, 1)).save(image)
    a = annotation("a", np.array([[1]], np.uint8), size, x0=128, y0=256)
    b = annotation("b", np.array([[1]], np.uint8), size, x0=99840, y0=99840, label="other")
    options = EditOptions(image="test.png", objects=[a,b], selected_ids=["a","b"], operation="merge", target_label="other")
    result = preview_edit(options, image, size, ["ring", "other"])
    item = result["objects"][0]
    assert len(result["objects"]) == 1
    assert item["label"] == "other" and item["category_id"] == 1
    assert item["area"] == 2 and len(item["mask"]["tiles"]) == 2
    assert item["provenance"]["merged_from"] == ["a", "b"]
    with pytest.raises(ValueError, match="类别"):
        preview_edit(EditOptions(image="test.png", objects=[a,b], selected_ids=["a","b"], operation="merge"),
                     image, size, ["ring", "other"])


def test_split_connected_parts_and_manual_cut_without_losing_pixels(tmp_path):
    size = (256, 256)
    image = tmp_path / "test.png"
    Image.new("RGB", size, "white").save(image)
    separated = np.zeros((256,256), np.uint8)
    separated[30:80,30:80] = 1
    separated[160:210,160:210] = 1
    original = annotation("parts", separated, size)
    result = request("split", image, [original], ["parts"])
    assert len(result["objects"]) == 2
    pieces = [decoded(o,size) for o in result["objects"]]
    assert np.array_equal(pieces[0] | pieces[1], separated)
    assert np.count_nonzero(pieces[0] & pieces[1]) == 0
    joined = np.zeros((256,256), np.uint8)
    joined[40:210,35:220] = 1
    original = annotation("joined", joined, size)
    with pytest.raises(ValueError, match="两个独立区域"):
        request("split", image, [original], ["joined"])
    cut = request("split", image, [original], ["joined"], cut_line=[[125, 20],[125, 230]], cut_width=3)
    assert len(cut["objects"]) == 2
    parts = [decoded(o,size) for o in cut["objects"]]
    assert np.array_equal(parts[0] | parts[1], joined)
    assert np.count_nonzero(parts[0] & parts[1]) == 0
    assert cut["report"]["before_area"] == cut["report"]["after_area"]


def test_edit_rejects_invalid_selection_and_is_preview_only(tmp_path):
    image = tmp_path / "test.png"
    Image.new("RGB", (256,256), "white").save(image)
    pixels = np.zeros((256,256), np.uint8)
    pixels[30:60,30:60] = 1
    item = annotation("a", pixels, (256,256))
    before = json.dumps(item, sort_keys=True)
    with pytest.raises(ValueError, match="不存在"):
        request("smooth", image, [item], ["missing"])
    result = request("smooth", image, [item], ["a"])
    assert json.dumps(item, sort_keys=True) == before
    assert result["objects"][0]["id"] == "a"
