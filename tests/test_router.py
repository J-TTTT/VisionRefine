from visionrefine.core.router import plan_image
from PIL import Image
import visionrefine.server


def test_small_image_goes_direct():
    assert plan_image(1024, 768).strategy == "direct"


def test_coarse_annotations_prefer_region_refinement():
    plan = plan_image(25000, 15000, has_coarse_annotations=True)
    assert plan.strategy == "annotation_crops"


def test_large_unannotated_image_is_tiled():
    plan = plan_image(25000, 15000, model_max_side=1536)
    assert plan.strategy == "overlap_tiles"
    assert plan.estimated_tiles > 1


def test_medium_image_can_resize_whole():
    assert plan_image(3000, 2000, model_max_side=1536).strategy == "resize_whole"


def test_gigapixel_metadata_is_an_explicit_requirement():
    assert Image.MAX_IMAGE_PIXELS is None
