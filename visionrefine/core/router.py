from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil


@dataclass(frozen=True)
class RoutePlan:
    strategy: str
    reason: str
    tile_width: int | None = None
    tile_height: int | None = None
    overlap: float | None = None
    estimated_tiles: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def plan_image(
    width: int,
    height: int,
    *,
    model_max_side: int = 1536,
    has_coarse_annotations: bool = False,
    task: str = "detection",
) -> RoutePlan:
    """Choose a deterministic model-input strategy from image metadata."""
    if width <= 0 or height <= 0 or model_max_side < 256:
        raise ValueError("invalid image dimensions or model input limit")

    max_side = max(width, height)
    megapixels = width * height / 1_000_000

    if max_side <= model_max_side and megapixels <= 8:
        return RoutePlan("direct", "Image fits the configured model input.")

    if has_coarse_annotations and task in {
        "detection", "instance_segmentation", "grounding", "ocr"
    }:
        return RoutePlan(
            "annotation_crops",
            "Existing regions can guide high-resolution refinement; a whole-image pass remains for gap checking.",
        )

    if megapixels <= 20 and max_side <= model_max_side * 3:
        return RoutePlan(
            "resize_whole",
            "The image is moderately larger than the model limit and can retain useful global detail after resizing.",
        )

    overlap = 0.2
    tile = model_max_side
    stride = max(1, round(tile * (1 - overlap)))
    cols = max(1, ceil(max(0, width - tile) / stride) + 1)
    rows = max(1, ceil(max(0, height - tile) / stride) + 1)
    return RoutePlan(
        "overlap_tiles",
        "The image is too large for reliable whole-image analysis; overlapping tiles preserve local detail.",
        tile_width=tile,
        tile_height=tile,
        overlap=overlap,
        estimated_tiles=cols * rows,
    )

