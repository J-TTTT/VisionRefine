from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from visionrefine.core.pilot import run_detection_crop


@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    width: int
    height: int


def _axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def build_tiles(width: int, height: int, tile_size: int, overlap: float = 0.2) -> list[Tile]:
    """Cover an image deterministically; crop origins are MCU-aligned for JPEG."""
    xs = _axis_starts(width, tile_size, overlap)
    ys = _axis_starts(height, tile_size, overlap)
    tiles = []
    seen = set()
    for y in ys:
        for x in xs:
            aligned_x, aligned_y = (x // 16) * 16, (y // 16) * 16
            key = (aligned_x, aligned_y)
            if key in seen:
                continue
            seen.add(key)
            tiles.append(Tile(
                len(tiles), aligned_x, aligned_y,
                min(tile_size + 15, width - aligned_x),
                min(tile_size + 15, height - aligned_y),
            ))
    return tiles


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def class_aware_nms(objects: list[dict], threshold: float = 0.5) -> list[dict]:
    ordered = sorted(
        objects,
        key=lambda item: (-float(item.get("confidence", 0.0)), str(item.get("label", "")), item["bbox"]),
    )
    kept: list[dict] = []
    for candidate in ordered:
        duplicate = any(
            candidate["label"] == existing["label"]
            and box_iou(candidate["bbox"], existing["bbox"]) >= threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    for index, item in enumerate(kept):
        item["id"] = f"ai-initial-{index}"
    return kept


def run_tiled_detection(
    image_path: Path,
    adapter: dict,
    tile_size: int,
    labels: list[str],
    *,
    overlap: float = 0.2,
    nms_threshold: float = 0.5,
    workers: int = 4,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    tiles = build_tiles(width, height, tile_size, overlap)

    def detect(tile: Tile) -> dict:
        result = run_detection_crop(
            image_path, adapter, (tile.x, tile.y, tile.width, tile.height), labels,
            id_prefix=f"tile-{tile.index}",
        )
        return {"tile": tile, "result": result}

    completed = 0
    raw_objects: list[dict] = []
    records = []
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tiles)))) as executor:
        futures = {executor.submit(detect, tile): tile for tile in tiles}
        for future in as_completed(futures):
            tile = futures[future]
            try:
                item = future.result()
                raw_objects.extend(item["result"]["objects"])
                records.append({
                    "index": tile.index,
                    "crop": item["result"]["crop"],
                    "object_count": len(item["result"]["objects"]),
                    "summary": item["result"]["summary"],
                    "raw_response": item["result"]["raw_response"],
                    "usage": item["result"]["usage"],
                })
            except Exception as exc:  # Per-tile failure should not discard completed inference.
                errors.append({"index": tile.index, "error": str(exc)[:1000]})
            completed += 1
            if progress:
                progress(completed, len(tiles), len(raw_objects))

    if errors and len(errors) == len(tiles):
        raise RuntimeError(f"All {len(tiles)} tile requests failed; first error: {errors[0]['error']}")
    objects = class_aware_nms(raw_objects, nms_threshold)
    return {
        "original_size": {"width": width, "height": height},
        "tile_size": tile_size,
        "overlap": overlap,
        "nms_threshold": nms_threshold,
        "tile_count": len(tiles),
        "objects_before_nms": len(raw_objects),
        "objects": objects,
        "tile_records": sorted(records, key=lambda item: item["index"]),
        "errors": sorted(errors, key=lambda item: item["index"]),
    }
