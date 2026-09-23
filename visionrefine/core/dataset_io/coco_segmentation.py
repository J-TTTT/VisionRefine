"""COCO instance segmentation with exact sparse-mask/RLE conversion.

COCO RLE is column-major over the whole image. VisionRefine's tile RLE is
row-major within each 128-pixel tile. Conversion walks foreground runs and
occupied tiles; it never allocates an image-sized bitmap.

The compressed RLE codec follows the Microsoft COCO mask API (Simplified BSD):
https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from visionrefine.core.segmentation import MAX_RUNS, MAX_TILES, TILE_SIZE

from .common import dump_json, finish, make_categories, read_image
from .compatibility import issue
from .models import Annotation, Dataset

MAX_COCO_PIXELS = 2**32 - 1  # The reference COCO mask API stores runs as uint32.


def decode_coco_counts(value: str | list[int], width: int, height: int) -> list[int]:
    """Validate and expand compressed or uncompressed COCO column-major RLE."""
    if width <= 0 or height <= 0 or width * height > MAX_COCO_PIXELS:
        raise ValueError("COCO RLE image dimensions exceed the 32-bit pixel limit")
    if isinstance(value, list):
        counts = value
    elif isinstance(value, str):
        counts = []
        index = 0
        if len(value) > 8_000_000:
            raise ValueError("COCO RLE is too long")
        while index < len(value):
            number = shift = 0
            while True:
                if index == len(value) or shift > 35:
                    raise ValueError("Truncated or oversized COCO RLE")
                char = ord(value[index]) - 48
                index += 1
                if char < 0 or char > 63:
                    raise ValueError("Invalid COCO RLE character")
                number |= (char & 31) << shift
                shift += 5
                if not char & 32:
                    if char & 16:
                        number -= 1 << shift
                    break
            if len(counts) > 2:
                number += counts[-2]
            counts.append(number)
    else:
        raise ValueError("COCO RLE counts must be a string or integer list")
    if (not counts or len(counts) > MAX_RUNS * 4 or
            any(type(n) is not int or n < 0 or (i > 0 and n == 0) for i, n in enumerate(counts)) or
            sum(counts) != width * height or not sum(counts[1::2])):
        raise ValueError("Invalid, empty, or oversized COCO RLE counts")
    return counts


def encode_coco_counts(counts: list[int]) -> str:
    """Encode validated runs using COCO's signed 5-bit delta representation."""
    output = []
    for index, count in enumerate(counts):
        value = count - counts[index - 2] if index > 2 else count
        while True:
            chunk = value & 31
            value >>= 5
            more = value != (-1 if chunk & 16 else 0)
            output.append(chr(chunk + (32 if more else 0) + 48))
            if not more:
                break
    return "".join(output)


def _tile_from_pixels(pixels, x: int, y: int) -> dict | None:
    import numpy as np

    flat = np.asarray(pixels, dtype=np.uint8).reshape(-1)
    if not np.any(flat):
        return None
    boundaries = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [flat.size]))
    counts = np.diff(boundaries).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return dict(x=x, y=y, counts=counts)


def _tile_pixels(tile):
    import numpy as np

    counts = np.asarray(tile.counts, dtype=np.int32)
    return np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts).reshape(TILE_SIZE, TILE_SIZE)


def _native_mask(tiles: dict[tuple[int, int], object]) -> dict:
    encoded = []
    for (x, y), pixels in sorted(tiles.items(), key=lambda item: (item[0][1], item[0][0])):
        tile = _tile_from_pixels(pixels, x, y)
        if tile:
            encoded.append(tile)
    if not encoded or len(encoded) > MAX_TILES or sum(len(t["counts"]) for t in encoded) > MAX_RUNS:
        raise ValueError("Mask is empty or exceeds VisionRefine's sparse-mask limit")
    return dict(encoding="tile-rle-row-v1", tile_size=TILE_SIZE, tiles=encoded)


def mask_from_coco_counts(counts: list[int], width: int, height: int) -> dict:
    """Scatter foreground column intervals into the occupied native tiles."""
    import numpy as np

    tiles = {}
    offset = 0
    for index, run in enumerate(counts):
        if index % 2 and run:
            end = offset + run
            first_x, first_y = divmod(offset, height)
            last_x = (end - 1) // height
            for x in range(first_x, last_x + 1):
                start_y = first_y if x == first_x else 0
                stop_y = (end - 1) % height + 1 if x == last_x else height
                y = start_y
                while y < stop_y:
                    tile_x, tile_y = x // TILE_SIZE * TILE_SIZE, y // TILE_SIZE * TILE_SIZE
                    key = tile_x, tile_y
                    if key not in tiles:
                        if len(tiles) >= MAX_TILES:
                            raise ValueError("COCO mask exceeds 4096 native tiles")
                        tiles[key] = np.zeros((TILE_SIZE, TILE_SIZE), np.uint8)
                    last_y = min(stop_y, tile_y + TILE_SIZE)
                    tiles[key][y-tile_y:last_y-tile_y, x-tile_x] = 1
                    y = last_y
        offset += run
    return _native_mask(tiles)


def coco_counts_from_mask(mask, width: int, height: int) -> list[int]:
    """Emit COCO column-major runs, skipping empty columns in constant time."""
    import numpy as np

    if width * height > MAX_COCO_PIXELS:
        raise ValueError("COCO RLE cannot represent an image above 2³²−1 pixels")
    bands = defaultdict(list)
    for tile in mask.tiles:
        bands[tile.x].append((tile.y, _tile_pixels(tile)))
    counts = []
    bit = 0
    length = 0

    def append(value, size):
        nonlocal bit, length
        if size <= 0:
            return
        if value != bit:
            counts.append(length)
            bit, length = value, size
        else:
            length += size
        if len(counts) > MAX_RUNS * 4:
            raise ValueError("COCO RLE exceeds the export run limit")

    cursor_x = 0
    for tile_x, rows in sorted(bands.items()):
        append(0, (tile_x - cursor_x) * height)
        rows.sort(key=lambda item: item[0])
        stop_x = min(tile_x + TILE_SIZE, width)
        for x in range(tile_x, stop_x):
            cursor_y = 0
            for tile_y, pixels in rows:
                append(0, tile_y - cursor_y)
                values = pixels[:min(TILE_SIZE, height - tile_y), x - tile_x]
                edges = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1, [values.size]))
                for start, end in zip(edges[:-1], edges[1:]):
                    append(int(values[start]), int(end-start))
                cursor_y = tile_y + len(values)
            append(0, height - cursor_y)
        cursor_x = stop_x
    append(0, (width - cursor_x) * height)
    counts.append(length)
    if sum(counts) != width * height:
        raise ValueError("COCO RLE conversion produced an invalid mask length")
    return counts


def _parse_polygons(raw):
    if not isinstance(raw, list) or not raw:
        raise ValueError("COCO polygon segmentation must contain at least one ring")
    polygons = []
    for values in raw:
        if not isinstance(values, list) or len(values) < 6 or len(values) % 2:
            raise ValueError("COCO polygon needs at least three x/y pairs")
        ring = []
        for index in range(0, len(values), 2):
            x, y = values[index:index+2]
            if type(x) not in (int, float) or type(y) not in (int, float) or not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("COCO polygon coordinates must be finite numbers")
            ring.append([float(x), float(y)])
        if ring[-1] == ring[0]:
            ring.pop()
        polygons.append(ring)
    return polygons


def _rasterize_polygons(polygons, width: int, height: int) -> dict:
    import numpy as np

    locations = set()
    for ring in polygons:
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        left, right = max(0, math.floor(min(xs) / TILE_SIZE)), min((width-1)//TILE_SIZE, math.floor(max(xs) / TILE_SIZE))
        top, bottom = max(0, math.floor(min(ys) / TILE_SIZE)), min((height-1)//TILE_SIZE, math.floor(max(ys) / TILE_SIZE))
        for ty in range(top, bottom+1):
            for tx in range(left, right+1):
                locations.add((tx*TILE_SIZE, ty*TILE_SIZE))
                if len(locations) > MAX_TILES:
                    raise ValueError("COCO polygon exceeds 4096 native tiles")
    tiles = {}
    for x, y in locations:
        canvas = Image.new("1", (TILE_SIZE, TILE_SIZE), 0)
        drawer = ImageDraw.Draw(canvas)
        for ring in polygons:
            drawer.polygon([(px-x, py-y) for px, py in ring], fill=1)
        pixels = np.asarray(canvas, dtype=np.uint8).copy()
        pixels[:, max(0, width-x):] = 0
        pixels[max(0, height-y):, :] = 0
        tiles[(x, y)] = pixels
    return _native_mask(tiles)


class CocoSegmentation:
    def read(self, root: Path, source: Path | None, labels: list[str], split: str) -> Dataset:
        if source is None:
            raise ValueError("COCO Instance Segmentation requires an annotation JSON file")
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or any(not isinstance(data.get(k), list) for k in ("images", "categories", "annotations")):
            raise ValueError("COCO requires images, categories and annotations arrays")
        dataset = Dataset(task="instance_segmentation")
        extra = sorted(set(data) - {"images", "categories", "annotations"})
        if extra:
            dataset.report.add("unsupported_fields", "dataset", f"Omitted top-level fields: {', '.join(extra)}")
        try:
            dataset.categories, categories = make_categories([(c["id"], c["name"]) for c in data["categories"]], dataset.report)
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid COCO category catalog") from exc
        images, seen_ids, seen_paths = {}, set(), set()
        for row in data["images"]:
            location = str(row.get("id", "unknown")) if isinstance(row, dict) else "images"
            try:
                image_id, relative = row["id"], row["file_name"]
                if type(image_id) is not int or image_id in seen_ids:
                    raise ValueError("Missing/noninteger or duplicate image ID")
                seen_ids.add(image_id)
                image = read_image(root, relative, row.get("split", split), dataset.report)
                if image is None:
                    continue
                if image.path in seen_paths:
                    raise ValueError("Duplicate image file_name")
                seen_paths.add(image.path)
                if (row.get("width"), row.get("height")) != (image.width, image.height):
                    dataset.report.add("dimension_mismatch", relative, "Using actual local image dimensions")
                image.status = "imported_coarse"
                image.provenance = {"source_id": image_id}
                images[image_id] = image
                dataset.images.append(image)
            except (KeyError, TypeError, ValueError) as exc:
                dataset.report.skipped_images += 1
                dataset.report.add("invalid_image", location, str(exc), "error")
        seen_annotations = set()
        for index, row in enumerate(data["annotations"]):
            location = f"annotations[{index}]"
            try:
                annotation_id = row["id"]
                if type(annotation_id) is not int or annotation_id in seen_annotations:
                    raise ValueError("Missing/noninteger or duplicate annotation ID")
                seen_annotations.add(annotation_id)
                image = images.get(row["image_id"])
                if image is None:
                    raise ValueError("Annotation references a missing or unreadable image")
                category = categories.get(row["category_id"])
                if category is None:
                    raise ValueError("Unknown category_id")
                crowd = row.get("iscrowd", 0)
                if type(crowd) is not int or crowd not in (0, 1):
                    raise ValueError("iscrowd must be 0 or 1")
                segmentation = row.get("segmentation")
                values = dict(id=f"import-{annotation_id}", category_id=category.id, label=category.name,
                              confidence=row.get("score"),
                              attributes={"iscrowd": crowd},
                              provenance={"source_id": annotation_id, "source_category_id": row["category_id"]})
                if isinstance(segmentation, dict):
                    if segmentation.get("size") != [image.height, image.width]:
                        raise ValueError("COCO RLE size differs from actual image dimensions")
                    counts = decode_coco_counts(segmentation.get("counts"), image.width, image.height)
                    values.update(kind="mask", mask=mask_from_coco_counts(counts, image.width, image.height))
                else:
                    polygons = _parse_polygons(segmentation)
                    values.update(kind="polygon", polygons=polygons)
                    try:
                        annotation = Annotation.model_validate(values)
                        if annotation.bbox[0] < 0 or annotation.bbox[1] < 0 or annotation.bbox[2] > image.width or annotation.bbox[3] > image.height:
                            raise ValueError("COCO polygon reaches outside the image")
                    except ValueError:
                        values.update(kind="mask", mask=_rasterize_polygons(polygons, image.width, image.height), polygons=None)
                        dataset.report.add("polygon_rasterized", location, "Overlapping, complex, or clipped COCO polygons were rasterized to one mask")
                annotation = Annotation.model_validate(values)
                image.objects.append(annotation)
                ignored = sorted(set(row) - {"id", "image_id", "category_id", "segmentation", "bbox", "area", "iscrowd", "score"})
                if ignored:
                    dataset.report.add("unsupported_fields", location, f"Omitted annotation fields: {', '.join(ignored)}")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                dataset.report.skipped_objects += 1
                dataset.report.add("invalid_annotation", location, str(exc), "error")
        return finish(dataset)

    def preflight(self, dataset: Dataset, include_images: bool) -> list[dict]:
        large = [image.path for image in dataset.images if image.width * image.height > MAX_COCO_PIXELS
                 and any(obj.kind == "mask" for obj in image.objects)]
        return [issue("coco_rle_size_limit", "COCO RLE cannot encode masks on images exceeding 2³²−1 pixels.",
                      len(large), "blocking", large)] if large else []

    def write(self, dataset: Dataset, output: Path) -> dict:
        category_ids, used = {}, set()
        for category in dataset.categories:
            candidate = category.source_id
            if candidate is None or candidate < 0 or candidate in used:
                candidate = 1
                while candidate in used:
                    candidate += 1
            category_ids[category.id] = candidate
            used.add(candidate)
        rows, annotations, image_paths, split_counts = [], [], {}, {}
        for image_id, image in enumerate(dataset.images, 1):
            rows.append(dict(id=image_id, file_name=image.path, width=image.width, height=image.height))
            split_counts.setdefault(image.split, []).append(image_id)
            image_paths[image.path] = f"images/{image.path}"
            for obj in image.objects:
                x1, y1, x2, y2 = obj.bbox
                if obj.kind == "mask":
                    counts = coco_counts_from_mask(obj.mask, image.width, image.height)
                    segmentation = dict(size=[image.height, image.width], counts=encode_coco_counts(counts))
                elif obj.kind == "polygon":
                    segmentation = [[coordinate for point in ring for coordinate in point] for ring in obj.polygons]
                else:
                    raise ValueError("COCO Instance Segmentation requires polygons or masks")
                annotations.append(dict(id=len(annotations)+1, image_id=image_id,
                    category_id=category_ids[obj.category_id], bbox=[x1, y1, x2-x1, y2-y1],
                    area=obj.area, segmentation=segmentation, iscrowd=obj.attributes.get("iscrowd", 0)))
        categories = [dict(id=category_ids[c.id], name=c.name) for c in dataset.categories]
        dump_json(output / "annotations.json", dict(images=rows, annotations=annotations, categories=categories))
        for split, ids in split_counts.items():
            subset = set(ids)
            dump_json(output / "annotations" / f"instances_{split}.json", dict(
                images=[i for i in rows if i["id"] in subset],
                annotations=[a for a in annotations if a["image_id"] in subset], categories=categories))
        return dict(image_paths=image_paths,
                    category_mapping=[dict(label=c.name, id=category_ids[c.id]) for c in dataset.categories],
                    warnings=["Image/annotation IDs are regenerated; confidence and workflow metadata remain in visionrefine.json.",
                              "annotations.json combines all splits; use annotations/instances_<split>.json for split-specific files."])
