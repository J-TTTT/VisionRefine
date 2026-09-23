"""Preview-only edits of instance masks in original-image coordinates.

OpenCV works on a bounded region for image-guided edits; merge uses sparse
128px tiles so distant parts do not allocate the rectangle between them.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from visionrefine.core.dataset_io.models import Annotation
from visionrefine.core.segmentation import MAX_RUNS, MAX_TILES, TILE_SIZE

MAX_EDIT_PIXELS = 4_194_304
MAX_COMPONENTS = 32


class EditOptions(BaseModel):
    image: str
    objects: list[dict] = Field(max_length=100_000)
    selected_ids: list[str] = Field(min_length=1, max_length=32)
    operation: Literal["smooth", "shrink", "expand", "snap", "merge", "split"]
    radius: int = Field(default=3, ge=1, le=64)
    distance: int = Field(default=10, ge=1, le=32)
    cut_line: list[list[float]] = Field(default_factory=list, max_length=4096)
    cut_width: int = Field(default=3, ge=1, le=31)
    preserve_holes: bool = True
    target_label: str | None = None

    @field_validator("cut_line")
    @classmethod
    def valid_line(cls, value):
        if any(len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) for p in value):
            raise ValueError("The cut line needs finite x/y points")
        return value


def _tile_pixels(tile) -> np.ndarray:
    counts = np.asarray(tile.counts, dtype=np.int32)
    values = np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts)
    return values.reshape(TILE_SIZE, TILE_SIZE)


def _encode_tile(values: np.ndarray, x: int, y: int) -> dict | None:
    flat = values.astype(np.uint8, copy=False).ravel()
    if not np.any(flat):
        return None
    edges = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [flat.size]))
    counts = np.diff(edges).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return dict(x=x, y=y, counts=counts)


def _tile_locations(obj: Annotation):
    if obj.kind == "mask":
        return {(t.x, t.y) for t in obj.mask.tiles}
    positions = set()
    for ring in obj.polygons:
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        for y in range(math.floor(min(ys) / TILE_SIZE) * TILE_SIZE,
                       (math.floor(max(ys) / TILE_SIZE) + 1) * TILE_SIZE, TILE_SIZE):
            for x in range(math.floor(min(xs) / TILE_SIZE) * TILE_SIZE,
                           (math.floor(max(xs) / TILE_SIZE) + 1) * TILE_SIZE, TILE_SIZE):
                positions.add((x, y))
                if len(positions) > MAX_TILES:
                    raise ValueError("轮廓超出本次像素编辑容量；请缩小对象范围")
    return positions


def _polygon_pixels(obj: Annotation, x: int, y: int, size: int = TILE_SIZE):
    result = np.zeros((size, size), np.uint8)
    for ring in obj.polygons:
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        if max(xs) < x or min(xs) > x + size or max(ys) < y or min(ys) > y + size:
            continue
        points = np.asarray([[(round((px - x) * 256), round((py - y) * 256)) for px, py in ring]], np.int32)
        cv2.fillPoly(result, points, 1, shift=8)
    return result


def _object_tile(obj: Annotation, x: int, y: int):
    if obj.kind == "mask":
        tile = next((t for t in obj.mask.tiles if t.x == x and t.y == y), None)
        return _tile_pixels(tile) if tile else np.zeros((TILE_SIZE, TILE_SIZE), np.uint8)
    return _polygon_pixels(obj, x, y)


def _encode_sparse(values: np.ndarray, x0: int, y0: int, image_size: tuple[int, int]):
    height, width = values.shape
    tiles = []
    left = (x0 // TILE_SIZE) * TILE_SIZE
    top = (y0 // TILE_SIZE) * TILE_SIZE
    for y in range(top, y0 + height, TILE_SIZE):
        for x in range(left, x0 + width, TILE_SIZE):
            sx, sy = max(x, x0), max(y, y0)
            ex, ey = min(x + TILE_SIZE, x0 + width), min(y + TILE_SIZE, y0 + height)
            if sx >= ex or sy >= ey:
                continue
            part = values[sy-y0:ey-y0, sx-x0:ex-x0]
            if not np.any(part):
                continue
            full = np.zeros((TILE_SIZE, TILE_SIZE), np.uint8)
            full[sy-y:ey-y, sx-x:ex-x] = part
            if x + TILE_SIZE > image_size[0]:
                full[:, image_size[0]-x:] = 0
            if y + TILE_SIZE > image_size[1]:
                full[image_size[1]-y:, :] = 0
            tile = _encode_tile(full, x, y)
            if tile:
                tiles.append(tile)
            if len(tiles) > MAX_TILES:
                raise ValueError("结果超过 4096 个掩码块，请缩小本次处理范围")
    if sum(len(t["counts"]) for t in tiles) > MAX_RUNS:
        raise ValueError("结果超过掩码压缩容量，请缩小本次处理范围")
    if not tiles:
        raise ValueError("操作使实例完全消失；请使用较小的调整强度")
    return {"encoding": "tile-rle-row-v1", "tile_size": TILE_SIZE, "tiles": tiles}


def _roi(obj: Annotation, image_size, margin: int):
    width, height = image_size
    x1, y1, x2, y2 = obj.bbox
    left, top = max(0, math.floor(x1 - margin)), max(0, math.floor(y1 - margin))
    right, bottom = min(width, math.ceil(x2 + margin)), min(height, math.ceil(y2 + margin))
    if (right-left) * (bottom-top) > MAX_EDIT_PIXELS:
        raise ValueError("选中实例的局部区域超过 419 万像素，请缩小实例或分区处理")
    return left, top, right, bottom


def _roi_mask(obj: Annotation, box):
    left, top, right, bottom = box
    mask = np.zeros((bottom-top, right-left), np.uint8)
    if obj.kind == "polygon":
        for ring in obj.polygons:
            xs, ys = [p[0] for p in ring], [p[1] for p in ring]
            if max(xs) < left or min(xs) > right or max(ys) < top or min(ys) > bottom:
                continue
            points = np.asarray([[(round((x-left)*256), round((y-top)*256)) for x, y in ring]], np.int32)
            cv2.fillPoly(mask, points, 1, shift=8)
    else:
        for tile in obj.mask.tiles:
            sx, sy = max(tile.x, left), max(tile.y, top)
            ex, ey = min(tile.x + TILE_SIZE, right), min(tile.y + TILE_SIZE, bottom)
            if sx < ex and sy < ey:
                source = _tile_pixels(tile)
                mask[sy-top:ey-top, sx-left:ex-left] = source[sy-tile.y:ey-tile.y, sx-tile.x:ex-tile.x]
    return mask


def _holes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return int(np.count_nonzero(hierarchy[0, :, 3] >= 0))


def _components(mask):
    count, _ = cv2.connectedComponents(mask, connectivity=8)
    return count - 1


def _boundary_score(mask, gray):
    median = float(np.median(gray))
    edges = cv2.Canny(gray, max(0, int(.66*median)), min(255, int(1.33*median)+1))
    border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    count = np.count_nonzero(border)
    return round(float(np.count_nonzero((edges > 0) & (border > 0))) / count, 3) if count else 0.0


def _snap(mask, image, distance):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*distance+1, 2*distance+1))
    eroded = cv2.erode(mask, kernel)
    dilated = cv2.dilate(mask, kernel)
    labels = np.where(mask, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    labels[dilated == 0] = cv2.GC_BGD
    labels[eroded > 0] = cv2.GC_FGD
    if not np.any(labels == cv2.GC_FGD):
        # Thin rings may have no pixels farther than distance from both edges.
        inner = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        strongest = np.unravel_index(np.argmax(inner), inner.shape)
        if inner[strongest] <= 0:
            raise ValueError("选中实例没有可用的前景种子")
        labels[strongest] = cv2.GC_FGD
    if not np.any(labels == cv2.GC_BGD):
        raise ValueError("选中区域外缺少背景，请缩小贴边距离")
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, labels, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        raise ValueError("当前颜色与边缘不足以完成贴边，请增加前景/背景笔画或缩小调整距离") from exc
    result = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    # A hard displacement bound also protects saved human geometry.
    result[dilated == 0] = 0
    result[eroded > 0] = 1
    return result


def _updated(original: Annotation, mask: dict) -> dict:
    values = original.model_dump()
    values.update(kind="mask", mask=mask, polygons=None)
    return Annotation.model_validate(values).model_dump()


def _single_edit(obj: Annotation, options: EditOptions, image_path: Path, image_size):
    radius = min(options.radius, 31) if options.operation == "smooth" else options.radius
    margin = max(radius, options.distance) + 3
    box = _roi(obj, image_size, margin)
    mask = _roi_mask(obj, box)
    before_holes, before_components = _holes(mask), _components(mask)
    before_area = int(np.count_nonzero(mask))
    op = options.operation
    if op == "smooth":
        result = cv2.medianBlur(mask, 2*radius+1)
    elif op in {"shrink", "expand"}:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        result = (cv2.erode if op == "shrink" else cv2.dilate)(mask, kernel)
    else:
        with Image.open(image_path) as source:
            crop = source.crop(box).convert("RGB")
        image = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
        result = _snap(mask, image, options.distance)
    after_holes, after_components = _holes(result), _components(result)
    if options.preserve_holes and after_holes < before_holes:
        raise ValueError("这次调整会消除孔洞。减小强度或取消“保护孔洞”后重新预览")
    if not np.any(result):
        raise ValueError("这次调整会清空实例，请减小强度")
    mask_data = _encode_sparse(result, box[0], box[1], image_size)
    updated = _updated(obj, mask_data)
    report = dict(operation=op, before_area=before_area, after_area=int(np.count_nonzero(result)),
                  before_holes=before_holes, after_holes=after_holes,
                  before_components=before_components, after_components=after_components,
                  pixel_changes=int(np.count_nonzero(mask != result)))
    if op == "snap":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        report["edge_score_before"] = _boundary_score(mask, gray)
        report["edge_score_after"] = _boundary_score(result, gray)
    return updated, report


def _merge(items: list[Annotation], image_size, target_label, category_id):
    positions = set()
    for obj in items:
        positions.update(_tile_locations(obj))
        if len(positions) > MAX_TILES:
            raise ValueError("合并结果超过 4096 个掩码块")
    tiles = []
    for x, y in sorted(positions, key=lambda p: (p[1], p[0])):
        part = np.zeros((TILE_SIZE, TILE_SIZE), np.uint8)
        for obj in items:
            part |= _object_tile(obj, x, y)
        if x + TILE_SIZE > image_size[0]:
            part[:, image_size[0]-x:] = 0
        if y + TILE_SIZE > image_size[1]:
            part[image_size[1]-y:, :] = 0
        tile = _encode_tile(part, x, y)
        if tile:
            tiles.append(tile)
    if not tiles or sum(len(t["counts"]) for t in tiles) > MAX_RUNS:
        raise ValueError("合并结果超过掩码容量")
    original = items[0]
    values = original.model_dump()
    values.update(id=f"merged-{uuid.uuid4().hex}", kind="mask", label=target_label, category_id=category_id,
                  mask={"encoding": "tile-rle-row-v1", "tile_size": TILE_SIZE, "tiles": tiles}, polygons=None)
    values["provenance"] = {**values.get("provenance", {}), "merged_from": [o.id for o in items]}
    return Annotation.model_validate(values).model_dump()


def _split(obj: Annotation, options: EditOptions, image_size):
    box = _roi(obj, image_size, 2 + options.cut_width)
    mask = _roi_mask(obj, box)
    work = mask.copy()
    if options.cut_line:
        if len(options.cut_line) < 2:
            raise ValueError("分割线至少需要两个点")
        if any(x < 0 or y < 0 or x >= image_size[0] or y >= image_size[1] for x, y in options.cut_line):
            raise ValueError("分割线超出图像")
        points = np.asarray([[round(x-box[0]), round(y-box[1])] for x, y in options.cut_line], np.int32)
        cv2.polylines(work, [points], False, 0, options.cut_width, lineType=cv2.LINE_8)
    count, labels = cv2.connectedComponents(work, connectivity=8)
    if count < 3:
        raise ValueError("未形成两个独立区域；请让分割线穿过实例的两侧边界")
    if count - 1 > MAX_COMPONENTS:
        raise ValueError("分割产生太多碎片，请先平滑轮廓或调整分割线")
    removed = (mask > 0) & (work == 0)
    if np.any(removed):
        best_distance = np.full(mask.shape, np.inf, np.float32)
        nearest = np.zeros(mask.shape, np.int32)
        for i in range(1, count):
            distances = cv2.distanceTransform((labels != i).astype(np.uint8), cv2.DIST_L2, 3)
            better = removed & (distances < best_distance)
            best_distance[better] = distances[better]
            nearest[better] = i
        labels[removed] = nearest[removed]
    if not np.array_equal((labels > 0).astype(np.uint8), mask):
        raise ValueError("分割未保留原实例的全部像素")
    pieces = []
    for index in range(1, count):
        values = obj.model_dump()
        values.update(id=f"split-{uuid.uuid4().hex}", kind="mask", polygons=None,
                      mask=_encode_sparse((labels == index).astype(np.uint8), box[0], box[1], image_size))
        values["provenance"] = {**values.get("provenance", {}), "split_from": obj.id}
        pieces.append(Annotation.model_validate(values).model_dump())
    return pieces


def preview_edit(options: EditOptions, image_path: Path, image_size: tuple[int, int], labels: list[str]) -> dict:
    allowed = set(labels)
    originals = []
    seen = set()
    for raw in options.objects:
        label = raw.get("label")
        if label not in allowed:
            raise ValueError("实例类别不在项目候选标签中")
        obj = Annotation.model_validate({**raw, "category_id": labels.index(label)})
        if obj.kind not in {"polygon", "mask"} or obj.label not in allowed:
            raise ValueError("实例包含不属于项目的类别或几何类型")
        x1, y1, x2, y2 = obj.bbox
        if x1 < 0 or y1 < 0 or x2 > image_size[0] or y2 > image_size[1]:
            raise ValueError("实例超出原图边界")
        if obj.id in seen:
            raise ValueError("图内实例 ID 重复")
        seen.add(obj.id)
        originals.append(obj)
    if len(set(options.selected_ids)) != len(options.selected_ids) or not set(options.selected_ids) <= seen:
        raise ValueError("要处理的实例不存在或被重复选中")
    chosen = [o for o in originals if o.id in options.selected_ids]
    if options.operation == "merge":
        if len(chosen) < 2:
            raise ValueError("至少选择两个实例才能合并")
        if len({o.label for o in chosen}) > 1 and options.target_label not in allowed:
            raise ValueError("跨类别合并需要明确指定结果类别")
        target_label = options.target_label or chosen[0].label
        result = [_merge(chosen, image_size, target_label, labels.index(target_label))]
        report = dict(operation="merge", source_count=len(chosen), result_count=1,
                      before_area=sum(o.area for o in chosen), after_area=result[0]["area"])
    else:
        if len(chosen) != 1:
            raise ValueError("请只选中一个实例进行这项操作")
        if options.operation == "split":
            result = _split(chosen[0], options, image_size)
            report = dict(operation="split", source_count=1, result_count=len(result),
                          before_area=chosen[0].area, after_area=sum(p["area"] for p in result))
        else:
            updated, report = _single_edit(chosen[0], options, image_path, image_size)
            result = [updated]
    result_objects = []
    inserted = False
    for obj in originals:
        if obj.id in options.selected_ids:
            if not inserted:
                result_objects.extend(result)
                inserted = True
        else:
            result_objects.append(obj.model_dump())
    return dict(objects=result_objects, affected_ids=[o["id"] for o in result], report=report)
