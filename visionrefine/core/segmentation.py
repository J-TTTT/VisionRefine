"""Original-pixel geometry. Sparse masks use 128px tiles and row-major RLE.

This is an internal encoding, deliberately independent of COCO's column-major RLE.
No operation here allocates a full image mask.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field, StrictInt, model_validator
from typing import Literal

TILE_SIZE = 128
MAX_TILES = 4096
MAX_RUNS = 1_000_000
MAX_VERTICES = 2048


class MaskTile(BaseModel):
    x: StrictInt = Field(ge=0)
    y: StrictInt = Field(ge=0)
    counts: list[StrictInt] = Field(min_length=2, max_length=TILE_SIZE * TILE_SIZE + 1)

    @model_validator(mode="after")
    def valid_rle(self):
        if self.x % TILE_SIZE or self.y % TILE_SIZE:
            raise ValueError("Mask tiles must be aligned to 128-pixel boundaries")
        if any(n < 0 or (i > 0 and n == 0) for i, n in enumerate(self.counts)):
            raise ValueError("RLE runs must be positive, except the first background run")
        if sum(self.counts) != TILE_SIZE ** 2 or not sum(self.counts[1::2]):
            raise ValueError("Mask tile must cover 128×128 pixels and contain foreground")
        return self


class TileMask(BaseModel):
    encoding: Literal["tile-rle-row-v1"] = "tile-rle-row-v1"
    tile_size: Literal[128] = 128
    tiles: list[MaskTile] = Field(min_length=1, max_length=MAX_TILES)

    @model_validator(mode="after")
    def unique_tiles(self):
        if len({(t.x, t.y) for t in self.tiles}) != len(self.tiles):
            raise ValueError("Duplicate mask tile")
        if sum(len(t.counts) for t in self.tiles) > MAX_RUNS:
            raise ValueError("Mask exceeds the one-million-run editing limit")
        self.tiles.sort(key=lambda t: (t.y, t.x))
        return self


def mask_metrics(mask: TileMask) -> tuple[list[float], int]:
    left = top = math.inf
    right = bottom = area = 0
    for tile in mask.tiles:
        offset = 0
        for index, length in enumerate(tile.counts):
            if index % 2 and length:
                start_y, start_x = divmod(offset, TILE_SIZE)
                end_y, end_x = divmod(offset + length - 1, TILE_SIZE)
                left = min(left, tile.x + (start_x if start_y == end_y else 0))
                right = max(right, tile.x + (end_x + 1 if start_y == end_y else TILE_SIZE))
                top = min(top, tile.y + start_y)
                bottom = max(bottom, tile.y + end_y + 1)
                area += length
            offset += length
    return [left, top, right, bottom], area


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return cross(a, b, c) * cross(a, b, d) <= 0 and cross(c, d, a) * cross(c, d, b) <= 0


def contains(ring, point):
    inside = False
    x, y = point
    for i, (ax, ay) in enumerate(ring):
        bx, by = ring[i - 1]
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def polygon_metrics(polygons: list[list[list[float]]]):
    if not polygons or len(polygons) > 256 or sum(len(p) for p in polygons) > MAX_VERTICES:
        raise ValueError("An instance requires 1–256 contours and at most 2048 vertices")
    points, area, edges = [], 0.0, []
    for ring_index, ring in enumerate(polygons):
        if len(ring) < 3 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in ring):
            raise ValueError("A contour requires at least three finite x/y points")
        if len({tuple(p) for p in ring}) != len(ring):
            raise ValueError("Contour vertices must be distinct; the closing point is implicit")
        signed = sum(ring[i - 1][0] * p[1] - p[0] * ring[i - 1][1] for i, p in enumerate(ring))
        if abs(signed) < 1e-8:
            raise ValueError("Contour has zero area")
        area += abs(signed) / 2
        for i, a in enumerate(ring):
            b = ring[(i + 1) % len(ring)]
            # Adjacent collinear backtracking also produces invalid geometry.
            c = ring[(i + 2) % len(ring)]
            if cross(a, b, c) == 0 and ((a[0]-b[0])*(c[0]-b[0]) + (a[1]-b[1])*(c[1]-b[1])) > 0:
                raise ValueError("Contour edges cannot double back")
            edges.append((ring_index, i, a, b))
        points.extend(ring)
    for index, (r, i, a, b) in enumerate(edges):
        for s, j, c, d in edges[index + 1:]:
            if r == s and (abs(i - j) == 1 or {i, j} == {0, len(polygons[r]) - 1}):
                continue
            if intersects(a, b, c, d):
                raise ValueError("Contours cannot self-intersect, overlap or touch; use a mask for holes")
    for i, ring in enumerate(polygons):
        for other in polygons[i + 1:]:
            if contains(ring, other[0]) or contains(other, ring[0]):
                raise ValueError("Contours must be separate components; use the eraser to create holes")
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)], area
