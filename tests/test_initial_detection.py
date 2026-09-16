from visionrefine.core.initial_detection import build_tiles, class_aware_nms


def test_tiles_cover_edges_and_use_aligned_origins():
    tiles = build_tiles(26753, 15052, 1536, 0.2)
    assert tiles
    assert all(tile.x % 16 == 0 and tile.y % 16 == 0 for tile in tiles)
    assert min(tile.x for tile in tiles) == 0
    assert min(tile.y for tile in tiles) == 0
    assert max(tile.x + tile.width for tile in tiles) == 26753
    assert max(tile.y + tile.height for tile in tiles) == 15052


def test_nms_is_class_aware_and_prefers_confidence():
    objects = [
        {"label": "person", "bbox": [0, 0, 100, 100], "confidence": 0.9},
        {"label": "person", "bbox": [5, 5, 105, 105], "confidence": 0.7},
        {"label": "car", "bbox": [5, 5, 105, 105], "confidence": 0.8},
    ]
    kept = class_aware_nms(objects, 0.5)
    assert [(item["label"], item["confidence"]) for item in kept] == [
        ("person", 0.9), ("car", 0.8),
    ]
    assert [item["id"] for item in kept] == ["ai-initial-0", "ai-initial-1"]
