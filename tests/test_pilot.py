from visionrefine.core.pilot import center_crop_box, parse_json_object


def test_center_crop_is_mcu_aligned():
    x, y, width, height = center_crop_box(26753, 15052, 1536)
    assert x % 16 == 0
    assert y % 16 == 0
    assert (width, height) == (1536, 1536)


def test_parse_fenced_model_json():
    result = parse_json_object('```json\n{"objects": [], "summary": "none"}\n```')
    assert result["objects"] == []


def test_parse_model_json_keeps_label_payload():
    result = parse_json_object('{"objects":[{"label":"vehicle","bbox":[1,2,3,4]}]}')
    assert result["objects"][0]["label"] == "vehicle"
