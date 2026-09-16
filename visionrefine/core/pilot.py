from __future__ import annotations

import base64
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from visionrefine.core.adapters import create_chat_completion


def center_crop_box(width: int, height: int, size: int) -> tuple[int, int, int, int]:
    crop_width = min(width, size)
    crop_height = min(height, size)
    # JPEG lossless crops are most predictable on common 16-pixel MCU boundaries.
    x = max(0, ((width - crop_width) // 2 // 16) * 16)
    y = max(0, ((height - crop_height) // 2 // 16) * 16)
    return x, y, crop_width, crop_height


def extract_jpeg_crop(path: Path, box: tuple[int, int, int, int]) -> bytes:
    x, y, width, height = box
    jpegtran = shutil.which("jpegtran")
    if jpegtran and path.suffix.lower() in {".jpg", ".jpeg"}:
        result = subprocess.run(
            [jpegtran, "-copy", "none", "-crop", f"{width}x{height}+{x}+{y}", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout
    with Image.open(path) as image:
        crop = image.crop((x, y, x + width, y + height)).convert("RGB")
        output = io.BytesIO()
        crop.save(output, "JPEG", quality=92)
        return output.getvalue()


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


def run_detection_pilot(
    image_path: Path,
    adapter: dict,
    model_max_side: int,
    labels: list[str],
) -> dict:
    with Image.open(image_path) as image:
        original_width, original_height = image.size
    box = center_crop_box(original_width, original_height, model_max_side)
    result = run_detection_crop(image_path, adapter, box, labels, id_prefix="ai-pilot")
    result["original_size"] = {"width": original_width, "height": original_height}
    result["requested_crop"] = {"width": box[2], "height": box[3]}
    return result


def run_detection_crop(
    image_path: Path,
    adapter: dict,
    box: tuple[int, int, int, int],
    labels: list[str],
    *,
    id_prefix: str = "ai",
) -> dict:
    crop_bytes = extract_jpeg_crop(image_path, box)
    with Image.open(io.BytesIO(crop_bytes)) as crop_image:
        crop_width, crop_height = crop_image.size
    data_uri = "data:image/jpeg;base64," + base64.b64encode(crop_bytes).decode("ascii")
    allowed_labels = [label.strip() for label in labels if label.strip()]
    label_text = "、".join(allowed_labels)
    example_label = allowed_labels[0]
    prompt = (
        f"你是视觉标注助手。检测图中所有可见的目标，允许的标签只有：{label_text}。只输出 JSON，不要解释。"
        f"格式为 {{\"objects\":[{{\"label\":\"{example_label}\",\"bbox\":[x1,y1,x2,y2],"
        "\"confidence\":0.0}],\"summary\":\"中文简述\"}。"
        "bbox 使用当前图像内 0 到 1000 的归一化坐标；忽略不属于允许标签的对象、雕像和反射。"
    )
    response = create_chat_completion(adapter, [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": prompt},
        ],
    }])
    raw_text = response["choices"][0]["message"]["content"]
    parsed = parse_json_object(raw_text)
    offset_x, offset_y, requested_width, requested_height = box
    objects = []
    for index, item in enumerate(parsed.get("objects", [])):
        coords = item.get("bbox")
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        x1, y1, x2, y2 = [max(0.0, min(1000.0, float(value))) for value in coords]
        if x2 <= x1 or y2 <= y1:
            continue
        label = str(item.get("label", example_label)).strip()
        if label not in allowed_labels:
            continue
        objects.append({
            "id": f"{id_prefix}-{index}",
            "label": label,
            "bbox": [
                offset_x + x1 / 1000 * crop_width,
                offset_y + y1 / 1000 * crop_height,
                offset_x + x2 / 1000 * crop_width,
                offset_y + y2 / 1000 * crop_height,
            ],
            "confidence": float(item.get("confidence", 0.5)),
            "source": "ai_suggestion",
        })
    return {
        "crop": {"x": offset_x, "y": offset_y, "width": crop_width, "height": crop_height},
        "objects": objects,
        "summary": str(parsed.get("summary", "")),
        "raw_response": raw_text,
        "usage": response.get("usage"),
        "crop_jpeg": crop_bytes,
    }
