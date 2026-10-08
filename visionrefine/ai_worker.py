"""Optional, single-GPU model worker. Install with scripts/setup_ai.sh.

The web app never imports torch. Models load lazily and are evicted when switching
families, so a 24 GB card can serve all default capabilities serially.
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageDraw

from .core.ai.providers import Cancelled, instruction, parse_json
from .core.video.service import atomic_json, identifier, read_json

ROOT = Path(os.environ.get("VISIONREFINE_AI_CACHE", Path(__file__).resolve().parent.parent / "workspace/cache/ai")).resolve()
os.environ.setdefault("HF_HOME", str(ROOT / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
MODELS = {
    "google/owlv2-base-patch16-ensemble": ["image_detection"],
    "facebook/sam2.1-hiera-tiny": ["image_segmentation", "video_segmentation"],
    "Qwen/Qwen3-VL-2B-Instruct": ["video_caption", "video_events"],
}
app = FastAPI(title="VisionRefine local model worker")
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-gpu")
_LOCK = threading.RLock()
_FLAGS = {}
_LOADED = (None, None)


@app.middleware("http")
async def bounds_and_auth(request: Request, call_next):
    from fastapi.responses import JSONResponse
    import hmac
    token = os.environ.get("VISIONREFINE_AI_TOKEN")
    if token and not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    if request.method == "POST":
        body = bytearray()
        async for block in request.stream():
            body.extend(block)
            if len(body) > 128 * 1024 * 1024:
                return JSONResponse({"detail": "Input exceeds 128 MB"}, status_code=413)
        request._body = bytes(body)
    return await call_next(request)


def model_path(model):
    paths = read_json(ROOT / "model_paths.json", {})
    return paths.get(model, model)


@app.get("/capabilities")
def capabilities():
    from importlib.util import find_spec
    rows = []
    for model, caps in MODELS.items():
        dependencies = ("torch", "sam2") if "sam2" in model else ("torch", "transformers", "scipy") if "owlv2" in model else ("torch", "transformers")
        deps = all(find_spec(name) for name in dependencies)
        local = Path(model_path(model))
        ready = deps and local.is_dir() and (any(local.glob("*.safetensors")) or any(local.glob("*.bin")) or any(local.glob("*.pt")))
        rows.append({"id": model, "capabilities": caps, "ready": ready,
                     "detail": "本地权重已就绪，首次使用会加载模型" if ready else "依赖或本地权重尚未准备，请运行 setup_ai.sh"})
    return {"protocol_version": "1.0", "models": rows, "visual_only": True, "max_frames": 1000}


def path(jid):
    return ROOT / "jobs" / identifier(jid) / "job.json"


@app.get("/jobs/{jid}")
def get_job(jid: str):
    try:
        with _LOCK:
            job = read_json(path(jid))
            if not job:
                raise HTTPException(404, "Job not found")
            if job["status"] in {"queued", "running", "cancelling"} and jid not in _FLAGS:
                job.update(status="interrupted", error="Worker restarted")
                atomic_json(path(jid), job)
            return job
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from None


@app.post("/jobs/{jid}/cancel")
def cancel_job(jid: str):
    with _LOCK:
        job = get_job(jid)
        if jid in _FLAGS:
            _FLAGS[jid].set()
            job["status"] = "cancelling"
            atomic_json(path(jid), job)
        return {"id": jid, "status": job["status"]}


@app.post("/jobs", status_code=202)
def create_job(payload: dict):
    model, cap = payload.get("model"), payload.get("capability")
    if cap not in MODELS.get(model, []):
        raise HTTPException(400, "Unsupported model or capability")
    frames = payload.get("frames")
    width, height = payload.get("width"), payload.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 1 or width * height > 16_777_216:
        raise HTTPException(400, "Invalid original image size; maximum 16 megapixels")
    limit = 1000 if cap == "video_segmentation" else 32 if cap.startswith("video_") else 1
    if not isinstance(frames, list) or not 1 <= len(frames) <= limit:
        raise HTTPException(400, "Invalid frame count")
    with _LOCK:
        if len(_FLAGS) >= 4:
            raise HTTPException(429, "Worker queue is full")
        jid = "worker-" + uuid.uuid4().hex
        directory = path(jid).parent / "frames"
        directory.mkdir(parents=True)
        try:
            indices = []
            for i, frame in enumerate(frames):
                index = frame["frame_index"]
                if not isinstance(index, int) or index < 0 or index in indices:
                    raise ValueError("Invalid frame index")
                indices.append(index)
                raw = base64.b64decode(frame["jpeg"], validate=True)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError("Frame too large")
                with Image.open(io.BytesIO(raw)) as img:
                    if img.width * img.height > 4_194_304:
                        raise ValueError("Input frame too large")
                    img.convert("RGB").save(directory / f"{i:06d}.jpg", quality=95)
            if cap == "video_segmentation" and indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError("Video propagation requires every consecutive frame")
            metadata = {**payload, "frames": [{k: v for k, v in f.items() if k != "jpeg"} for f in frames]}
            atomic_json(path(jid).with_name("input.json"), metadata)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            shutil.rmtree(path(jid).parent)
            raise HTTPException(400, str(exc)) from None
        job = {"id": jid, "status": "queued", "progress": {"message": "等待 GPU", "current": 0, "total": 1}, "error": None}
        atomic_json(path(jid), job)
        flag = _FLAGS[jid] = threading.Event()
        _POOL.submit(run_job, jid, metadata, flag)
        return job


def load_model(model, video=False):
    global _LOADED
    key = (model, video)
    if _LOADED[0] == key:
        return _LOADED[1]
    import torch
    _LOADED = (None, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    location = model_path(model)
    if not Path(location).is_dir():
        raise ValueError("本地权重尚未安装，请先运行 scripts/setup_ai.sh")
    if "sam2" in model:
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        weights = next(Path(location).glob("*.pt"))
        builder = build_sam2_video_predictor if video else build_sam2
        sam = builder("configs/sam2.1/sam2.1_hiera_t.yaml", str(weights), device=device, apply_postprocessing=False)
        if video:
            sam.add_all_frames_to_correct_as_cond = True
            value = sam
        else:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            value = SAM2ImagePredictor(sam)
    elif "owlv2" in model:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        value = (Owlv2Processor.from_pretrained(location, local_files_only=True),
                 Owlv2ForObjectDetection.from_pretrained(location, local_files_only=True).to(device).eval())
    else:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        value = (AutoProcessor.from_pretrained(location, local_files_only=True, min_pixels=128 * 32 * 32, max_pixels=256 * 32 * 32),
                 Qwen3VLForConditionalGeneration.from_pretrained(location, local_files_only=True, dtype=dtype,
                                                                  attn_implementation="sdpa").to(device).eval())
    _LOADED = key, value
    return value


def geometry_mask(geometry, width, height):
    import numpy as np
    from .core.dataset_io.models import Annotation
    obj = Annotation.model_validate({**geometry, "id": "seed", "label": "object", "category_id": 0})
    if obj.bbox[0] < 0 or obj.bbox[1] < 0 or obj.bbox[2] > width or obj.bbox[3] > height:
        raise ValueError("Seed geometry outside image")
    image = Image.new("L", (width, height))
    if obj.kind == "polygon":
        draw = ImageDraw.Draw(image)
        for polygon in obj.polygons:
            draw.polygon([tuple(p) for p in polygon], fill=1)
    elif obj.kind == "mask":
        for tile in obj.mask.tiles:
            values = np.repeat(np.arange(len(tile.counts)) % 2, tile.counts).astype("uint8").reshape(128, 128)
            image.paste(Image.fromarray(values), (tile.x, tile.y))
    else:
        raise ValueError("A contour or mask is required")
    return image


def output_geometry(mask, width, height):
    import numpy as np
    from .core.segmentation_edit import _encode_sparse
    image = Image.fromarray(np.asarray(mask, dtype="uint8")).resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(image)
    if not array.any():
        return None
    return {"kind": "mask", "mask": _encode_sparse(array, 0, 0, (width, height)), "polygons": None}


def run_inference(jid, payload, checkpoint, progress):
    import numpy as np
    import torch
    from contextlib import nullcontext
    cap = payload["capability"]
    progress("加载模型，首次使用需要一些时间", 0, 1)
    model = load_model(payload["model"], video=cap == "video_segmentation")
    checkpoint()
    folder = path(jid).parent / "frames"
    width, height = payload["width"], payload["height"]
    context = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() and "sam2" in payload["model"] else nullcontext()
    with torch.inference_mode(), context:
        if cap == "video_segmentation":
            state = model.init_state(str(folder), offload_video_to_cpu=True, offload_state_to_cpu=True)
            checkpoint()
            offset = payload["frames"][0]["frame_index"]
            first = Image.open(folder / "000000.jpg")
            shape = first.size
            first.close()
            for seed in payload["seeds"]:
                checkpoint()
                mask = np.asarray(geometry_mask(seed["geometry"], width, height).resize(shape, Image.Resampling.NEAREST)).astype(bool)
                model.add_new_mask(state, seed["frame_index"] - offset, 1, mask)
            anchor = payload["seed_frame"] - offset
            rows = {}
            total = len(payload["frames"])
            for reverse in (False, True):
                for index, ids, logits in model.propagate_in_video(state, start_frame_idx=anchor, reverse=reverse):
                    checkpoint()
                    mask = (logits[0, 0] > 0).cpu().numpy()
                    geometry = output_geometry(mask, width, height)
                    rows[index] = {"frame_index": index + offset, "visibility": "visible" if geometry else "occluded", "geometry": geometry}
                    progress("传播可见轮廓", len(rows), total)
            del state
            return {"keyframes": [rows[k] for k in sorted(rows)]}
        images = [Image.open(folder / f"{i:06d}.jpg").convert("RGB") for i in range(len(payload["frames"]))]
        if cap == "image_segmentation":
            image = images[0]
            model.set_image(np.array(image))
            prompt = payload["prompt"]
            sx, sy = image.width / width, image.height / height
            points = np.asarray(prompt["points"], dtype=np.float32) * [sx, sy] if prompt["points"] else None
            labels = np.asarray(prompt["point_labels"], dtype=np.int32) if points is not None else None
            box = np.asarray(prompt["box"]) * [sx, sy, sx, sy] if prompt["box"] else None
            seed = None
            if prompt.get("geometry"):
                seed = np.asarray(geometry_mask(prompt["geometry"], width, height).resize((256, 256), Image.Resampling.NEAREST), dtype=np.float32)
                seed = (seed * 20 - 10)[None, :, :]
            masks, scores, _ = model.predict(point_coords=points, point_labels=labels, box=box, mask_input=seed, multimask_output=False)
            geometry = output_geometry(masks[int(np.argmax(scores))], width, height)
            checkpoint()
            return {"objects": [{**geometry, "label": payload["label"], "confidence": float(np.clip(scores.max(), 0, 1))}] if geometry else []}
        processor, network = model
        if cap == "image_detection":
            labels = payload["labels"]
            if not labels or len(labels) > 200:
                raise ValueError("默认检测模型单次支持 1–200 个类别")
            inputs = processor(text=[labels], images=images[0], return_tensors="pt").to(network.device)
            outputs = network(**inputs)
            detections = processor.post_process_object_detection(outputs, threshold=payload["threshold"], target_sizes=torch.tensor([[height, width]], device=network.device))[0]
            rows = []
            for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                box = [max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2)]
                if box[2] > box[0] and box[3] > box[1]:
                    rows.append({"label": labels[int(label)], "confidence": float(score), "bbox": box})
            # Remove duplicate proposals of the same class.
            from torchvision.ops import batched_nms
            if rows:
                keep = batched_nms(torch.tensor([r["bbox"] for r in rows]), torch.tensor([r["confidence"] for r in rows]),
                                   torch.tensor([labels.index(r["label"]) for r in rows]), 0.5)[:500]
                rows = [rows[int(i)] for i in keep]
            checkpoint()
            return {"objects": rows}
        content = [{"type": "text", "text": instruction(payload)}, {"type": "video"}]
        text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        # The processor derives timestamps as index/fps. Encode original PTS as
        # microsecond ticks and disable resampling, preserving variable frame timing.
        from transformers.video_utils import VideoMetadata
        metadata = VideoMetadata(total_num_frames=len(images), fps=1_000_000,
                                 frames_indices=[round(frame["timestamp"] * 1_000_000) for frame in payload["frames"]])
        inputs = processor(text=[text], videos=[np.stack([np.asarray(img) for img in images])],
                           video_metadata=[metadata], do_sample_frames=False,
                           size={"shortest_edge": 128 * 32 * 32, "longest_edge": 256 * 32 * 32},
                           padding=True, return_tensors="pt").to(network.device)
        from transformers import StoppingCriteria, StoppingCriteriaList
        class StopOnCancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                checkpoint()
                return False
        progress("根据采样画面生成建议", 0, 1)
        output = network.generate(**inputs, max_new_tokens=2048, do_sample=False,
                                  stopping_criteria=StoppingCriteriaList([StopOnCancel()]))
        result = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        checkpoint()
        return parse_json(result)


def run_job(jid, payload, flag):
    def checkpoint():
        if flag.is_set():
            raise Cancelled()

    def progress(message, current=0, total=1):
        checkpoint()
        with _LOCK:
            job = get_job(jid)
            job.update(status="running", progress={"message": message, "current": current, "total": total})
            atomic_json(path(jid), job)
    try:
        checkpoint()
        result = run_inference(jid, payload, checkpoint, progress)
        checkpoint()
        result["model_revision"] = read_json(ROOT / "model_revisions.json", {}).get(payload["model"])
        if len(json.dumps(result)) > 48 * 1024 * 1024:
            raise ValueError("模型输出过大，请缩短范围")
        with _LOCK:
            job = get_job(jid)
            job.update(status="completed", result=result, progress={"message": "完成", "current": 1, "total": 1})
            atomic_json(path(jid), job)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        with _LOCK:
            job = get_job(jid)
            job.update(status="cancelled" if flag.is_set() else "failed", error=None if flag.is_set() else str(exc)[:1000])
            atomic_json(path(jid), job)
    finally:
        shutil.rmtree(path(jid).parent / "frames", ignore_errors=True)
        with _LOCK:
            _FLAGS.pop(jid, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("VISIONREFINE_AI_TOKEN"):
        parser.error("Set VISIONREFINE_AI_TOKEN before exposing the worker to other machines")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
