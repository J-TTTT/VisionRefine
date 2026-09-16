from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from visionrefine.core.adapters import normalize_base_url, test_openai_compatible
from visionrefine.core.initial_detection import build_tiles, run_tiled_detection
from visionrefine.core.pilot import extract_jpeg_crop, run_detection_pilot
from visionrefine.core.router import plan_image
from visionrefine.core.store import ProjectStore


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
STATIC_ROOT = PACKAGE_ROOT / "static"
WORKSPACE_ROOT = REPO_ROOT / "workspace"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
TASKS = {
    "detection", "instance_segmentation", "grounding", "captioning",
    "vqa", "ocr", "classification",
}
DETECTION_LABELS = {
    "person", "vehicle", "car", "bus", "truck", "motorcycle", "bicycle",
    "traffic light", "stop sign", "fire hydrant", "bench", "backpack",
    "umbrella", "handbag", "suitcase", "dog",
}

# Large local datasets are an explicit product requirement. The server only
# reads dimensions during analysis; later pixel decoding is handled by bounded
# crops and pyramids rather than allocating an entire gigapixel image blindly.
Image.MAX_IMAGE_PIXELS = None

app = FastAPI(title="VisionRefine", version="0.1.0")
store = ProjectStore(WORKSPACE_ROOT / "projects")
initial_detection_jobs: dict[str, dict] = {}
initial_detection_lock = threading.Lock()


class ProjectInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    task: str
    dataset_path: str = Field(min_length=1)
    annotation_path: str | None = None
    model_max_side: int = Field(default=1536, ge=256, le=8192)
    labels: list[str] = Field(default_factory=list, max_length=100)


class AdapterInput(BaseModel):
    kind: str = "openai_compatible"
    base_url: str
    model: str
    api_key_env: str | None = None


class LabelsInput(BaseModel):
    labels: list[str] = Field(min_length=1, max_length=100)


class AnnotationInput(BaseModel):
    image: str
    objects: list[dict] = Field(default_factory=list, max_length=100_000)


class InitialDetectionInput(BaseModel):
    image: str


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": "0.1.0"}


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return store.list()


@app.post("/api/projects", status_code=201)
def create_project(payload: ProjectInput) -> dict:
    if payload.task not in TASKS:
        raise HTTPException(400, "Unsupported task")
    dataset = Path(payload.dataset_path).expanduser().resolve()
    if not dataset.is_dir():
        raise HTTPException(400, "Dataset directory does not exist")
    annotation = None
    if payload.annotation_path:
        annotation = Path(payload.annotation_path).expanduser().resolve()
        if not annotation.is_file():
            raise HTTPException(400, "Annotation file does not exist")
    data = payload.model_dump()
    labels = []
    for raw_label in payload.labels:
        label = raw_label.strip()
        if label and label not in labels:
            labels.append(label)
    if payload.task in {"detection", "instance_segmentation", "grounding", "classification"} and not labels:
        raise HTTPException(400, "Add at least one label for this task")
    if payload.task == "detection" and any(label not in DETECTION_LABELS for label in labels):
        raise HTTPException(400, "Detection labels must be selected from the supported label catalog")
    data["labels"] = labels
    data["dataset_path"] = str(dataset)
    data["annotation_path"] = str(annotation) if annotation else None
    return store.create(data)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    try:
        return store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict:
    try:
        destination = store.delete(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    return {
        "ok": True,
        "project_id": project_id,
        "recoverable_from": str(destination),
        "dataset_untouched": True,
    }


@app.put("/api/projects/{project_id}/labels")
def save_labels(project_id: str, payload: LabelsInput) -> dict:
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    if project.get("labels_locked"):
        raise HTTPException(409, "Label schema is locked after inference starts")
    labels = []
    for raw_label in payload.labels:
        label = raw_label.strip()
        if label and label not in labels:
            labels.append(label)
    if not labels:
        raise HTTPException(400, "At least one label is required")
    if project.get("task") == "detection" and any(label not in DETECTION_LABELS for label in labels):
        raise HTTPException(400, "Detection labels must be selected from the supported label catalog")
    project["labels"] = labels
    store.save(project)
    return project


@app.post("/api/projects/{project_id}/adapter/test")
def test_adapter(project_id: str, payload: AdapterInput) -> dict:
    try:
        store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    if payload.kind != "openai_compatible":
        raise HTTPException(400, "Unsupported adapter kind")
    try:
        result = test_openai_compatible(
            payload.base_url,
            api_key_env=payload.api_key_env or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return result.to_dict()


@app.put("/api/projects/{project_id}/adapter")
def save_adapter(project_id: str, payload: AdapterInput) -> dict:
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    if payload.kind != "openai_compatible":
        raise HTTPException(400, "Unsupported adapter kind")
    try:
        base_url = normalize_base_url(payload.base_url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    project["ai_adapter"] = {
        "kind": payload.kind,
        "base_url": base_url,
        "model": payload.model.strip(),
        "api_key_env": payload.api_key_env.strip() if payload.api_key_env else None,
    }
    store.save(project)
    return project


@app.post("/api/projects/{project_id}/pilot")
def run_pilot(project_id: str) -> dict:
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    if project.get("task") != "detection":
        raise HTTPException(400, "The first pilot currently supports detection projects only")
    if not project.get("ai_adapter"):
        raise HTTPException(400, "Configure an AI adapter first")
    if project.get("task") == "detection":
        project["labels_locked"] = True
        store.save(project)
    analysis = project.get("analysis") or {}
    images = analysis.get("images") or []
    if not images:
        raise HTTPException(400, "Analyze the dataset before running a pilot")

    root = Path(project["dataset_path"]).resolve()
    relative_path = images[0]["path"]
    image_path = (root / relative_path).resolve()
    if not image_path.is_relative_to(root) or not image_path.is_file():
        raise HTTPException(404, "Pilot image not found")
    try:
        result = run_detection_pilot(
            image_path,
            project["ai_adapter"],
            project["model_max_side"],
            project.get("labels") or ["person"],
        )
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        raise HTTPException(502, f"Pilot inference failed: {exc}") from None

    revision_id = datetime.now(timezone.utc).strftime("pilot-%Y%m%dT%H%M%S%fZ")
    revision_dir = store.root / project_id / "revisions" / revision_id
    revision_dir.mkdir(parents=True, exist_ok=False)
    crop_bytes = result.pop("crop_jpeg")
    (revision_dir / "crop.jpg").write_bytes(crop_bytes)
    suggestion = {
        "revision_id": revision_id,
        "type": "ai_suggestion",
        "task": "detection",
        "image": relative_path,
        "model": project["ai_adapter"]["model"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    (revision_dir / "suggestion.json").write_text(
        json.dumps(suggestion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    project["latest_suggestion"] = {
        "revision_id": revision_id,
        "image": relative_path,
        "object_count": len(suggestion["objects"]),
        "summary": suggestion["summary"],
    }
    project["status"] = "ai_suggested"
    store.save(project)
    return {**suggestion, "crop_url": f"/api/projects/{project_id}/revisions/{revision_id}/crop"}


@app.get("/api/projects/{project_id}/revisions/{revision_id}/crop")
def revision_crop(project_id: str, revision_id: str):
    try:
        store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    if not re.fullmatch(r"pilot-[0-9TZ]+", revision_id):
        raise HTTPException(404, "Revision not found")
    path = store.root / project_id / "revisions" / revision_id / "crop.jpg"
    if not path.is_file():
        raise HTTPException(404, "Revision crop not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/projects/{project_id}/suggestions/latest")
def latest_suggestion(project_id: str) -> dict:
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    latest = project.get("latest_suggestion")
    if not latest:
        raise HTTPException(404, "No AI suggestion exists")
    revision_id = latest["revision_id"]
    if not re.fullmatch(r"pilot-[0-9TZ]+", revision_id):
        raise HTTPException(404, "Revision not found")
    path = store.root / project_id / "revisions" / revision_id / "suggestion.json"
    if not path.is_file():
        raise HTTPException(404, "Suggestion data not found")
    suggestion = json.loads(path.read_text(encoding="utf-8"))
    suggestion["crop_url"] = f"/api/projects/{project_id}/revisions/{revision_id}/crop"
    return suggestion


@app.post("/api/projects/{project_id}/analyze")
def analyze_project(project_id: str) -> dict:
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None

    root = Path(project["dataset_path"])
    image_paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("._")
    )
    if not image_paths:
        raise HTTPException(400, "No supported images were found")

    has_coarse = bool(project.get("annotation_path"))
    routes: Counter[str] = Counter()
    valid = []
    errors = []
    total_pixels = 0
    for path in image_paths:
        try:
            with Image.open(path) as image:
                width, height = image.size
            route = plan_image(
                width, height,
                model_max_side=project["model_max_side"],
                has_coarse_annotations=has_coarse,
                task=project["task"],
            )
            routes[route.strategy] += 1
            total_pixels += width * height
            valid.append({
                "path": str(path.relative_to(root)),
                "width": width,
                "height": height,
                "megapixels": round(width * height / 1_000_000, 2),
                "route": route.to_dict(),
            })
        except (OSError, UnidentifiedImageError) as exc:
            errors.append({"path": str(path.relative_to(root)), "error": str(exc)})

    if not valid:
        raise HTTPException(400, "Images were found but none could be read")

    analysis = {
        "image_count": len(valid),
        "unreadable_count": len(errors),
        "has_coarse_annotations": has_coarse,
        "average_megapixels": round(total_pixels / len(valid) / 1_000_000, 2),
        "max_width": max(row["width"] for row in valid),
        "max_height": max(row["height"] for row in valid),
        "routes": dict(routes),
        "images": valid[:200],
        "truncated": len(valid) > 200,
        "errors": errors[:20],
    }
    project["analysis"] = analysis
    project["status"] = "routed"
    store.save(project)
    return project


def project_image(project: dict, relative_path: str) -> tuple[Path, Path]:
    root = Path(project["dataset_path"]).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Image not found")
    return root, path


def annotation_path(project_id: str, relative_path: str) -> Path:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return store.root / project_id / "annotations" / f"{digest}.json"


def suggestion_path(project_id: str, relative_path: str) -> Path:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return store.root / project_id / "suggestions" / f"{digest}.json"


def _update_initial_job(job_id: str, **values) -> None:
    with initial_detection_lock:
        if job_id in initial_detection_jobs:
            initial_detection_jobs[job_id].update(values)


def _run_initial_detection_job(job_id: str, project_id: str, relative_path: str) -> None:
    try:
        project = store.get(project_id)
        _, image_path = project_image(project, relative_path)
        revision_id = datetime.now(timezone.utc).strftime("initial-%Y%m%dT%H%M%S%fZ")

        def progress(completed: int, total: int, object_count: int) -> None:
            _update_initial_job(
                job_id, status="running", completed_tiles=completed,
                total_tiles=total, candidate_count=object_count,
            )

        result = run_tiled_detection(
            image_path,
            project["ai_adapter"],
            project["model_max_side"],
            project.get("labels") or ["person"],
            overlap=0.2,
            nms_threshold=0.5,
            workers=4,
            progress=progress,
        )
        revision_dir = store.root / project_id / "revisions" / revision_id
        revision_dir.mkdir(parents=True, exist_ok=False)
        suggestion = {
            "revision_id": revision_id,
            "type": "ai_suggestion",
            "task": "detection",
            "image": relative_path,
            "model": project["ai_adapter"]["model"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ai_suggestion",
            **result,
        }
        (revision_dir / "suggestion.json").write_text(
            json.dumps(suggestion, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        current_path = suggestion_path(project_id, relative_path)
        current_path.parent.mkdir(parents=True, exist_ok=True)
        temp = current_path.with_suffix(".tmp")
        temp.write_text(json.dumps(suggestion, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(current_path)
        latest_project = store.get(project_id)
        latest_project["status"] = "ai_suggested"
        store.save(latest_project)
        _update_initial_job(
            job_id, status="completed", completed_tiles=result["tile_count"],
            total_tiles=result["tile_count"], candidate_count=len(result["objects"]),
            revision_id=revision_id, error_count=len(result["errors"]),
        )
    except Exception as exc:
        _update_initial_job(job_id, status="failed", error=str(exc)[:2000])


@app.post("/api/projects/{project_id}/initial-detection", status_code=202)
def start_initial_detection(project_id: str, payload: InitialDetectionInput) -> dict:
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    if project.get("task") != "detection":
        raise HTTPException(400, "Initial detection currently supports detection projects only")
    if not project.get("ai_adapter"):
        raise HTTPException(400, "Configure an AI adapter first")
    if not project.get("labels_locked"):
        project["labels_locked"] = True
        store.save(project)
    _, image_path = project_image(project, payload.image)
    with Image.open(image_path) as image:
        width, height = image.size
    total_tiles = len(build_tiles(width, height, project["model_max_side"], 0.2))
    with initial_detection_lock:
        running = next((job for job in initial_detection_jobs.values()
                        if job["project_id"] == project_id and job["status"] in {"queued", "running"}), None)
        if running:
            raise HTTPException(409, f"Initial detection job {running['job_id']} is already running")
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "project_id": project_id,
            "image": payload.image,
            "status": "queued",
            "completed_tiles": 0,
            "total_tiles": total_tiles,
            "candidate_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        initial_detection_jobs[job_id] = job
    threading.Thread(
        target=_run_initial_detection_job,
        args=(job_id, project_id, payload.image),
        name=f"visionrefine-{job_id[:8]}",
        daemon=True,
    ).start()
    return job


@app.get("/api/projects/{project_id}/initial-detection/{job_id}")
def initial_detection_status(project_id: str, job_id: str) -> dict:
    try:
        store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    with initial_detection_lock:
        job = initial_detection_jobs.get(job_id)
        if not job or job["project_id"] != project_id:
            raise HTTPException(404, "Initial detection job not found")
        return dict(job)


@app.get("/api/projects/{project_id}/images")
def project_images(project_id: str) -> list[dict]:
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    return (project.get("analysis") or {}).get("images") or []


@app.get("/api/projects/{project_id}/thumbnail/{relative_path:path}")
def image_thumbnail(project_id: str, relative_path: str):
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    _, path = project_image(project, relative_path)
    cache_dir = WORKSPACE_ROOT / "cache" / project_id / "thumbnails"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    cached = cache_dir / f"{key}.jpg"
    if not cached.exists():
        with Image.open(path) as image:
            original = image.size
            image.draft("RGB", (min(1800, original[0]), min(1200, original[1])))
            image = image.convert("RGB")
            image.thumbnail((1800, 1200))
            image.save(cached, "JPEG", quality=88)
    return FileResponse(cached, media_type="image/jpeg")


@app.get("/api/projects/{project_id}/crop/{relative_path:path}")
def image_crop(project_id: str, relative_path: str, x: int, y: int, width: int, height: int):
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    _, path = project_image(project, relative_path)
    with Image.open(path) as image:
        image_width, image_height = image.size
    width = max(64, min(4096, width, image_width))
    height = max(64, min(4096, height, image_height))
    x = max(0, min(image_width - width, x))
    y = max(0, min(image_height - height, y))
    x = (x // 16) * 16
    y = (y // 16) * 16
    try:
        data = extract_jpeg_crop(path, (x, y, width, height))
    except (OSError, subprocess.SubprocessError) as exc:
        raise HTTPException(400, f"Crop failed: {exc}") from None
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/projects/{project_id}/annotations")
def get_annotations(project_id: str, image: str) -> dict:
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    project_image(project, image)
    path = annotation_path(project_id, image)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    current_suggestion = suggestion_path(project_id, image)
    if current_suggestion.is_file():
        suggestion = json.loads(current_suggestion.read_text(encoding="utf-8"))
        return {
            "image": image,
            "objects": suggestion.get("objects", []),
            "status": "ai_suggestion",
            "revision_id": suggestion.get("revision_id"),
        }
    latest = project.get("latest_suggestion") or {}
    if latest.get("image") == image:
        revision_id = latest["revision_id"]
        latest_path = store.root / project_id / "revisions" / revision_id / "suggestion.json"
        if latest_path.is_file():
            suggestion = json.loads(latest_path.read_text(encoding="utf-8"))
            return {"image": image, "objects": suggestion.get("objects", []), "status": "ai_suggestion", "revision_id": revision_id}
    return {"image": image, "objects": [], "status": "unreviewed", "revision_id": None}


@app.put("/api/projects/{project_id}/annotations")
def save_annotations(project_id: str, payload: AnnotationInput) -> dict:
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    _, image_path = project_image(project, payload.image)
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    allowed = set(project.get("labels") or ["person"])
    objects = []
    for index, item in enumerate(payload.objects):
        label = str(item.get("label", "")).strip()
        bbox = item.get("bbox")
        if label not in allowed or not isinstance(bbox, list) or len(bbox) != 4:
            raise HTTPException(400, f"Invalid object at index {index}")
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            raise HTTPException(400, f"Invalid bbox at index {index}") from None
        x1, x2 = sorted((max(0, min(image_width, x1)), max(0, min(image_width, x2))))
        y1, y2 = sorted((max(0, min(image_height, y1)), max(0, min(image_height, y2))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            raise HTTPException(400, f"Empty bbox at index {index}")
        objects.append({
            "id": str(item.get("id") or f"human-{index}"),
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "confidence": item.get("confidence"),
            "source": "human_reviewed",
        })
    revision_id = datetime.now(timezone.utc).strftime("human-%Y%m%dT%H%M%S%fZ")
    document = {
        "image": payload.image,
        "objects": objects,
        "status": "human_reviewed",
        "revision_id": revision_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = annotation_path(project_id, payload.image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history = store.root / project_id / "revisions" / revision_id
        history.mkdir(parents=True)
        (history / "previous.json").write_bytes(path.read_bytes())
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    project["status"] = "human_reviewed"
    store.save(project)
    return document


@app.get("/api/projects/{project_id}/preview/{relative_path:path}")
def preview_image(project_id: str, relative_path: str):
    try:
        project = store.get(project_id)
    except KeyError:
        raise HTTPException(404, "Project not found") from None
    root = Path(project["dataset_path"]).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


app.mount("/assets", StaticFiles(directory=STATIC_ROOT), name="assets")


@app.get("/")
def index():
    return FileResponse(STATIC_ROOT / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VisionRefine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
