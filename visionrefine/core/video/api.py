"""Video HTTP routes. The store getter supports app tests and alternate workspaces."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from . import media
from .models import Annotations, ExtractInput, ImportInput, LabelsInput, ProjectInput, SamplingInput, SegmentationDocument, TrackOperation, GeometryPreview
from .service import VideoConflict, VideoService


@contextmanager
def errors():
    try:
        yield
    except VideoConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except KeyError:
        raise HTTPException(404, "Video project, resource or job not found") from None
    except (ValueError, OSError, ValidationError) as exc:
        raise HTTPException(400, str(exc)) from None


def create_video_router(get_store):
    router = APIRouter(prefix="/api/video", tags=["video"])

    def service():
        return VideoService(get_store())

    @router.get("/capabilities")
    def capabilities():
        return media.capabilities()

    @router.get("/projects")
    def projects():
        return [project for project in get_store().list() if project.get("task") == "video"]

    @router.post("/projects", status_code=201)
    def create_project(payload: ProjectInput):
        with errors():
            return service().create(payload)

    @router.get("/projects/{pid}")
    def project(pid: str):
        with errors():
            video = service()
            with video.store.locked(pid):
                return {"project": video.project(pid), "videos": video.videos(pid),
                        "labels": video.labels(pid), "jobs": video.jobs(pid)}

    @router.post("/projects/{pid}/import", status_code=202)
    def import_video(pid: str, payload: ImportInput):
        with errors():
            video = service()
            video.project(pid)
            return video.start_job(pid, "import", {"paths": video.resolve_sources(payload.paths)})

    @router.get("/projects/{pid}/jobs")
    def jobs(pid: str):
        with errors():
            return service().jobs(pid)

    @router.post("/projects/{pid}/jobs/{jid}/cancel")
    def cancel_job(pid: str, jid: str):
        with errors():
            return service().cancel_job(pid, jid)

    @router.post("/projects/{pid}/jobs/{jid}/retry", status_code=202)
    def retry_job(pid: str, jid: str):
        with errors():
            return service().retry_job(pid, jid)

    @router.get("/projects/{pid}/labels")
    def labels(pid: str):
        with errors():
            return {"labels": service().labels(pid)}

    @router.put("/projects/{pid}/labels")
    def save_labels(pid: str, payload: LabelsInput):
        with errors():
            return service().save_labels(pid, payload)

    @router.get("/projects/{pid}/object-labels")
    def object_labels(pid: str):
        with errors():
            return {"labels": service().object_labels(pid)}

    @router.put("/projects/{pid}/object-labels")
    def save_object_labels(pid: str, payload: LabelsInput):
        with errors():
            return service().save_object_labels(pid, payload)

    @router.get("/projects/{pid}/videos/{vid}/segmentation")
    def segmentation(pid: str, vid: str):
        with errors():
            return service().segmentation(pid, vid)

    @router.put("/projects/{pid}/videos/{vid}/segmentation")
    def save_segmentation(pid: str, vid: str, payload: SegmentationDocument):
        with errors():
            return service().save_segmentation(pid, vid, payload.model_dump())

    @router.post("/projects/{pid}/videos/{vid}/segmentation/operations")
    def segmentation_operation(pid: str, vid: str, payload: TrackOperation):
        with errors():
            return service().segmentation_operation(pid, vid, payload)

    @router.post("/projects/{pid}/videos/{vid}/segmentation/preview")
    def segmentation_preview(pid: str, vid: str, payload: GeometryPreview):
        with errors():
            return service().segmentation_preview(pid, vid, payload)

    @router.get("/projects/{pid}/videos/{vid}/annotations")
    def annotations(pid: str, vid: str):
        with errors():
            return service().annotations(pid, vid)

    @router.put("/projects/{pid}/videos/{vid}/annotations")
    def save_annotations(pid: str, vid: str, payload: Annotations):
        with errors():
            return service().save_annotations(pid, vid, payload.model_dump())

    @router.get("/projects/{pid}/videos/{vid}/index")
    def index(pid: str, vid: str):
        with errors():
            data = service().index(pid, vid)
            return {key: data[key] for key in ("timestamps", "frame_count", "duration", "width", "height")}

    @router.get("/projects/{pid}/videos/{vid}/media")
    def video_media(pid: str, vid: str, format: Literal["mp4", "webm"] = "mp4"):
        with errors():
            return FileResponse(service().preview_path(pid, vid, format), media_type=f"video/{format}")

    @router.get("/projects/{pid}/videos/{vid}/frames/{frame_index}")
    def frame(pid: str, vid: str, frame_index: int):
        with errors():
            return FileResponse(service().frame_path(pid, vid, frame_index), media_type="image/jpeg")

    @router.post("/projects/{pid}/videos/{vid}/extract/preview")
    def sample(pid: str, vid: str, payload: SamplingInput):
        with errors():
            return service().sample(pid, vid, payload)

    @router.post("/projects/{pid}/videos/{vid}/extract", status_code=202)
    def extract(pid: str, vid: str, payload: ExtractInput):
        with errors():
            return service().extract(pid, vid, payload)

    @router.get("/projects/{pid}/export")
    def export(pid: str, policy: str = "all"):
        with errors():
            bundle = service().export(pid, policy)
            return JSONResponse(bundle, headers={"Content-Disposition": f'attachment; filename="{pid}-video.json"'})

    @router.post("/projects/{pid}/annotations/import")
    def import_annotations(pid: str, payload: dict):
        with errors():
            return service().import_annotations(pid, payload)

    return router
