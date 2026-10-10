"""Durable local video annotations, background imports and frame datasets."""
from __future__ import annotations

import bisect
import copy
import hashlib
import gzip
import json
import math
import re
import shutil
import threading
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import media
from .models import Annotations, LabelsInput, Label, ProjectInput, SamplingInput, ExtractInput, SegmentationDocument
from . import segmentation as seg
from ..dataset_io.models import Dataset, Category, ImageRecord, Annotation
from ..dataset_io.service import load_dataset, persist_import

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video")
_JOB_GUARD = threading.RLock()
_ACTIVE: dict[tuple, threading.Event] = {}
_TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mpeg", ".mpg", ".mts", ".m2ts"}


class VideoConflict(ValueError):
    pass


class Cancelled(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", value):
        raise KeyError(value)
    return value


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def read_json(path, default=None):
    if not path.is_file():
        return copy.deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


def job_key(store, pid, jid):
    return (str(store.root.resolve()), pid, jid)


def ensure_no_active_jobs(store, pid):
    with _JOB_GUARD:
        if any(key[:2] == (str(store.root.resolve()), pid) for key in _ACTIVE):
            raise VideoConflict("Video processing is active. Cancel its jobs before deleting the project.")


class VideoService:
    def __init__(self, store):
        self.store = store

    def project(self, pid):
        project = self.store.get(identifier(pid))
        if project.get("task") != "video":
            raise ValueError("This operation requires a video project")
        return project

    def folder(self, pid):
        self.project(pid)
        result = (self.store.root / pid / "video").resolve()
        if not result.is_relative_to(self.store.root.resolve()):
            raise ValueError("Video metadata path escapes workspace")
        return result

    def manifest(self, pid):
        return read_json(self.folder(pid) / "manifest.json", {"videos": []})

    def videos(self, pid):
        return self.manifest(pid)["videos"]

    def video(self, pid, vid):
        identifier(vid)
        video = next((v for v in self.videos(pid) if v["id"] == vid), None)
        if video is None:
            raise KeyError(vid)
        return video

    def index(self, pid, vid):
        video = self.video(pid, vid)
        return read_json(self.folder(pid) / "media" / vid / "index.json", video)

    def create(self, payload: ProjectInput):
        paths = self.resolve_sources(payload.paths) if payload.paths else []
        labels = []
        for raw in payload.labels:
            name = raw.strip()
            if name and name not in [label["name"] for label in labels]:
                labels.append(Label(id=f"label-{uuid.uuid4().hex[:12]}", name=name).model_dump())
        project = self.store.create(dict(name=payload.name, task="video", dataset_path="", labels=[],
                                         dataset_format="visionrefine_video", split=payload.split))
        project.update(video_schema="1.0", video_labels=labels, labels=[label["name"] for label in labels],
                       dataset_path=str(self.store.root / project["id"] / "video"), video_count=0)
        self.store.save(project)
        atomic_json(self.folder(project["id"]) / "manifest.json", {"videos": []})
        job = self.start_job(project["id"], "import", {"paths": paths}) if paths else None
        return {"project": project, "job": job}

    def labels(self, pid):
        return self.project(pid).get("video_labels", [])

    def save_labels(self, pid, payload: LabelsInput):
        with self.store.locked(pid):
            project = self.project(pid)
            labels = [label.model_dump() for label in payload.labels]
            ids = {label["id"] for label in labels}
            for video in self.videos(pid):
                if any(event.get("label_id") and event["label_id"] not in ids
                       for event in self.annotations(pid, video["id"])["events"]):
                    raise VideoConflict("A label is still used by events. Reassign those events before deleting it.")
            project.update(video_labels=labels, labels=[label["name"] for label in labels])
            self.store.save(project)
            return {"labels": labels}

    def _state(self, pid, vid):
        return read_json(self.folder(pid) / "state" / f"{identifier(vid)}.json")

    def annotations(self, pid, vid):
        with self.store.locked(pid):
            self.video(pid, vid)
            state = self._state(pid, vid)
            if state:
                return state["annotations"]
            return read_json(self.folder(pid) / "annotations" / f"{vid}.json", Annotations().model_dump())

    def validate_annotations(self, pid, vid, payload, labels=None, tracks=None):
        document = Annotations.model_validate(payload)
        duration = self.video(pid, vid)["duration"]
        allowed = {label["id"] for label in (self.labels(pid) if labels is None else labels)}
        track_ids = {track["id"] for track in (self.segmentation(pid, vid)["tracks"] if tracks is None else tracks)}
        for event in document.events:
            if len(event.track_ids) != len(set(event.track_ids)) or any(ref not in track_ids for ref in event.track_ids):
                raise ValueError("Event object references must be unique existing track IDs")
            if event.label_id is not None and event.label_id not in allowed:
                raise ValueError(f"Unknown event label: {event.label_id}")
        for item in [*document.events, *document.captions]:
            if any(value is not None and value > duration + 0.000001 for value in (item.start, item.end)):
                raise ValueError(f"Annotation {item.id} extends beyond the video duration")
        for task in ("events", "captions"):
            if getattr(document.review, task) == "reviewed" and any(not item.reviewed for item in getattr(document, task)):
                raise ValueError(f"Confirm every {task} annotation before marking the entire task reviewed")
        return document

    def save_annotations(self, pid, vid, payload):
        with self.store.locked(pid):
            document = self.validate_annotations(pid, vid, payload)
            current = self.annotations(pid, vid)
            if current["revision"] != document.revision:
                raise VideoConflict("Annotations changed since you opened them. Reload before saving.")
            document.revision += 1
            document.updated_at = now()
            result = document.model_dump()
            folder = self.folder(pid)
            # Unique revision files retain every edit even if a crash happens between these writes.
            snapshot = folder / "revisions" / vid / f"{document.revision:08d}-{uuid.uuid4().hex}.json"
            atomic_json(snapshot, result)
            state = self._state(pid, vid)
            if state:
                atomic_json(folder / "state" / f"{vid}.json", {**state, "annotations": result})
            else:
                atomic_json(folder / "annotations" / f"{vid}.json", result)
            return result

    def object_labels(self, pid):
        return self.project(pid).get("video_object_labels", [])

    def save_object_labels(self, pid, payload):
        with self.store.locked(pid):
            project = self.project(pid)
            labels = payload.model_dump()["labels"]
            allowed = {label["id"] for label in labels}
            for video in self.videos(pid):
                if any(track["label_id"] not in allowed for track in self.segmentation(pid, video["id"])["tracks"]):
                    raise VideoConflict("An object label is still used by a track; reassign that track before deleting the label")
            project["video_object_labels"] = labels
            self.store.save(project)
            return {"labels": labels}

    def segmentation(self, pid, vid):
        with self.store.locked(pid):
            self.video(pid, vid)
            state = self._state(pid, vid)
            filename = state.get("segmentation") if state else None
            if not filename:
                return SegmentationDocument().model_dump()
            if not re.fullmatch(r"[0-9]{8,}-[0-9a-f]{32}\.json", filename):
                raise ValueError("Invalid segmentation snapshot reference")
            result = read_json(self.folder(pid) / "segmentation_revisions" / vid / filename)
            if result is None:
                raise ValueError("The segmentation snapshot is unavailable")
            for track in result["tracks"]:
                for key in track["keyframes"]:
                    geometry = key.get("geometry")
                    if geometry and geometry.get("mask_ref"):
                        digest = geometry.pop("mask_ref")
                        if not re.fullmatch(r"[0-9a-f]{64}", digest):
                            raise ValueError("Invalid mask asset reference")
                        path = self.folder(pid) / "mask_assets" / f"{digest}.json.gz"
                        with gzip.open(path, "rb") as handle:
                            encoded = handle.read(seg.MAX_DOCUMENT_BYTES + 1)
                        if len(encoded) > seg.MAX_DOCUMENT_BYTES or hashlib.sha256(encoded).hexdigest() != digest:
                            raise ValueError("Mask asset integrity check failed")
                        geometry["mask"] = json.loads(encoded)
            return result

    def _publish_segmentation(self, pid, vid, document, annotations=None):
        # Both document heads are published with one atomic pointer update.
        document.revision += 1
        document.updated_at = now()
        result = document.model_dump()
        persisted = copy.deepcopy(result)
        for track in persisted["tracks"]:
            for key in track["keyframes"]:
                geometry = key.get("geometry")
                if geometry and geometry.get("mask"):
                    encoded = json.dumps(geometry.pop("mask"), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                    digest = hashlib.sha256(encoded).hexdigest()
                    path = self.folder(pid) / "mask_assets" / f"{digest}.json.gz"
                    if not path.exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temp = path.with_name(f".{digest}-{uuid.uuid4().hex}.tmp")
                        try:
                            temp.write_bytes(gzip.compress(encoded, mtime=0))
                            temp.replace(path)
                        finally:
                            temp.unlink(missing_ok=True)
                    geometry["mask_ref"] = digest
        filename = f"{document.revision:08d}-{uuid.uuid4().hex}.json"
        atomic_json(self.folder(pid) / "segmentation_revisions" / vid / filename, persisted)
        if annotations is None:
            annotations = self.annotations(pid, vid)
        else:
            annotations = copy.deepcopy(annotations)
            annotations["revision"] += 1
            annotations["updated_at"] = now()
            atomic_json(self.folder(pid) / "revisions" / vid / f"{annotations['revision']:08d}-{uuid.uuid4().hex}.json", annotations)
        atomic_json(self.folder(pid) / "state" / f"{vid}.json", {"annotations": annotations, "segmentation": filename})
        return result, annotations

    def save_segmentation(self, pid, vid, payload):
        with self.store.locked(pid):
            previous = self.segmentation(pid, vid)
            old_keys = {(t["id"], k["frame_index"]): k for t in previous["tracks"] for k in t["keyframes"]}
            payload = copy.deepcopy(payload)
            for track in payload.get("tracks", []):
                for key in track.get("keyframes", []):
                    old = old_keys.get((track["id"], key["frame_index"]))
                    if old and key.get("provenance") and key.get("provenance") == old.get("provenance") and any(key.get(field) != old.get(field) for field in ("geometry", "visibility")):
                        # A user correction becomes a protected human anchor on the next run.
                        key.pop("provenance", None)
                        key["reviewed"] = False
            document = seg.validate_document(payload, self.video(pid, vid), self.object_labels(pid))
            if document.revision != self.segmentation(pid, vid)["revision"]:
                raise VideoConflict("Segmentation changed since you opened it. Reload before saving.")
            allowed = {track.id for track in document.tracks}
            if any(ref not in allowed for event in self.annotations(pid, vid)["events"] for ref in event.get("track_ids", [])):
                raise VideoConflict("An event references a removed object. Update its object links before deleting the track.")
            return self._publish_segmentation(pid, vid, document)[0]

    def segmentation_operation(self, pid, vid, payload):
        with self.store.locked(pid):
            current = self.segmentation(pid, vid)
            annotations = self.annotations(pid, vid)
            if payload.revision != current["revision"] or (payload.annotations_revision is not None and payload.annotations_revision != annotations["revision"]):
                raise VideoConflict("Video annotations changed. Reload before editing object tracks.")
            result, events, changed_events = seg.operate(current, annotations, payload, self.index(pid, vid))
            document = seg.validate_document(result, self.video(pid, vid), self.object_labels(pid))
            self.validate_annotations(pid, vid, events, tracks=result["tracks"])
            saved, events = self._publish_segmentation(pid, vid, document, events if changed_events else None)
            return {"segmentation": saved, "annotations": events}

    def segmentation_preview(self, pid, vid, payload):
        try:
            from ..segmentation_edit import EditOptions, preview_edit
        except ImportError:
            raise ValueError("Install the segmentation dependencies: pip install -e '.[segmentation]'") from None
        video = self.video(pid, vid)
        shape = seg.geometry_annotation(payload.geometry)
        options = EditOptions(image="video-frame.jpg", objects=[shape.model_dump()], selected_ids=[shape.id],
                              operation=payload.operation, radius=payload.radius, distance=payload.snap_distance,
                              preserve_holes=payload.protect_holes)
        response = preview_edit(options, self.frame_path(pid, vid, payload.frame_index),
                                (video["width"], video["height"]), [shape.label])
        updated = Annotation.model_validate(response["objects"][0])
        return {"geometry": seg.geometry_values(updated), "report": response["report"]}

    def resolve_sources(self, values):
        result = []
        for value in values:
            path = Path(value).expanduser().resolve()
            if path.is_dir():
                found = sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink() and p.suffix.lower() in _VIDEO_SUFFIXES)
                if not found:
                    raise ValueError(f"No supported videos in directory: {path}")
            elif path.is_file():
                found = [path]
            else:
                raise ValueError(f"Video source does not exist: {path}")
            for item in found:
                absolute = str(item.resolve())
                if absolute not in result:
                    result.append(absolute)
            if len(result) > 1000:
                raise ValueError("Import at most 1000 videos at a time")
        return result

    def job_path(self, pid, jid):
        return self.folder(pid) / "jobs" / f"{identifier(jid)}.json"

    def jobs(self, pid):
        folder = self.folder(pid) / "jobs"
        result = []
        with self.store.locked(pid), _JOB_GUARD:
            for path in folder.glob("*.json"):
                job = read_json(path)
                if job["status"] in {"queued", "running", "cancelling"} and job_key(self.store, pid, job["id"]) not in _ACTIVE:
                    job.update(status="interrupted", error="Processing stopped when the service exited. Retry to resume safely.", updated_at=now())
                    atomic_json(path, job)
                result.append(job)
        return sorted(result, key=lambda row: row["created_at"], reverse=True)

    def job(self, pid, jid):
        identifier(jid)
        return next((j for j in self.jobs(pid) if j["id"] == jid), None)

    def start_job(self, pid, kind, payload):
        self.project(pid)
        job = dict(id=f"job-{uuid.uuid4().hex}", kind=kind, status="queued", payload=payload,
                   progress={"current": 0, "total": 1, "message": "Queued"}, result=None, error=None,
                   created_at=now(), updated_at=now())
        flag = threading.Event()
        key = job_key(self.store, pid, job["id"])
        with self.store.locked(pid), _JOB_GUARD:
            _ACTIVE[key] = flag
            atomic_json(self.job_path(pid, job["id"]), job)
            _EXECUTOR.submit(self._run, pid, job, flag)
        return copy.deepcopy(job)

    def cancel_job(self, pid, jid):
        with self.store.locked(pid), _JOB_GUARD:
            job = self.job(pid, jid)
            if job is None:
                raise KeyError(jid)
            flag = _ACTIVE.get(job_key(self.store, pid, jid))
            if flag:
                flag.set()
                job.update(status="cancelling", updated_at=now())
                atomic_json(self.job_path(pid, jid), job)
            return job

    def retry_job(self, pid, jid):
        old = self.job(pid, jid)
        if old is None:
            raise KeyError(jid)
        if old["status"] not in {"failed", "cancelled", "interrupted"}:
            raise VideoConflict("Only failed, interrupted or cancelled jobs can be retried")
        return self.start_job(pid, old["kind"], old["payload"])

    def _run(self, pid, job, flag):
        def checkpoint():
            if flag.is_set():
                raise Cancelled()
        def progress(value):
            checkpoint()
            with self.store.locked(pid):
                job.update(progress=value, updated_at=now())
                atomic_json(self.job_path(pid, job["id"]), job)
        try:
            checkpoint()
            job["status"] = "running"
            progress({"current": 0, "total": 1, "message": "Starting"})
            if job["kind"] == "import":
                result = self._import(pid, job["payload"]["paths"], checkpoint, progress, flag)
            elif job["kind"] == "extract":
                result = self._extract(pid, job["payload"], checkpoint, progress)
            else:
                raise ValueError("Unknown video job type")
            # A committed operation has finished even if cancellation arrives immediately afterward.
            job.update(status="completed", result=result, error=None)
        except Exception as exc:
            job.update(status="cancelled" if flag.is_set() or isinstance(exc, Cancelled) else "failed",
                       error=None if flag.is_set() or isinstance(exc, Cancelled) else str(exc))
        finally:
            job["updated_at"] = now()
            try:
                with self.store.locked(pid):
                    atomic_json(self.job_path(pid, job["id"]), job)
            finally:
                with _JOB_GUARD:
                    _ACTIVE.pop(job_key(self.store, pid, job["id"]), None)

    def _import(self, pid, paths, checkpoint, progress, flag):
        imported, skipped = [], []
        for position, raw in enumerate(paths):
            checkpoint()
            source = Path(raw)
            before = source.stat()
            if before.st_size > media.MAX_SOURCE_BYTES:
                raise ValueError("Video exceeds the 8 GB import limit; split it into shorter clips")
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checkpoint()
                    digest.update(chunk)
            sha256 = digest.hexdigest()
            vid = f"video-{sha256[:24]}"
            cache = self.folder(pid) / "media" / vid
            with self.store.locked(f"video-import-{pid}-{vid}"):
                checkpoint()
                with self.store.locked(pid):
                    existing = next((v for v in self.videos(pid) if v["sha256"] == sha256), None)
                    if existing is not None:
                        old_source = Path(existing["source_path"])
                        try:
                            old_stat = old_source.stat()
                            unchanged = (old_stat.st_size, old_stat.st_mtime_ns) == (existing["source_size"], existing["source_mtime_ns"])
                        except OSError:
                            unchanged = False
                        cache_complete = (existing.get("cache_version") == media.CACHE_VERSION
                            and (cache / "index.json").is_file()
                            and all((cache / name).is_file() for name in ("preview.mp4", "preview.webm", "poster.jpg")))
                        if unchanged and cache_complete:
                            skipped.append(str(source))
                            continue
                progress({"current": position, "total": len(paths), "message": f"Indexing {source.name}"})
                cache = self.folder(pid) / "media" / vid
                metadata = media.prepare_video(source, cache, progress=lambda detail: progress({
                    "current": position, "total": len(paths), "message": f"{source.name}: {detail.get('message', 'Processing')}",
                    "media": detail}), cancel=flag.is_set)
                checkpoint()
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Video source changed while importing: {source.name}")
                if metadata.get("sha256") != sha256:
                    raise ValueError(f"Video source changed while importing: {source.name}")
                timestamps = metadata["timestamps"]
                record = {key: value for key, value in metadata.items() if key not in {"timestamps", "frames", "source_pts"}}
                record.update(id=vid, name=source.name, source_path=str(source), sha256=sha256,
                              source_size=after.st_size, source_mtime_ns=after.st_mtime_ns,
                              split=self.project(pid).get("split", "unspecified"), group_id=sha256,
                              imported_at=now(), frame_count=len(timestamps))
                atomic_json(cache / "index.json", {**metadata, **record, "timestamps": timestamps})
                with self.store.locked(pid):
                    manifest = self.manifest(pid)
                    previous = next((v for v in manifest["videos"] if v["sha256"] == sha256), None)
                    if previous:
                        # Same bytes may be re-imported to restore a moved source. Keep identity and annotation links.
                        record.update(split=previous["split"], group_id=previous["group_id"], imported_at=previous["imported_at"])
                        manifest["videos"] = [record if v["id"] == vid else v for v in manifest["videos"]]
                    else:
                        manifest["videos"].append(record)
                    atomic_json(self.folder(pid) / "manifest.json", manifest)
                    project = self.project(pid)
                    project["video_count"] = len(manifest["videos"])
                    self.store.save(project)
                    imported.append(vid)
                progress({"current": position + 1, "total": len(paths), "message": f"Imported {source.name}"})
        return {"video_ids": imported, "skipped": skipped}

    def source(self, pid, vid):
        video = self.video(pid, vid)
        source = Path(video["source_path"])
        if not source.is_file():
            raise ValueError("Original video is unavailable; restore it at its imported location")
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (video["source_size"], video["source_mtime_ns"]):
            raise VideoConflict("Original video changed after import. Import the changed file as a new resource.")
        return source

    def preview_path(self, pid, vid, format="mp4"):
        if format not in {"mp4", "webm"}:
            raise ValueError("Choose mp4 or webm preview format")
        video = self.video(pid, vid)
        cache = (self.folder(pid) / "media" / vid).resolve()
        value = video.get("preview_path" if format == "mp4" else "preview_webm_path")
        if not value:
            raise ValueError("This preview format is unavailable; re-import the source video to regenerate previews")
        path = Path(value).resolve()
        if not path.is_relative_to(cache) or not path.is_file():
            raise ValueError("Video preview is unavailable; retry the import")
        return path

    def frame_path(self, pid, vid, frame_index):
        metadata = self.index(pid, vid)
        if not 0 <= frame_index < len(metadata["timestamps"]):
            raise ValueError("Frame index is outside this video")
        source = self.source(pid, vid)
        with self.store.locked(f"frame-{pid}-{vid}-{frame_index}"):
            return media.frame_image(source, self.folder(pid) / "media" / vid, metadata, frame_index)

    def sample(self, pid, vid, payload: SamplingInput):
        index = self.index(pid, vid)
        timestamps = index["timestamps"]
        start = 0 if payload.start is None else payload.start
        end = index["duration"] if payload.end is None else payload.end
        if start > end or end > index["duration"] + 0.000001:
            raise ValueError("Sampling range is outside the video")
        lower = bisect.bisect_left(timestamps, start)
        upper = bisect.bisect_right(timestamps, end) - 1
        if payload.mode != "current" and lower > upper:
            return {"frames": [], "count": 0}
        def nearest(value):
            pos = bisect.bisect_left(timestamps, value)
            return min((max(lower, min(upper, pos - 1)), max(lower, min(upper, pos))),
                       key=lambda i: abs(timestamps[i] - value))
        if payload.mode == "current":
            if payload.frame_index is None or payload.frame_index >= len(timestamps):
                raise ValueError("Choose a valid current frame")
            frames = [payload.frame_index]
        elif payload.mode == "events":
            boundaries = [value for event in self.annotations(pid, vid)["events"]
                          for value in (event["start"], event.get("end")) if value is not None and start <= value <= end]
            # Include immediate neighbours of each event boundary for context.
            frames = [offset for value in boundaries for offset in (nearest(value) - 1, nearest(value), nearest(value) + 1)
                      if 0 <= offset < len(timestamps)]
        elif payload.mode == "interval":
            if payload.interval is None:
                raise ValueError("Choose a positive sampling interval")
            steps = (end - start) / payload.interval
            if not math.isfinite(steps) or steps >= 1000:
                raise ValueError("Sample at most 1000 frames per request; increase the interval")
            count = int(math.floor(steps)) + 1
            frames = [nearest(start + offset * payload.interval) for offset in range(count)]
        else:
            if payload.count is None:
                raise ValueError("Choose the number of frames")
            frames = [nearest(start if payload.count == 1 else start + (end - start) * offset / (payload.count - 1))
                      for offset in range(payload.count)]
        frames = sorted(set(frames))
        if len(frames) > 1000:
            raise ValueError("Sample at most 1000 frames per request")
        return {"frames": [{"frame_index": frame, "timestamp": timestamps[frame]} for frame in frames], "count": len(frames)}

    def extract(self, pid, vid, payload: ExtractInput):
        index = self.index(pid, vid)
        if any(frame >= len(index["timestamps"]) for frame in payload.frame_indices):
            raise ValueError("A selected frame is outside this video")
        with self.store.locked(pid):
            values = {"video_id": vid, **payload.model_dump()}
            if payload.include_segmentation:
                document = self.segmentation(pid, vid)
                labels = {label["id"]: label["name"] for label in self.object_labels(pid)}
                frames = {}
                for track in document["tracks"]:
                    for key in track["keyframes"]:
                        if key["frame_index"] in payload.frame_indices and key["visibility"] == "visible":
                            frames.setdefault(str(key["frame_index"]), []).append({"track_id": track["id"],
                                "label": labels[track["label_id"]], "geometry": key["geometry"], "source_reviewed": key["reviewed"]})
                snapshot = {"revision": document["revision"], "frames": frames}
                filename = f"snapshot-{uuid.uuid4().hex}.json"
                atomic_json(self.folder(pid) / "extraction_snapshots" / filename, snapshot)
                values["segmentation_snapshot"] = filename
                values["labels"] = list(dict.fromkeys([*values["labels"], *(shape["label"] for shapes in frames.values() for shape in shapes)]))
            return self.start_job(pid, "extract", values)

    def _extract(self, pid, payload, checkpoint, progress):
        vid = payload["video_id"]
        index = self.index(pid, vid)
        task = payload["task"]
        snapshot = None
        if payload.get("include_segmentation"):
            filename = payload.get("segmentation_snapshot", "")
            if not re.fullmatch(r"snapshot-[0-9a-f]{32}\.json", filename):
                raise ValueError("Invalid extraction segmentation snapshot")
            snapshot = read_json(self.folder(pid) / "extraction_snapshots" / filename)
            if snapshot is None:
                raise ValueError("The extraction segmentation snapshot is unavailable")
        # Serialise child dataset updates, without blocking annotation saves or cancellation.
        with self.store.locked(f"video-extract-{pid}-{task}"):
            with self.store.locked(pid):
                parent = self.project(pid)
                child_id = parent.get("video_derived_projects", {}).get(task)
                try:
                    child = self.store.get(child_id) if child_id else None
                except KeyError:
                    child = None
                if child is None:
                    child = self.store.create(dict(name=f"{parent['name']} · extracted {task}", task=task,
                                                   dataset_path="", labels=payload["labels"], split=parent.get("split", "unspecified")))
                    child["dataset_path"] = str(self.store.root / child["id"] / "images")
                    child["video_parent_project"] = pid
                    child["video_source"] = {"project_id": pid, "video_id": vid}
                    initial = Dataset(task=task,
                        categories=[Category(id=i, name=name) for i, name in enumerate(payload["labels"])],
                        provenance={"format": "visionrefine", "video_project_id": pid})
                    persist_import(self.store, child, initial, hash_images=False)
                    parent.setdefault("video_derived_projects", {})[task] = child["id"]
                    self.store.save(parent)
            image_root = Path(child["dataset_path"])
            image_root.mkdir(parents=True, exist_ok=True)
            with self.store.locked(child["id"]):
                child = self.store.get(child["id"])
                dataset = load_dataset(self.store, child) or Dataset(task=task, provenance={"format": "visionrefine", "video_project_id": pid})
                category_names = [category.name for category in dataset.categories]
                for name in payload["labels"]:
                    if name not in category_names:
                        dataset.categories.append(Category(id=max((c.id for c in dataset.categories), default=-1) + 1, name=name))
                        category_names.append(name)
                existing = {record.path for record in dataset.images}
                selected = sorted(set(payload["frame_indices"]))
                added, skipped = [], []
                with tempfile.TemporaryDirectory(prefix=".video-extract-", dir=self.store.root / child["id"]) as staging:
                    for position, frame in enumerate(selected):
                        checkpoint()
                        name = f"{vid}-frame-{frame:08d}.jpg"
                        if name in existing:
                            skipped.append(frame)
                            continue
                        source = self.frame_path(pid, vid, frame)
                        checkpoint()
                        target = Path(staging) / name
                        pending = target.with_suffix(".jpg.part")
                        shutil.copyfile(source, pending)
                        pending.replace(target)
                        digest = hashlib.sha256(target.read_bytes()).hexdigest()
                        objects = []
                        frame_provenance = {"project_id": pid, "video_id": vid, "frame_index": frame,
                            "timestamp": index["timestamps"][frame], "source_sha256": index["sha256"], "group_id": index["sha256"]}
                        if snapshot is not None:
                            frame_provenance.update(segmentation_revision=snapshot["revision"], sparse_segmentation=True)
                            categories = {category.name: category.id for category in dataset.categories}
                            for shape in snapshot["frames"].get(str(frame), []):
                                annotation = seg.geometry_annotation(shape["geometry"], object_id=shape["track_id"],
                                    label=shape["label"], category_id=categories[shape["label"]])
                                annotation.provenance["video"] = {**frame_provenance, "track_id": shape["track_id"],
                                    "source_reviewed": shape["source_reviewed"]}
                                objects.append(annotation)
                        dataset.images.append(ImageRecord(id=f"{vid}-{frame}", path=name, width=index["width"], height=index["height"],
                            split=index.get("split", parent.get("split", "unspecified")), sha256=digest,
                            objects=objects, status="imported_coarse" if objects else "unreviewed",
                            provenance={"video": frame_provenance}))
                        existing.add(name)
                        added.append(frame)
                        progress({"current": position + 1, "total": len(selected), "message": f"Extracted frame {frame}"})
                    checkpoint()
                    dataset.report.image_count = len(dataset.images)
                    dataset.report.empty_images = sum(not record.objects for record in dataset.images)
                    dataset.report.object_count = sum(len(record.objects) for record in dataset.images)
                    dataset.provenance.update(format="visionrefine", video_project_id=pid)
                    for staged in Path(staging).glob("*.jpg"):
                        staged.replace(image_root / staged.name)
                    persist_import(self.store, child, dataset, hash_images=False)
            return {"project_id": child["id"], "added": added, "skipped": skipped, "image_count": len(dataset.images)}

    def export(self, pid, policy="all"):
        if policy not in {"all", "reviewed"}:
            raise ValueError("Unknown export policy")
        with self.store.locked(pid):
            project = self.project(pid)
            videos = []
            has_segmentation = bool(self.object_labels(pid))
            for video in self.videos(pid):
                annotations = self.annotations(pid, video["id"])
                segmentation = self.segmentation(pid, video["id"])
                has_segmentation |= segmentation["revision"] > 0
                if policy == "reviewed":
                    annotations["events"] = [e for e in annotations["events"] if e["reviewed"]]
                    annotations["captions"] = [c for c in annotations["captions"] if c["reviewed"]]
                    # Keep identity/lifetime records for event links. Sparse unreviewed geometry is omitted.
                    for track in segmentation["tracks"]:
                        track["keyframes"] = [key for key in track["keyframes"] if key["reviewed"]]
                        track["visibility_ranges"] = [state for state in track["visibility_ranges"] if state["reviewed"]]
                videos.append({key: video[key] for key in ("id", "name", "sha256", "duration", "width", "height", "frame_count", "split", "group_id")})
                videos[-1].update(timestamps=self.index(pid, video["id"])["timestamps"],
                                  annotations=annotations, segmentation=segmentation)
            result = {"schema": "visionrefine-video", "schema_version": "2.0" if has_segmentation else "1.0",
                      "created_at": now(), "policy": policy,
                      "project": {"name": project["name"], "split": project.get("split", "unspecified")},
                      "labels": self.labels(pid), "videos": videos, "media_included": False}
            has_ai = any(item.get("provenance") for v in videos for item in
                         [*v["annotations"]["events"], *v["annotations"]["captions"],
                          *(k for t in v["segmentation"]["tracks"] for k in t["keyframes"])])
            if has_ai:
                result["schema_version"] = "3.0"
                has_segmentation = True
            if has_segmentation:
                result["object_labels"] = self.object_labels(pid)
            else:
                for video in videos:
                    video.pop("segmentation")
            return result

    @staticmethod
    def _merge_label_catalog(current, incoming):
        labels, mapping = copy.deepcopy(current), {}
        for label in incoming:
            by_id = next((old for old in labels if old["id"] == label["id"]), None)
            by_name = next((old for old in labels if old["name"] == label["name"]), None)
            if by_id and by_id["name"] != label["name"]:
                raise VideoConflict("An imported label ID already has a different name")
            chosen = by_id or by_name
            if chosen:
                mapping[label["id"]] = chosen["id"]
            else:
                labels.append(label)
                mapping[label["id"]] = label["id"]
        # Each catalog may be valid independently while their union exceeds the editable limit.
        labels = LabelsInput.model_validate({"labels": labels}).model_dump()["labels"]
        return labels, mapping

    def import_annotations(self, pid, bundle):
        expected_revisions = bundle.get("expected_revisions") if "bundle" in bundle else None
        expected_segmentation = bundle.get("expected_segmentation_revisions") if "bundle" in bundle else None
        bundle = bundle.get("bundle", bundle)
        if not isinstance(bundle, dict):
            raise ValueError("Expected a video annotation bundle")
        for mapping in (expected_revisions, expected_segmentation):
            if mapping is not None and not isinstance(mapping, dict):
                raise ValueError("Expected revisions must map video IDs to revision numbers")
        schema = bundle.get("schema_version")
        if bundle.get("schema") != "visionrefine-video" or schema not in {"1.0", "2.0", "3.0"}:
            raise ValueError("Unsupported video annotation format")
        incoming_labels = LabelsInput.model_validate({"labels": bundle.get("labels", [])}).model_dump()["labels"]
        incoming_objects = LabelsInput.model_validate({"labels": bundle.get("object_labels", [])}).model_dump()["labels"]
        incoming = bundle.get("videos")
        if not isinstance(incoming, list) or len(incoming) > 1000:
            raise ValueError("Expected a video list with at most 1000 entries")
        with self.store.locked(pid):
            labels, mapping = self._merge_label_catalog(self.labels(pid), incoming_labels)
            objects, object_mapping = self._merge_label_catalog(self.object_labels(pid), incoming_objects)
            local = {video["sha256"]: video for video in self.videos(pid)}
            staged, seen = [], set()
            for item in incoming:
                if not isinstance(item, dict) or item.get("sha256") not in local:
                    raise ValueError("Import the matching source video first (SHA-256 fingerprint must match)")
                video = local[item["sha256"]]
                vid = video["id"]
                if vid in seen:
                    raise ValueError("The bundle contains a duplicate video")
                seen.add(vid)
                if (item.get("width"), item.get("height"), item.get("frame_count")) != (video["width"], video["height"], video["frame_count"]):
                    raise ValueError("Imported video geometry or frame count does not match")
                if not isinstance(item.get("duration"), (int, float)) or not math.isfinite(item["duration"]) or abs(item["duration"] - video["duration"]) > 0.000001:
                    raise ValueError("Imported video duration does not match")
                timestamps = item.get("timestamps")
                if timestamps is not None and timestamps != self.index(pid, vid)["timestamps"]:
                    raise ValueError("Imported presentation timestamps do not match")
                seg_document = None
                tracks = self.segmentation(pid, vid)["tracks"]
                if schema in {"2.0", "3.0"}:
                    if not isinstance(item.get("segmentation"), dict):
                        raise ValueError("Version 2 videos require a segmentation document")
                    seg_payload = SegmentationDocument.model_validate(item["segmentation"]).model_dump()
                    revision = self.segmentation(pid, vid)["revision"]
                    expected = expected_segmentation.get(vid) if expected_segmentation is not None else seg_payload["revision"]
                    if (expected_segmentation is not None or revision > 0) and expected != revision:
                        raise VideoConflict("Segmentation changed since this import snapshot. Reload before importing.")
                    seg_payload.update(revision=revision, reviewed_ranges=[])
                    for track in seg_payload["tracks"]:
                        if track["label_id"] not in object_mapping:
                            raise ValueError("An imported object references an undeclared object label")
                        track.update(label_id=object_mapping[track["label_id"]], reviewed=False)
                        for state in [*track["keyframes"], *track["visibility_ranges"]]:
                            state["reviewed"] = False
                    seg_document = seg.validate_document(seg_payload, video, objects)
                    tracks = seg_document.model_dump()["tracks"]
                if not isinstance(item.get("annotations"), dict):
                    raise ValueError("Each imported video needs an annotations document")
                payload = Annotations.model_validate(item["annotations"]).model_dump()
                current_revision = self.annotations(pid, vid)["revision"]
                expected = expected_revisions.get(vid) if expected_revisions is not None else payload["revision"]
                if (expected_revisions is not None or current_revision > 0) and expected != current_revision:
                    raise VideoConflict("Annotations changed since this import snapshot. Reload before importing.")
                payload["revision"] = current_revision
                payload["review"] = {"events": "unreviewed", "captions": "unreviewed"}
                for event in payload.get("events", []):
                    if event.get("label_id") is not None:
                        if event["label_id"] not in mapping:
                            raise ValueError("An imported event references an undeclared label")
                        event["label_id"] = mapping[event["label_id"]]
                for annotation in [*payload.get("events", []), *payload.get("captions", [])]:
                    annotation["reviewed"] = False
                self.validate_annotations(pid, vid, payload, labels=labels, tracks=tracks)
                staged.append((vid, payload, seg_document))
            # No project or annotation file changes until every incoming video has validated.
            project = self.project(pid)
            project.update(video_labels=labels, labels=[label["name"] for label in labels])
            if schema in {"2.0", "3.0"}:
                project["video_object_labels"] = objects
            self.store.save(project)
            for vid, payload, seg_document in staged:
                if seg_document is not None:
                    self._publish_segmentation(pid, vid, seg_document, payload)
                else:
                    self.save_annotations(pid, vid, payload)
            return {"imported": len(staged), "video_ids": [vid for vid, _, _ in staged], "review_reset": True}
