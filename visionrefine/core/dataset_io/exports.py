"""Durable local export plans/jobs. No worker can mutate project annotations."""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from zipfile import ZipFile, ZIP_STORED

from pydantic import BaseModel, Field
from PIL import Image

from . import registry
from .common import dump_json
from .compatibility import issue
from .models import Dataset, Split
from .portable import file_digest, redact
from .service import image_location, package_snapshot, select_snapshot


class ExportOptions(BaseModel):
    formats: list[str] = Field(min_length=1, max_length=20)
    policy: Literal["reviewed", "reviewed_or_ai", "all_annotated"] = "reviewed"
    splits: list[Split] = Field(default_factory=list)
    include_images: bool = False


class ExportCommit(BaseModel):
    preview_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    acknowledgements: dict[str, list[str]] = Field(default_factory=dict)


class ExportPreset(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    options: ExportOptions


class ExportCancelled(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    dump_json(temp, value)
    temp.replace(path)


class ExportCenter:
    """One service process per workspace. Interrupted work is explicitly retryable."""
    def __init__(self, store):
        self.store = store
        self._events = {}
        self._slots = threading.Semaphore(2)
        for path in store.root.glob("*/export-jobs/*.json"):
            job = json.loads(path.read_text())
            if job["status"] in {"queued", "running"}:
                job["status"] = "interrupted"
                job["error"] = "服务重启中断了导出；可重试同一快照。"
                for result in job["formats"].values():
                    if result["status"] in {"queued", "running"}:
                        result["status"] = "interrupted"
                atomic_json(path, job)

    def _path(self, project_id, kind, key):
        self.store.get(project_id)
        if not re.fullmatch(r"[0-9a-f]{32}", key):
            raise ValueError("Invalid export plan/job ID")
        return self.store.root / project_id / kind / f"{key}.json"

    def preview(self, project_id, options: ExportOptions):
        with self.store.locked(project_id):
            project = self.store.get(project_id)
            selected, selection = select_snapshot(self.store, project, options.policy, options.splits)
            errors = []
            for image in selected.images:
                try:
                    _, path = image_location(project, image, selected)
                    checksum = file_digest(path)
                    if image.sha256 and image.sha256 != checksum:
                        raise ValueError("图片内容与导入时不同")
                    image.sha256 = checksum
                    with Image.open(path) as pixels:
                        if pixels.size != (image.width, image.height):
                            raise ValueError("图片尺寸与标注快照不同")
                        image.provenance["image_encoding"] = pixels.format
                        image.provenance["exif_orientation"] = int(pixels.getexif().get(274, 1))
                except (OSError, ValueError) as exc:
                    errors.append(issue("source_image_unavailable", f"{image.path}: {redact(str(exc))}", severity="blocking"))
            formats = []
            options.formats = list(dict.fromkeys(options.formats))
            for format_id in options.formats:
                try:
                    adapter = registry.get(format_id, project["task"], "exporter")
                    issues = errors + adapter.preflight(selected, options.include_images)
                    version = adapter.version
                except ValueError as exc:
                    issues, version = [issue("unsupported_format", str(exc), severity="blocking")], None
                formats.append(dict(format=format_id, version=version, issues=issues,
                    preserved=(["图片选择", "类别名称", "实例轮廓与掩码", "孔洞及分离区域", "显式人工空标注"]
                               if project["task"] == "instance_segmentation" else
                               ["图片选择", "类别名称", "普通矩形框", "显式人工空标注"]),
                    blocked=any(i["severity"] == "blocking" for i in issues)))
            preview_id = uuid.uuid4().hex
            result = dict(preview_id=preview_id, created_at=now(), project_id=project_id,
                dataset_revision=project.get("dataset_revision"), options=options.model_dump(), formats=formats,
                image_count=len(selected.images), object_count=sum(len(i.objects) for i in selected.images),
                category_count=len(selected.categories), splits={s: sum(i.split == s for i in selected.images) for s in ("train", "val", "test", "unspecified")},
                skipped_images=selection["skipped_images"],
                snapshot_note="导出使用本次预览的固定标注快照；之后保存的标注不会混入本批导出。")
            plan = dict(result=result, project={"id": project_id, "dataset_path": project["dataset_path"]},
                        dataset=selected.model_dump(), selection=selection)
            atomic_json(self._path(project_id, "export-plans", preview_id), plan)
            return result

    def get(self, project_id, job_id):
        return json.loads(self._path(project_id, "export-jobs", job_id).read_text())

    def list(self, project_id):
        self.store.get(project_id)
        rows = [json.loads(path.read_text()) for path in (self.store.root / project_id / "export-jobs").glob("*.json")]
        return sorted(rows, key=lambda j: j["created_at"], reverse=True)

    def _save(self, job):
        atomic_json(self._path(job["project_id"], "export-jobs", job["job_id"]), job)

    def start(self, project_id, request: ExportCommit, *, retry=False):
        with self.store.locked(project_id):
            plan = json.loads(self._path(project_id, "export-plans", request.preview_id).read_text())
            if plan.get("job_id") and not retry:
                return self.get(project_id, plan["job_id"])
            enabled = []
            for row in plan["result"]["formats"]:
                if row["blocked"]:
                    raise ValueError("有不能导出的格式，请修改选择并重新预览")
                required = {i["code"] for i in row["issues"] if i["requires_ack"]}
                if not required <= set(request.acknowledgements.get(row["format"], [])):
                    raise ValueError(f"请明确确认 {row['format']} 的每项转换/信息损失")
                enabled.append(row["format"])
            job_id = uuid.uuid4().hex
            job = dict(job_id=job_id, project_id=project_id, preview_id=request.preview_id, status="queued", created_at=now(),
                       snapshot_at=plan["result"]["created_at"], cancel_requested=False, options=plan["result"]["options"],
                       acknowledgements=request.acknowledgements,
                       formats={name: dict(status="queued", completed_images=0, total_images=plan["result"]["image_count"]) for name in enabled})
            self._save(job)
            plan["job_id"] = job_id
            atomic_json(self._path(project_id, "export-plans", request.preview_id), plan)
            self._events[job_id] = threading.Event()
            threading.Thread(target=self._run, args=(project_id, job_id, plan), daemon=True, name=f"export-{job_id[:8]}").start()
            return job

    def cancel(self, project_id, job_id):
        with self.store.locked(project_id):
            job = self.get(project_id, job_id)
            if job["status"] in {"queued", "running"}:
                job["cancel_requested"] = True
                self._save(job)
                if job_id in self._events:
                    self._events[job_id].set()
            return job

    def retry(self, project_id, job_id):
        job = self.get(project_id, job_id)
        if job["status"] in {"queued", "running"}:
            raise ValueError("Cannot retry an active job")
        return self.start(project_id, ExportCommit(preview_id=job["preview_id"], acknowledgements=job["acknowledgements"]), retry=True)

    def _update(self, project_id, job_id, format_id=None, **values):
        with self.store.locked(project_id):
            job = self.get(project_id, job_id)
            (job["formats"][format_id] if format_id else job).update(values)
            self._save(job)

    def _run(self, project_id, job_id, plan):
        def checkpoint():
            if self._events[job_id].is_set():
                raise ExportCancelled("已取消导出")
        try:
            with self._slots:
                checkpoint()
                self._update(project_id, job_id, status="running")
                for format_id in plan["result"]["options"]["formats"]:
                    checkpoint()
                    self._update(project_id, job_id, format_id, status="running")
                    try:
                        options = plan["result"]["options"]
                        report = package_snapshot(self.store, plan["project"], Dataset.model_validate(plan["dataset"]),
                            format_id, options["policy"], options["include_images"], plan["selection"], checkpoint=checkpoint,
                            progress=lambda done, total, stage: self._update(project_id, job_id, format_id,
                                completed_images=done, total_images=total, stage=stage))
                        report["download_url"] = f"/api/projects/{project_id}/dataset/exports/{report['export_id']}"
                        self._update(project_id, job_id, format_id, status="completed", report=report,
                                     completed_images=len(plan["dataset"]["images"]))
                    except ExportCancelled:
                        raise
                    except Exception as exc:
                        self._update(project_id, job_id, format_id, status="failed", error=redact(str(exc)))
                job = self.get(project_id, job_id)
                completed = {f: row for f, row in job["formats"].items() if row["status"] == "completed"}
                status = "completed" if len(completed) == len(job["formats"]) else "partial" if completed else "failed"
                finished_at = now()
                job.update(status=status, finished_at=finished_at)
                if completed:
                    output = self.store.root / project_id / "export-jobs" / f"{job_id}.zip"
                    pending = output.with_suffix(".zip.part")
                    with ZipFile(pending, "w", ZIP_STORED) as bundle:
                        bundle.writestr("batch-report.json", json.dumps(job, ensure_ascii=False, indent=2))
                        for name, row in completed.items():
                            source = self.store.root / project_id / "exports" / f"{row['report']['export_id']}.zip"
                            with source.open("rb") as stream, bundle.open(f"{name}.zip", "w", force_zip64=True) as target:
                                for block in iter(lambda: stream.read(1024*1024), b""):
                                    checkpoint()
                                    target.write(block)
                    checkpoint()
                    pending.replace(output)
                self._update(project_id, job_id, status=status, finished_at=finished_at,
                    download_url=f"/api/projects/{project_id}/dataset/export/jobs/{job_id}/download" if completed else None)
        except ExportCancelled:
            with self.store.locked(project_id):
                job = self.get(project_id, job_id)
                for row in job["formats"].values():
                    if row["status"] in {"queued", "running"}:
                        row["status"] = "cancelled"
                job.update(status="cancelled", finished_at=now())
                self._save(job)
        except Exception as exc:
            try:
                self._update(project_id, job_id, status="failed", error=redact(str(exc)), finished_at=now())
            except (OSError, KeyError):
                pass
        finally:
            self._events.pop(job_id, None)

    def presets(self, project_id):
        project = self.store.get(project_id)
        return project.get("export_presets", [])

    def save_preset(self, project_id, preset):
        with self.store.locked(project_id):
            project = self.store.get(project_id)
            for name in preset.options.formats:
                registry.get(name, project["task"], "exporter")
            rows = [p for p in project.get("export_presets", []) if p["name"] != preset.name]
            if len(rows) >= 50:
                raise ValueError("At most 50 presets per project")
            rows.append(preset.model_dump())
            project["export_presets"] = rows
            self.store.save(project)
            return rows
