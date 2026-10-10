"""Durable inference jobs, frozen inputs and revision-safe suggestion review."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from .. import catalog
from ..dataset_io.models import Annotation, safe_relative_path
from ..dataset_io.service import document_path, effective_annotation
from ..dataset_io.portable import file_digest
from ..video.models import Caption, Event, Keyframe
from ..video.service import VideoConflict, VideoService, atomic_json, identifier, now, read_json
from . import providers
from .contracts import Bindings, Configuration, JobInput, Provider, default_configuration

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="visionrefine-ai")
_GUARD = threading.RLock()
_ACTIVE = {}
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class AIService:
    def __init__(self, store):
        self.store = store
        self.video = VideoService(store)

    def configuration(self):
        return read_json(self.store.root.parent / "ai" / "config.json", default_configuration().model_dump())

    def save_configuration(self, payload: Configuration):
        with self.store.locked("ai-configuration"):
            current = self.configuration()
            if current["revision"] != payload.revision:
                raise VideoConflict("模型配置已变化，请刷新后再保存")
            # Deleting a provider cannot leave a project silently using another service.
            allowed = {p.id: p for p in payload.providers}
            for project in self.store.list():
                for cap, pid in self.bindings(project["id"])["providers"].items():
                    if pid not in allowed or cap not in allowed[pid].capabilities:
                        raise ValueError(f'项目 {project["name"]} 仍在使用这个模型服务')
            result = payload.model_dump()
            result["revision"] += 1
            atomic_json(self.store.root.parent / "ai" / "config.json", result)
            return result

    def folder(self, pid):
        self.store.get(pid)
        return self.store.root / pid / "ai"

    def bindings(self, pid):
        return read_json(self.folder(pid) / "bindings.json", Bindings().model_dump())

    def save_bindings(self, pid, payload: Bindings):
        with self.store.locked("ai-configuration"), self.store.locked(pid):
            if self.bindings(pid)["revision"] != payload.revision:
                raise VideoConflict("项目模型选择已变化，请刷新")
            available = {p["id"]: p for p in self.configuration()["providers"]}
            for cap, provider in payload.providers.items():
                if provider not in available or cap not in available[provider]["capabilities"]:
                    raise ValueError(f"模型服务不支持 {cap}")
            result = payload.model_dump()
            result["revision"] += 1
            atomic_json(self.folder(pid) / "bindings.json", result)
            return result

    def provider(self, pid, capability):
        config = self.configuration()
        selected = self.bindings(pid)["providers"].get(capability, config["defaults"][capability])
        provider = next((p for p in config["providers"] if p["id"] == selected), None)
        if not provider or capability not in provider["capabilities"]:
            raise ValueError("所选模型服务已失效，请在模型设置中重新选择")
        return copy.deepcopy(provider)

    def path(self, pid, jid):
        return self.folder(pid) / "jobs" / identifier(jid) / "job.json"

    def key(self, pid, jid):
        return str(self.store.root.resolve()), pid, jid

    def write_job(self, pid, job):
        path = self.path(pid, job["id"])
        atomic_json(path, job)
        atomic_json(path.with_name("summary.json"), self.summary(job))

    @staticmethod
    def summary(job):
        result = {k: v for k, v in job.items() if k != "items"}
        result["request"] = {k: v for k, v in job["request"].items() if k != "prompt"}
        result["counts"] = {s: sum(i["status"] == s for i in job["items"]) for s in ("pending", "accepted", "rejected")}
        return result

    def job(self, pid, jid):
        with self.store.locked(pid), _GUARD:
            job = read_json(self.path(pid, jid))
            if not job:
                raise KeyError(jid)
            if job["status"] not in TERMINAL and self.key(pid, jid) not in _ACTIVE:
                job.update(status="interrupted", error="服务重启中断了任务，可以重试。", updated_at=now())
                self.write_job(pid, job)
            return job

    def jobs(self, pid):
        rows = []
        for path in (self.folder(pid) / "jobs").glob("*/job.json"):
            job = read_json(path.with_name("summary.json"))
            if job is None or (job["status"] not in TERMINAL and self.key(pid, path.parent.name) not in _ACTIVE):
                job = self.summary(self.job(pid, path.parent.name))
            rows.append(job)
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    def image_path(self, project, image):
        relative = safe_relative_path(image)
        root, path = catalog.image_location(self.store, project, relative)
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError("图片不存在")
        return path

    def snapshot(self, pid, req):
        project = self.store.get(pid)
        cap = req.capability
        if cap.startswith("video_"):
            if not req.video_id or project["task"] != "video":
                raise ValueError("请选择视频项目中的视频")
            media = self.video.video(pid, req.video_id)
            index = self.video.index(pid, req.video_id)
            start = req.start_frame if req.start_frame is not None else 0
            end = req.end_frame if req.end_frame is not None else media["frame_count"] - 1
            if start > end or end >= media["frame_count"]:
                raise ValueError("帧范围超出视频或顺序不正确")
            annotations = self.video.annotations(pid, req.video_id)
            segmentation = self.video.segmentation(pid, req.video_id)
            result = dict(width=media["width"], height=media["height"], source_sha256=media["sha256"],
                          annotations_revision=annotations["revision"], segmentation_revision=segmentation["revision"],
                          start_frame=start, end_frame=end, start=index["timestamps"][start],
                          end=index["timestamps"][end + 1] if end + 1 < media["frame_count"] else media["duration"],
                          whole_video=start == 0 and end == media["frame_count"] - 1,
                          labels=self.video.labels(pid), seeds=[], protected=[])
            if cap == "video_segmentation":
                if end - start >= 1000:
                    raise ValueError("单次传播最多 1000 帧，请缩短区间")
                track = next((t for t in segmentation["tracks"] if t["id"] == req.track_id), None)
                if not track or start < track["start_frame"] or end > track["end_frame"]:
                    raise ValueError("请选择已有对象生命周期内的区间")
                if req.seed_frame is None or not start <= req.seed_frame <= end:
                    raise ValueError("起始关键帧必须在传播区间内")
                seed = next((k for k in track["keyframes"] if k["frame_index"] == req.seed_frame and k["visibility"] == "visible"), None)
                if not seed:
                    raise ValueError("请先在起始关键帧画出可见轮廓并保存")
                # Human anchors and explicit absence ranges always win over model predictions.
                result["seeds"] = [k for k in track["keyframes"] if start <= k["frame_index"] <= end
                                   and k["visibility"] == "visible" and (not k.get("provenance") or k["reviewed"] or k == seed)]
                result["protected"] = [k["frame_index"] for k in track["keyframes"] if not k.get("provenance") or k["reviewed"] or k == seed]
                result["protected_ranges"] = track["visibility_ranges"]
                result["labels"] = self.video.object_labels(pid)
                result["track_label"] = track["label_id"]
            return result
        if not req.image or project["task"] not in {"detection", "instance_segmentation"}:
            raise ValueError("请选择检测或实例分割项目中的图片")
        if (cap == "image_segmentation") != (project["task"] == "instance_segmentation"):
            raise ValueError("此模型能力与当前图片标注任务不匹配")
        path = self.image_path(project, req.image)
        with Image.open(path) as img:
            width, height = img.size
        if width * height > 16_777_216:
            raise ValueError("此模型的整图输入最多 1600 万像素；大图请使用现有切片检测或先裁剪")
        current = effective_annotation(self.store, project, req.image)
        stat = path.stat()
        result = dict(width=width, height=height, image_revision=digest(current), image_document=current,
                      source_stat=[stat.st_size, stat.st_mtime_ns], source_sha256=file_digest(path), labels=project["labels"])
        if cap == "image_segmentation":
            if req.label not in project["labels"] or not req.prompt or not (req.prompt.points or req.prompt.box or req.prompt.geometry):
                raise ValueError("请选择对象类别，并提供点、框或已有轮廓")
            prompt = req.prompt
            coords = prompt.points + ([prompt.box[:2], prompt.box[2:]] if prompt.box else [])
            if any(not 0 <= x <= width or not 0 <= y <= height for x, y in coords):
                raise ValueError("提示点或框超出原始图片")
            if prompt.geometry:
                self.validate_object({**prompt.geometry, "label": req.label}, project["labels"], width, height, "image_segmentation")
        return result

    def start(self, pid, req: JobInput):
        with self.store.locked(pid), _GUARD:
            if sum(key[0] == str(self.store.root.resolve()) and key[1] == pid for key in _ACTIVE) >= 2:
                raise VideoConflict("该项目已有两个 AI 任务，请等待完成或取消")
            snapshot = self.snapshot(pid, req)
            provider = self.provider(pid, req.capability)
            jid = "ai-" + uuid.uuid4().hex
            job = dict(id=jid, status="queued", request=req.model_dump(), provider=provider,
                       created_at=now(), updated_at=now(), error=None, progress={"message": "等待处理", "current": 0, "total": 1},
                       items=[], base={k: v for k, v in snapshot.items() if k in {"annotations_revision", "segmentation_revision", "image_revision"}})
            atomic_json(self.path(pid, jid).with_name("input.json"), snapshot)
            self.write_job(pid, job)
            flag = threading.Event()
            _ACTIVE[self.key(pid, jid)] = flag
            _EXECUTOR.submit(self.run, pid, jid, snapshot, flag)
            return copy.deepcopy(job)

    def cancel(self, pid, jid):
        with self.store.locked(pid), _GUARD:
            job = self.job(pid, jid)
            flag = _ACTIVE.get(self.key(pid, jid))
            if flag:
                flag.set()
                job.update(status="cancelling", updated_at=now())
                self.write_job(pid, job)
            return job

    def retry(self, pid, jid):
        job = self.job(pid, jid)
        if job["status"] not in {"failed", "cancelled", "interrupted"}:
            raise VideoConflict("只能重试失败、取消或中断的任务")
        # Rebuild from current annotations, so a correction is never lost on retry.
        return self.start(pid, JobInput.model_validate(job["request"]))

    @staticmethod
    def jpeg(path, frame, timestamp, max_side):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
        return {"frame_index": frame, "timestamp": timestamp, "jpeg": base64.b64encode(buffer.getvalue()).decode()}

    def prepare(self, pid, job, snapshot, checkpoint, progress):
        req = job["request"]
        payload = {**{k: v for k, v in snapshot.items() if k not in {"image_document", "source_stat", "protected", "protected_ranges"}},
                   "capability": req["capability"], "threshold": req["threshold"], "prompt": req["prompt"],
                   "label": req["label"], "seed_frame": req["seed_frame"], "instruction": req["instruction"], "frames": []}
        if req["video_id"]:
            start, end = snapshot["start_frame"], snapshot["end_frame"]
            if req["capability"] == "video_segmentation":
                indices = list(range(start, end + 1))
            else:
                count = min(req["sample_count"], end - start + 1)
                indices = sorted({round(start + (end - start) * i / max(1, count - 1)) for i in range(count)})
            index = self.video.index(pid, req["video_id"])
            for position, frame in enumerate(indices):
                checkpoint()
                payload["frames"].append(self.jpeg(self.video.frame_path(pid, req["video_id"], frame), frame, index["timestamps"][frame], 1024))
                progress("准备视频画面", position + 1, len(indices))
        else:
            path = self.image_path(self.store.get(pid), req["image"])
            stat = path.stat()
            if [stat.st_size, stat.st_mtime_ns] != snapshot["source_stat"]:
                raise ValueError("原始图片发生变化，请重新提交任务")
            payload["frames"] = [self.jpeg(path, 0, 0, 2048)]
        if sum(len(frame["jpeg"]) for frame in payload["frames"]) > 96 * 1024 * 1024:
            raise ValueError("画面输入超过 96 MB，请缩短传播区间")
        return payload

    def run(self, pid, jid, snapshot, flag):
        def checkpoint():
            if flag.is_set():
                raise providers.Cancelled()

        def progress(message, current=0, total=1):
            checkpoint()
            with self.store.locked(pid):
                job = self.job(pid, jid)
                job.update(status="running", updated_at=now(), progress={"message": message, "current": current, "total": total})
                self.write_job(pid, job)

        try:
            job = self.job(pid, jid)
            progress("准备输入快照")
            payload = self.prepare(pid, job, snapshot, checkpoint, progress)
            result = providers.infer(job["provider"], payload, checkpoint, progress)
            checkpoint()
            items = self.validate_result(job, snapshot, result)
            if len(json.dumps(items)) > 32 * 1024 * 1024:
                raise ValueError("模型结果超过 32 MB，请缩短区间")
            with self.store.locked(pid):
                checkpoint()
                job = self.job(pid, jid)
                job.update(status="completed", items=items, updated_at=now(), progress={"message": f"已生成 {len(items)} 条待审核建议", "current": 1, "total": 1})
                self.write_job(pid, job)
        except Exception as exc:
            with self.store.locked(pid):
                job = self.job(pid, jid)
                job.update(status="cancelled" if flag.is_set() or isinstance(exc, providers.Cancelled) else "failed",
                           error=None if flag.is_set() else str(exc)[:2000], updated_at=now())
                self.write_job(pid, job)
        finally:
            with _GUARD:
                _ACTIVE.pop(self.key(pid, jid), None)

    @staticmethod
    def validate_object(obj, labels, width, height, capability):
        label = obj.get("label")
        if label not in labels:
            raise ValueError("模型返回了未知的对象类别")
        data = {**obj, "id": "candidate", "category_id": labels.index(label), "source": "ai_suggestion"}
        if capability == "image_detection":
            data["kind"] = data.get("kind", "bbox")
        annotation = Annotation.model_validate(data)
        allowed = {"bbox"} if capability == "image_detection" else {"polygon", "mask"}
        if annotation.kind not in allowed:
            raise ValueError("模型返回了错误的几何类型")
        x1, y1, x2, y2 = annotation.bbox
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError("模型结果超出原始图像坐标范围")
        return annotation.model_dump()

    def validate_result(self, job, snapshot, result):
        if not isinstance(result, dict):
            raise ValueError("模型输出必须是 JSON 对象")
        cap = job["request"]["capability"]
        origin = {"job_id": job["id"], "provider_id": job["provider"]["id"], "model": job["provider"]["model"],
                  "generated_at": now(), "input_revision": digest(job["base"]), "method": cap,
                  "model_revision": str(result["model_revision"])[:128] if result.get("model_revision") else None,
                  "source_sha256": snapshot.get("source_sha256"),
                  "frame_range": [snapshot["start_frame"], snapshot["end_frame"]] if cap.startswith("video_") else None}
        key = "objects" if cap.startswith("image_") else "keyframes" if cap == "video_segmentation" else "captions" if cap == "video_caption" else "events"
        rows = result.get(key)
        if not isinstance(rows, list) or len(rows) > 2000:
            raise ValueError(f"模型需要返回 {key} 数组（最多 2000 条）")
        items, seen = [], set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("模型返回了错误的标注结构")
            iid = "suggestion-" + uuid.uuid4().hex
            if key == "objects":
                row = self.validate_object(row, snapshot["labels"], snapshot["width"], snapshot["height"], cap)
                row.update(id=iid, provenance={"ai": origin})
            elif key == "keyframes":
                row = Keyframe.model_validate({**row, "reviewed": False, "provenance": origin}).model_dump()
                frame = row["frame_index"]
                if not snapshot["start_frame"] <= frame <= snapshot["end_frame"] or frame in seen:
                    raise ValueError("模型返回了重复或越界的视频帧")
                seen.add(frame)
                if frame in snapshot["protected"] or any(r["start_frame"] <= frame <= r["end_frame"] for r in snapshot["protected_ranges"]):
                    continue
                if row["geometry"]:
                    self.validate_object({**row["geometry"], "label": "object"}, ["object"], snapshot["width"], snapshot["height"], "image_segmentation")
            else:
                row = {**row, "id": iid, "reviewed": False, "provenance": origin}
                if key == "captions":
                    # This version supplies visual frames only, never pretend to have heard audio.
                    row.update(basis="visual", start=None if snapshot["whole_video"] else snapshot["start"], end=None if snapshot["whole_video"] else snapshot["end"])
                    row = Caption.model_validate(row).model_dump()
                else:
                    row = Event.model_validate(row).model_dump()
                    if row.get("label_id") and row["label_id"] not in {label["id"] for label in snapshot["labels"]}:
                        raise ValueError("模型返回了未知的事件标签")
                    if row.get("track_ids"):
                        raise ValueError("模型不能凭空建立对象关联")
                if any(t is not None and (t < snapshot["start"] or t > snapshot["end"] + 1e-6) for t in (row.get("start"), row.get("end"))):
                    raise ValueError("模型返回的时间超出请求区间")
            items.append({"id": iid, "kind": key, "status": "pending", "value": row})
        return items

    def review(self, pid, jid, request):
        with self.store.locked(pid):
            job = self.job(pid, jid)
            if job["status"] != "completed":
                raise VideoConflict("任务尚未完成")
            ids = set(request.item_ids)
            if not ids.issubset({i["id"] for i in job["items"]}):
                raise ValueError("未知的建议 ID")
            chosen = [i for i in job["items"] if i["id"] in ids and i["status"] == "pending"]
            if not chosen:
                return job
            if request.action == "accept":
                self.apply(pid, job, chosen)
            for item in chosen:
                item["status"] = "accepted" if request.action == "accept" else "rejected"
            job["updated_at"] = now()
            self.write_job(pid, job)
            return job

    def apply(self, pid, job, chosen):
        req = job["request"]
        cap = req["capability"]
        snapshot = read_json(self.path(pid, job["id"]).with_name("input.json"))
        if req["video_id"]:
            vid = req["video_id"]
            media = self.video.video(pid, vid)
            if media["sha256"] != snapshot["source_sha256"]:
                raise VideoConflict("视频来源发生变化，请重新生成建议")
            if cap == "video_segmentation":
                current = self.video.segmentation(pid, vid)
                if current["revision"] != job["base"]["segmentation_revision"]:
                    existing = next((t for t in current["tracks"] if t["id"] == req["track_id"]), {})
                    if all(i["value"] in existing.get("keyframes", []) for i in chosen):
                        job["base"]["segmentation_revision"] = current["revision"]
                        return
                    raise VideoConflict("分割已被修改，请保留修改并重新传播")
                track = next((t for t in current["tracks"] if t["id"] == req["track_id"]), None)
                if not track:
                    raise VideoConflict("对象已删除")
                keys = {k["frame_index"]: k for k in track["keyframes"]}
                for item in chosen:
                    row = item["value"]
                    old = keys.get(row["frame_index"])
                    if old and (old["reviewed"] or not old.get("provenance")):
                        raise VideoConflict("该帧已有人工标注，请重新生成建议")
                    keys[row["frame_index"]] = row
                track.update(keyframes=sorted(keys.values(), key=lambda k: k["frame_index"]), reviewed=False)
                selected_frames = {i["value"]["frame_index"] for i in chosen}
                current["reviewed_ranges"] = [r for r in current["reviewed_ranges"] if not any(r["start_frame"] <= f <= r["end_frame"] for f in selected_frames)]
                saved = self.video.save_segmentation(pid, vid, current)
                job["base"]["segmentation_revision"] = saved["revision"]
            else:
                current = self.video.annotations(pid, vid)
                if current["revision"] != job["base"]["annotations_revision"]:
                    if all(i["value"] in current[i["kind"]] for i in chosen):
                        job["base"]["annotations_revision"] = current["revision"]
                        return
                    raise VideoConflict("事件或描述已被修改，请重新生成建议")
                for item in chosen:
                    # Deterministic IDs also make recovery after a process exit idempotent.
                    if not any(v["id"] == item["id"] for v in current[item["kind"]]):
                        current[item["kind"]].append(item["value"])
                    current["review"][item["kind"]] = "unreviewed"
                saved = self.video.save_annotations(pid, vid, current)
                job["base"]["annotations_revision"] = saved["revision"]
        else:
            project = self.store.get(pid)
            path = self.image_path(project, req["image"])
            stat = path.stat()
            if [stat.st_size, stat.st_mtime_ns] != snapshot["source_stat"]:
                raise VideoConflict("图片来源已变化，请重新生成")
            current = effective_annotation(self.store, project, req["image"])
            if digest(current) != job["base"]["image_revision"]:
                if all(i["value"] in current["objects"] for i in chosen):
                    job["base"]["image_revision"] = digest(current)
                    return
                raise VideoConflict("图片标注已修改，请重新生成建议")
            values = [self.validate_object(i["value"], project["labels"], snapshot["width"], snapshot["height"], cap) for i in chosen]
            for row, item in zip(values, chosen):
                row.update(id=item["id"], provenance=item["value"]["provenance"])
            base = effective_annotation(self.store, project, req["image"], include_draft=False)
            draft = {**current, "objects": current["objects"] + values, "status": "ai_suggestion",
                     "revision_id": "ai-draft-" + uuid.uuid4().hex, "base_revision": digest(base), "updated_at": now()}
            atomic_json(document_path(self.store, pid, req["image"], "ai_drafts"), draft)
            job["base"]["image_revision"] = digest(draft)
