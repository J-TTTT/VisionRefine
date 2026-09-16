from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return value[:48] or "project"


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict]:
        projects = []
        for path in sorted(self.root.glob("*/project.json")):
            try:
                projects.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(projects, key=lambda row: row.get("created_at", ""), reverse=True)

    def create(self, payload: dict) -> dict:
        project_id = f"{_slug(payload['name'])}-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        project = {
            "id": project_id,
            "name": payload["name"].strip(),
            "task": payload["task"],
            "labels": payload.get("labels") or (["person"] if payload["task"] == "detection" else []),
            "labels_locked": bool(payload.get("labels_locked", False)),
            "dataset_path": payload["dataset_path"],
            "annotation_path": payload.get("annotation_path") or None,
            "model_max_side": int(payload.get("model_max_side", 1536)),
            "status": "imported",
            "created_at": now,
            "analysis": None,
            "ai_adapter": None,
        }
        folder = self.root / project_id
        folder.mkdir(parents=True)
        (folder / "revisions").mkdir()
        self.save(project)
        return project

    def save(self, project: dict) -> None:
        folder = self.root / project["id"]
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / "project.json.tmp"
        temp.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(folder / "project.json")

    def get(self, project_id: str) -> dict:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", project_id):
            raise KeyError(project_id)
        path = self.root / project_id / "project.json"
        if not path.exists():
            raise KeyError(project_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, project_id: str) -> Path:
        """Move only VisionRefine metadata to recoverable local trash."""
        project = self.get(project_id)
        source = self.root / project["id"]
        trash = self.root.parent / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = trash / f"{project_id}-{stamp}"
        counter = 1
        while destination.exists():
            destination = trash / f"{project_id}-{stamp}-{counter}"
            counter += 1
        shutil.move(str(source), str(destination))
        return destination
