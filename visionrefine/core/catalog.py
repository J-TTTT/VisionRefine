"""Disposable SQLite indexes for browsing immutable dataset/analysis snapshots.

JSON manifests remain the source of truth for import/export. Interactive requests
read only a page of metadata or one image, including after a service restart.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path


@contextmanager
def build_index(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with closing(sqlite3.connect(pending)) as db:
            with db:
                yield db
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)


def connect(path: Path):
    return closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True))


def write_dataset_index(path: Path, dataset, project: dict) -> None:
    roots = {source.id: source.root for source in dataset.sources}
    with build_index(path) as db:
        db.execute("CREATE TABLE images (path TEXT PRIMARY KEY, root TEXT, source_path TEXT, record TEXT)")
        db.executemany("INSERT INTO images VALUES (?, ?, ?, ?)", (
            (record.path, roots[record.source_id] if record.source_id else project["dataset_path"],
             record.source_path or record.path, record.model_dump_json())
            for record in dataset.images
        ))
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("INSERT INTO metadata VALUES ('report', ?)", (dataset.report.model_dump_json(),))


def dataset_index(store, project: dict) -> Path | None:
    revision = project.get("dataset_revision")
    if not revision:
        return None
    if not re.fullmatch(r"import-[0-9a-f]{32}", revision):
        raise ValueError("Invalid dataset revision")
    path = store.root / project["id"] / "datasets" / f"{revision}.sqlite3"
    if not path.exists():
        # Old projects pay this cost once; the index also survives restarts.
        with store.locked(project["id"]):
            if not path.exists():
                from .dataset_io.service import load_dataset
                write_dataset_index(path, load_dataset(store, project), project)
    return path


def image_location(store, project: dict, image: str):
    from .dataset_io.models import contained_path
    path = dataset_index(store, project)
    if path is None:
        root = Path(project["dataset_path"]).resolve()
        return root, contained_path(root, image)
    with connect(path) as db:
        row = db.execute("SELECT root, source_path FROM images WHERE path = ?", (image,)).fetchone()
    if row is None:
        raise KeyError(image)
    root = Path(row[0]).resolve()
    return root, contained_path(root, row[1])


def image_record(store, project: dict, image: str):
    from .dataset_io.models import ImageRecord
    path = dataset_index(store, project)
    if path is None:
        return None
    with connect(path) as db:
        row = db.execute("SELECT record FROM images WHERE path = ?", (image,)).fetchone()
    return ImageRecord.model_validate_json(row[0]) if row else None


def import_report(store, project: dict) -> dict | None:
    path = dataset_index(store, project)
    if path is None:
        return None
    with connect(path) as db:
        return json.loads(db.execute("SELECT value FROM metadata WHERE key = 'report'").fetchone()[0])


def write_analysis_index(folder: Path, analysis: dict) -> dict:
    index_id = uuid.uuid4().hex
    with build_index(folder / "analysis" / f"{index_id}.sqlite3") as db:
        db.execute("CREATE TABLE images (position INTEGER PRIMARY KEY, path TEXT UNIQUE, split TEXT, record TEXT)")
        db.executemany("INSERT INTO images VALUES (?, ?, ?, ?)", (
            (position, row["path"], row.get("split", "unspecified"), json.dumps(row, ensure_ascii=False))
            for position, row in enumerate(analysis["images"])
        ))
        db.execute("CREATE INDEX images_split ON images (split, position)")
    return {**{key: value for key, value in analysis.items() if key != "images"}, "image_index": index_id}


def image_page(store, project: dict, *, offset=0, limit=50, query="", split=None) -> dict:
    analysis = project.get("analysis") or {}
    result = dict(items=[], total=0, offset=offset, limit=limit, revision=analysis.get("image_index"))
    if not analysis:
        return result
    index_id = analysis.get("image_index", "")
    if not re.fullmatch(r"[0-9a-f]{32}", index_id):
        raise ValueError("Analyze the project to rebuild its image list")
    path = store.root / project["id"] / "analysis" / f"{index_id}.sqlite3"
    if not path.is_file():
        raise ValueError("Image list index is missing; reanalyze the project to rebuild it")
    conditions, values = [], []
    if query:
        conditions.append("instr(lower(path), lower(?)) > 0")
        values.append(query)
    if split:
        conditions.append("split = ?")
        values.append(split)
    with connect(path) as db:
        if conditions:
            where = " AND ".join(conditions)
            result["total"] = db.execute(f"SELECT count(*) FROM images WHERE {where}", values).fetchone()[0]
            rows = db.execute(f"SELECT record FROM images WHERE {where} ORDER BY position LIMIT ? OFFSET ?",
                              [*values, limit, offset]).fetchall()
        else:
            result["total"] = analysis["image_count"]
            rows = db.execute("SELECT record FROM images WHERE position >= ? ORDER BY position LIMIT ?",
                              (offset, limit)).fetchall()
    result["items"] = [json.loads(row[0]) for row in rows]
    return result
