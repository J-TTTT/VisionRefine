"""Portable snapshots, input fingerprints, and bounded native ZIP extraction."""
from __future__ import annotations

import hashlib
import re
import stat
from pathlib import Path
from zipfile import ZipFile, BadZipFile, LargeZipFile

from .models import Dataset, contained_path, safe_relative_path


def file_digest(path: Path) -> str:
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while reading: {path.name}")
    return value.hexdigest()


def redact(value):
    """Do not publish local paths, credentials, or model connection settings."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()
                if not re.search(r"password|secret|token|api.?key|authorization|ai_adapter|base_url|source_file|source_sha256|annotation_path", k, re.I)
                and k not in {"root", "fingerprint", "roots", "files"}}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        # Absolute path fragments can also occur inside import error messages.
        return re.sub(r"(?<![\w:/])(?:/[\w.~-][^\s\"'<>]*|[A-Za-z]:[\\/][^\s\"'<>]*)", "[local-path]", value)
    return value


def portable_dataset(dataset: Dataset, paths: dict[str, str]) -> Dataset:
    data = dataset.model_dump()
    # Labels are user data, not paths (e.g. Open Images class /m/01g317).
    # Sanitize only extensible metadata; never change class/geometry semantics.
    data["provenance"] = redact(data["provenance"])
    data["report"] = redact(data["report"])
    for category in data["categories"]:
        category["provenance"] = redact(category["provenance"])
    data["sources"] = []
    for image in data["images"]:
        original = image["path"]
        image["path"] = safe_relative_path(paths[original])
        image["source_id"] = image["source_path"] = None
        image["provenance"] = redact(image["provenance"])
        image["provenance"]["original_logical_path"] = original
        for obj in image["objects"]:
            obj["attributes"] = redact(obj["attributes"])
            obj["provenance"] = redact(obj["provenance"])
    return Dataset.model_validate(data)


def extract_native(archive: Path, destination: Path, *, max_files=100_000, max_bytes=20 * 1024**3):
    try:
        return _extract_native(archive, destination, max_files=max_files, max_bytes=max_bytes)
    except (BadZipFile, LargeZipFile, NotImplementedError, RuntimeError) as exc:
        raise ValueError(f"Invalid or unsupported native ZIP: {exc}") from None


def _extract_native(archive: Path, destination: Path, *, max_files, max_bytes):
    """Extract only into a fresh application-owned directory, never onto source data."""
    with ZipFile(archive) as zip_file:
        entries = zip_file.infolist()
        if len(entries) > max_files or sum(e.file_size for e in entries) > max_bytes:
            raise ValueError("Native ZIP exceeds the file count or expanded size limit")
        seen = set()
        for entry in entries:
            name = safe_relative_path(entry.filename.rstrip("/"))
            if name.casefold() in seen:
                raise ValueError("Native ZIP has duplicate/case-colliding paths")
            seen.add(name.casefold())
            mode = (entry.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                raise ValueError("Native ZIP contains a link or special file")
            if entry.flag_bits & 1 or entry.file_size > max(1, entry.compress_size) * 1000:
                raise ValueError("Encrypted or suspiciously compressed ZIP member")
        if "dataset.json" not in seen:
            raise ValueError("Native ZIP must contain dataset.json at its root")
        destination.mkdir(parents=True, exist_ok=False)
        total = 0
        for entry in entries:
            target = contained_path(destination, entry.filename.rstrip("/"))
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(entry) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("Native ZIP exceeded expanded size limit")
                    output.write(block)
