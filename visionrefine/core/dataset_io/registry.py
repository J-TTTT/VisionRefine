"""Adapters convert at the boundary; the workspace only sees Dataset objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import Dataset


class Importer(Protocol):
    def read(self, root: Path, source: Path | None, labels: list[str], split: str) -> Dataset: ...


class Exporter(Protocol):
    def write(self, dataset: Dataset, output: Path) -> dict: ...


@dataclass(frozen=True)
class Format:
    id: str
    title: str
    tasks: tuple[str, ...]
    importer: Importer | None = None
    exporter: Exporter | None = None
    version: str = "1.0"
    geometries: tuple[str, ...] = ("bbox",)
    attributes: tuple[str, ...] = ()
    preserves_confidence: bool = False
    preserves_splits: bool = False
    status: str = "tested"
    input: dict = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    reference: str = ""

    def preflight(self, dataset: Dataset, include_images: bool) -> list[dict]:
        from .compatibility import inspect_export
        issues = inspect_export(self, dataset, include_images)
        if self.exporter and hasattr(self.exporter, "preflight"):
            issues.extend(self.exporter.preflight(dataset, include_images))
        return issues


class FormatRegistry:
    def __init__(self):
        self._formats: dict[str, Format] = {}

    def register(self, adapter: Format) -> None:
        if adapter.id in self._formats:
            raise ValueError(f"Format already registered: {adapter.id}")
        if adapter.status not in {"experimental", "tested", "stable"}:
            raise ValueError("Invalid compatibility status")
        self._formats[adapter.id] = adapter

    def get(self, format_id: str, task: str, operation: str) -> Format:
        adapter = self._formats.get(format_id)
        if not adapter or task not in adapter.tasks or getattr(adapter, operation, None) is None:
            raise ValueError(f"Unsupported {operation} format {format_id!r} for task {task!r}")
        return adapter

    def capabilities(self) -> list[dict]:
        return [dict(id=f.id, title=f.title, tasks=list(f.tasks), version=f.version,
                     geometries=list(f.geometries), attributes=list(f.attributes),
                     preserves_confidence=f.preserves_confidence, preserves_splits=f.preserves_splits,
                     status=f.status, input=f.input, limitations=list(f.limitations), reference=f.reference,
                     can_import=f.importer is not None, can_export=f.exporter is not None)
                for f in self._formats.values()]
