"""Format-independent dataset contract. Coordinates are original-image pixels."""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Task = Literal["detection", "instance_segmentation", "grounding", "captioning", "vqa", "ocr", "classification"]
Split = Literal["train", "val", "test", "unspecified"]
RevisionSource = Literal["unreviewed", "imported_coarse", "ai_suggestion", "human_reviewed", "final_reviewed"]


class Issue(BaseModel):
    severity: Literal["warning", "error"] = "warning"
    code: str
    location: str
    message: str


class ImportReport(BaseModel):
    image_count: int = 0
    object_count: int = 0
    empty_images: int = 0
    skipped_images: int = 0
    skipped_objects: int = 0
    issues: list[Issue] = Field(default_factory=list)

    def add(self, code: str, location: str, message: str, severity: str = "warning") -> None:
        self.issues.append(Issue(code=code, location=location, message=message, severity=severity))


class Category(BaseModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1)
    # Original IDs belong to provenance, not to editor or inference logic.
    source_id: int | None = None
    provenance: dict = Field(default_factory=dict)


class Annotation(BaseModel):
    id: str
    kind: Literal["bbox"] = "bbox"
    category_id: int
    label: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: RevisionSource = "imported_coarse"
    attributes: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_box(self):
        x1, y1, x2, y2 = self.bbox
        if not all(math.isfinite(v) for v in self.bbox) or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must be finite xyxy with positive area")
        if "iscrowd" in self.attributes and self.attributes["iscrowd"] not in (0, 1):
            raise ValueError("iscrowd must be 0 or 1")
        return self


class ImageRecord(BaseModel):
    id: str
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    split: Split = "unspecified"
    status: RevisionSource = "unreviewed"
    objects: list[Annotation] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    source_id: str | None = None
    source_path: str | None = None
    sha256: str | None = None

    @model_validator(mode="after")
    def relative_path(self):
        self.path = safe_relative_path(self.path)
        if self.source_path is not None:
            self.source_path = safe_relative_path(self.source_path)
        if bool(self.source_id) != bool(self.source_path):
            raise ValueError("Image source_id and source_path must be supplied together")
        return self


class DatasetSource(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    root: str
    format: str
    annotation_path: str | None = None
    provenance: dict = Field(default_factory=dict)


class Dataset(BaseModel):
    schema_version: Literal["1.0", "2.0"] = "2.0"
    task: Task = "detection"
    categories: list[Category] = Field(default_factory=list)
    images: list[ImageRecord] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    report: ImportReport = Field(default_factory=ImportReport)
    sources: list[DatasetSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self):
        categories = {c.id: c.name for c in self.categories}
        if len(categories) != len(self.categories) or len(set(categories.values())) != len(categories):
            raise ValueError("Category IDs and names must be unique")
        if len({i.path for i in self.images}) != len(self.images):
            raise ValueError("Image paths must be unique")
        if len({i.id for i in self.images}) != len(self.images):
            raise ValueError("Image IDs must be unique")
        sources = {s.id for s in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("Source IDs must be unique")
        for image in self.images:
            if image.source_id is not None and image.source_id not in sources:
                raise ValueError("Image references an unknown source")
            if len({o.id for o in image.objects}) != len(image.objects):
                raise ValueError("Annotation IDs must be unique within each image")
            for obj in image.objects:
                if categories.get(obj.category_id) != obj.label:
                    raise ValueError("Annotation category does not match the category catalog")
                x1, y1, x2, y2 = obj.bbox
                if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height:
                    raise ValueError("Annotation is outside image bounds")
        return self


def safe_relative_path(value: str) -> str:
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or "\\" in value or ":" in value or any(ord(c) < 32 for c in value)):
        raise ValueError("Expected a nonempty portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError("Image path must stay within the dataset root")
    return path.as_posix()


def contained_path(root: Path, relative: str) -> Path:
    path = (root / safe_relative_path(relative)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes the dataset root (including symlinks)")
    return path
