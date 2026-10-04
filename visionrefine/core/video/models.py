"""Version-one video annotation contracts (seconds on the presentation timeline)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer

Split = Literal["unspecified", "train", "val", "test"]
ReviewState = Literal["unreviewed", "reviewed"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class Label(Contract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    name: str = Field(min_length=1, max_length=200)
    color: str = Field(default="#4f86ff", pattern=r"^#[0-9a-fA-F]{6}$")


class LabelsInput(Contract):
    labels: list[Label] = Field(default_factory=list, max_length=2000)

    @model_validator(mode="after")
    def unique(self):
        if len({label.id for label in self.labels}) != len(self.labels):
            raise ValueError("Label IDs must be unique")
        if len({label.name for label in self.labels}) != len(self.labels):
            raise ValueError("Label names must be unique")
        return self


class ProjectInput(Contract):
    name: str = Field(min_length=1, max_length=200)
    paths: list[str] = Field(default_factory=list, max_length=1000)
    labels: list[str] = Field(default_factory=list, max_length=2000)
    split: Split = "unspecified"


class ImportInput(Contract):
    paths: list[str] = Field(min_length=1, max_length=1000)


class Event(Contract):
    track_ids: list[str] = Field(default_factory=list, max_length=1000)

    @model_serializer(mode="wrap")
    def omit_empty_tracks(self, handler):
        result = handler(self)
        if not self.track_ids:
            result.pop("track_ids", None)
        return result

    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    kind: Literal["interval", "point"] = "interval"
    start: float = Field(ge=0)
    end: float | None = Field(default=None, ge=0)
    label_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    text: str = Field(default="", max_length=20000)
    reviewed: bool = False

    @model_validator(mode="after")
    def interval(self):
        if not self.label_id and not self.text:
            raise ValueError("An event needs a label or description")
        if self.kind == "interval" and (self.end is None or self.end <= self.start):
            raise ValueError("An interval must end after it starts")
        if self.kind == "point" and self.end is not None:
            if self.end != self.start:
                raise ValueError("A point event cannot have a duration")
            self.end = None
        return self


class Caption(Contract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    text: str = Field(min_length=1, max_length=20000)
    basis: Literal["visual", "audiovisual"] = "visual"
    reviewed: bool = False

    @model_validator(mode="after")
    def interval(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("A segment caption needs both start and end")
        if self.start is not None and self.end <= self.start:
            raise ValueError("A caption must end after it starts")
        return self


class Review(Contract):
    events: ReviewState = "unreviewed"
    captions: ReviewState = "unreviewed"


class Annotations(Contract):
    revision: int = Field(default=0, ge=0)
    events: list[Event] = Field(default_factory=list, max_length=10000)
    captions: list[Caption] = Field(default_factory=list, max_length=10000)
    review: Review = Field(default_factory=Review)
    updated_at: str | None = None

    @model_validator(mode="after")
    def unique(self):
        identifiers = [item.id for item in [*self.events, *self.captions]]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Annotation IDs must be unique within a video")
        return self


class SamplingInput(Contract):
    mode: Literal["current", "interval", "count", "events"]
    frame_index: int | None = Field(default=None, ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    interval: float | None = Field(default=None, gt=0)
    count: int | None = Field(default=None, ge=1, le=1000)


class ExtractInput(Contract):
    include_segmentation: bool = False
    frame_indices: list[int] = Field(min_length=1, max_length=1000)
    task: Literal["detection", "instance_segmentation"] = "detection"
    labels: list[str] = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def valid(self):
        if self.include_segmentation and self.task != "instance_segmentation":
            raise ValueError("Including video shapes requires an instance segmentation image project")
        if any(index < 0 for index in self.frame_indices):
            raise ValueError("Frame indices cannot be negative")
        if not all(label.strip() for label in self.labels):
            raise ValueError("Image labels cannot be blank")
        self.labels = list(dict.fromkeys(label.strip() for label in self.labels))
        return self


class Geometry(Contract):
    kind: Literal["polygon", "mask"]
    polygons: list[list[list[float]]] | None = None
    mask: dict | None = None

    @model_validator(mode="after")
    def shape(self):
        if self.kind == "polygon" and (self.polygons is None or self.mask is not None):
            raise ValueError("Polygon geometry needs contours and cannot contain a mask")
        if self.kind == "mask" and (self.mask is None or self.polygons is not None):
            raise ValueError("Mask geometry needs a tile mask and cannot contain contours")
        return self


class FrameRange(Contract):
    start_frame: int = Field(ge=0, strict=True)
    end_frame: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_frame < self.start_frame:
            raise ValueError("Frame ranges are inclusive and must end at or after their start")
        return self


class VisibilityRange(FrameRange):
    visibility: Literal["occluded", "outside"]
    reviewed: bool = False


class Keyframe(Contract):
    frame_index: int = Field(ge=0, strict=True)
    visibility: Literal["visible", "occluded", "outside"] = "visible"
    geometry: Geometry | None = None
    reviewed: bool = False

    @model_validator(mode="after")
    def visibility_geometry(self):
        if (self.visibility == "visible") != (self.geometry is not None):
            raise ValueError("Visible frames require geometry; fully occluded/outside frames cannot contain geometry")
        return self


class Track(FrameRange):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    name: str = Field(default="", max_length=200)
    label_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    keyframes: list[Keyframe] = Field(default_factory=list, max_length=10000)
    visibility_ranges: list[VisibilityRange] = Field(default_factory=list, max_length=10000)
    reviewed: bool = False


class SegmentationDocument(Contract):
    revision: int = Field(default=0, ge=0, strict=True)
    tracks: list[Track] = Field(default_factory=list, max_length=1000)
    reviewed_ranges: list[FrameRange] = Field(default_factory=list, max_length=10000)
    updated_at: str | None = None


class TrackOperation(Contract):
    revision: int = Field(ge=0, strict=True)
    annotations_revision: int | None = Field(default=None, ge=0, strict=True)
    operation: Literal["split", "merge", "delete_range"]
    track_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    other_track_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    new_track_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    frame_index: int | None = Field(default=None, ge=0, strict=True)
    start_frame: int | None = Field(default=None, ge=0, strict=True)
    end_frame: int | None = Field(default=None, ge=0, strict=True)


class GeometryPreview(Contract):
    frame_index: int = Field(ge=0, strict=True)
    geometry: Geometry
    operation: Literal["smooth", "shrink", "expand", "snap"]
    radius: int = Field(default=3, ge=1, le=64)
    snap_distance: int = Field(default=10, ge=1, le=32)
    protect_holes: bool = True
