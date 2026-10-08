"""Public contracts shared by the annotation server and optional model worker."""
from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

Capability = Literal["image_detection", "image_segmentation", "video_segmentation", "video_caption", "video_events"]
CAPABILITIES = list(Capability.__args__)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Provider(Contract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    name: str = Field(min_length=1, max_length=120)
    protocol: Literal["visionrefine_http", "openai_compatible"] = "visionrefine_http"
    base_url: str = Field(max_length=2000)
    model: str = Field(min_length=1, max_length=300)
    capabilities: list[Capability] = Field(min_length=1, max_length=5)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    timeout: int = Field(default=600, ge=10, le=3600)

    @model_validator(mode="after")
    def valid(self):
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Use an HTTP(S) base URL without credentials, query or fragment")
        self.base_url = self.base_url.rstrip("/")
        if self.protocol == "openai_compatible" and any("segmentation" in cap for cap in self.capabilities):
            raise ValueError("Segmentation requires the VisionRefine worker protocol")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("Duplicate capability")
        return self


class Configuration(Contract):
    revision: int = Field(default=0, ge=0)
    providers: list[Provider] = Field(max_length=50)
    defaults: dict[Capability, str]

    @model_validator(mode="after")
    def references(self):
        providers = {p.id: p for p in self.providers}
        if len(providers) != len(self.providers):
            raise ValueError("Provider IDs must be unique")
        if set(self.defaults) != set(CAPABILITIES):
            raise ValueError("Configure a default for every capability")
        for cap, pid in self.defaults.items():
            if pid not in providers or cap not in providers[pid].capabilities:
                raise ValueError(f"Provider does not support {cap}")
        return self


class Bindings(Contract):
    revision: int = Field(default=0, ge=0)
    providers: dict[Capability, str] = Field(default_factory=dict)


class Prompt(Contract):
    points: list[list[float]] = Field(default_factory=list, max_length=64)
    point_labels: list[Literal[0, 1]] = Field(default_factory=list, max_length=64)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    geometry: dict | None = None

    @model_validator(mode="after")
    def coordinates(self):
        if len(self.points) != len(self.point_labels) or any(len(p) != 2 for p in self.points):
            raise ValueError("Every x/y point needs a foreground (1) or background (0) label")
        if self.box and (self.box[2] <= self.box[0] or self.box[3] <= self.box[1]):
            raise ValueError("The box must use x1,y1,x2,y2 with positive area")
        return self


class JobInput(Contract):
    capability: Capability
    image: str | None = Field(default=None, max_length=4000)
    video_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    track_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,96}$")
    label: str | None = Field(default=None, max_length=200)
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, ge=0)
    seed_frame: int | None = Field(default=None, ge=0)
    sample_count: int = Field(default=12, ge=2, le=32)
    threshold: float = Field(default=0.2, ge=0.01, le=0.99)
    instruction: str = Field(default="", max_length=2000)
    prompt: Prompt | None = None

    @model_validator(mode="after")
    def resource(self):
        if self.capability.startswith("video_"):
            if not self.video_id or self.image is not None or self.prompt is not None:
                raise ValueError("Video tasks need one video resource")
        elif not self.image or self.video_id is not None or self.track_id is not None:
            raise ValueError("Image tasks need one image resource")
        return self


class ReviewInput(Contract):
    action: Literal["accept", "reject"]
    item_ids: list[str] = Field(min_length=1, max_length=2000)


def default_configuration():
    endpoint = "http://127.0.0.1:8030"
    return Configuration(providers=[
        Provider(id="default-detection", name="默认目标检测 · OWLv2", base_url=endpoint,
                 model="google/owlv2-base-patch16-ensemble", capabilities=["image_detection"]),
        Provider(id="default-segmentation", name="默认分割 · SAM 2.1 Tiny", base_url=endpoint,
                 model="facebook/sam2.1-hiera-tiny", capabilities=["image_segmentation", "video_segmentation"]),
        Provider(id="default-understanding", name="默认视频理解 · Qwen3-VL 2B", base_url=endpoint,
                 model="Qwen/Qwen3-VL-2B-Instruct", capabilities=["video_caption", "video_events"]),
    ], defaults={cap: "default-" + ("detection" if cap == "image_detection" else "segmentation" if "segmentation" in cap else "understanding") for cap in CAPABILITIES})
