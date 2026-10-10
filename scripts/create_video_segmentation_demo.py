#!/usr/bin/env python3
"""Create a NEW editable video segmentation demo, preserving every existing project.

Run from the repository:
  .venv/bin/python -m scripts.create_video_segmentation_demo
  .venv/bin/python -m scripts.create_video_segmentation_demo --workspace /tmp/demo-workspace

Requires the video extra (PyAV and NumPy). The synthetic source is four seconds,
320×240, 12 FPS. Sparse draft contours mark only visible pixels; unannotated frames
are intentionally left for practice. No content is downloaded.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import time
import uuid

import av
import numpy as np

from visionrefine.core.store import ProjectStore
from visionrefine.core.video.models import LabelsInput, ProjectInput
from visionrefine.core.video.service import VideoService

WIDTH, HEIGHT, FPS, FRAME_COUNT = 320, 240, 12, 48
OBJECT_WIDTH, OBJECT_TOP, OBJECT_BOTTOM = 36, 90, 150
OCCLUDER_LEFT, OCCLUDER_RIGHT = 140, 200
KEYFRAMES = (0, 12, 19, 21, 27, 30, 36, 47)
TRACK_ID = "demo-orange-object"


def object_left(frame_index: int) -> int:
    return 16 + round(264 * frame_index / (FRAME_COUNT - 1))


def visible_geometry(frame_index: int) -> dict | None:
    """Pixel-edge polygons matching the half-open NumPy drawing slices below."""
    left = object_left(frame_index)
    right = left + OBJECT_WIDTH
    parts = []
    for a, b in ((left, min(right, OCCLUDER_LEFT)), (max(left, OCCLUDER_RIGHT), right)):
        if b > a:
            parts.append([[a, OBJECT_TOP], [b, OBJECT_TOP], [b, OBJECT_BOTTOM], [a, OBJECT_BOTTOM]])
    return {"kind": "polygon", "polygons": parts, "mask": None} if parts else None


def create_source(path: Path) -> None:
    with av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.time_base = Fraction(1, FPS)
        stream.codec_context.time_base = Fraction(1, FPS)
        stream.codec_context.max_b_frames = 0
        stream.options = {"preset": "fast", "crf": "12"}
        for frame_index in range(FRAME_COUNT):
            pixels = np.full((HEIGHT, WIDTH, 3), (25, 38, 53), dtype=np.uint8)
            # Background grid and ground line remain behind both foreground objects.
            pixels[::30, :, :] = (38, 52, 67)
            pixels[:, ::40, :] = (38, 52, 67)
            pixels[182:184, :, :] = (89, 103, 116)
            left = object_left(frame_index)
            pixels[OBJECT_TOP:OBJECT_BOTTOM, left:left+OBJECT_WIDTH, :] = (243, 149, 48)
            pixels[60:180, OCCLUDER_LEFT:OCCLUDER_RIGHT, :] = (118, 142, 163)
            # Progress bar avoids the annotated object and gives a playback cue.
            pixels[216:220, 16:304, :] = (56, 70, 82)
            pixels[216:220, 16:16+round(288*(frame_index+1)/FRAME_COUNT), :] = (74, 193, 165)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = frame_index, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def wait_for_import(service: VideoService, project_id: str, job_id: str) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        job = next(item for item in service.jobs(project_id) if item["id"] == job_id)
        if job["status"] == "completed":
            return
        if job["status"] in {"failed", "cancelled", "interrupted"}:
            raise RuntimeError(f"Demo import failed: {job.get('error') or job['status']}")
        time.sleep(.05)
    service.cancel_job(project_id, job_id)
    raise RuntimeError("Demo import did not complete within 180 seconds; cancellation requested")


def build_demo(workspace: Path, output: Path) -> dict:
    # Each invocation gets an isolated source folder and a new project ID.
    source_folder = output.resolve() / f"run-{uuid.uuid4().hex[:12]}"
    source_folder.mkdir(parents=True, exist_ok=False)
    source = source_folder / "visible-only-occlusion-4s.mp4"
    create_source(source)
    service = VideoService(ProjectStore(workspace.resolve() / "projects"))
    created = service.create(ProjectInput(
        name="[体验] 视频实例分割：遮挡与重新出现", paths=[str(source)], labels=["经过遮挡物"]
    ))
    project_id = created["project"]["id"]
    wait_for_import(service, project_id, created["job"]["id"])
    video = service.videos(project_id)[0]
    video_id = video["id"]
    if (video["width"], video["height"], video["frame_count"]) != (WIDTH, HEIGHT, FRAME_COUNT):
        raise RuntimeError("Imported demo dimensions or frame count differ from the generated source")
    index = service.index(project_id, video_id)
    service.save_object_labels(project_id, LabelsInput(labels=[
        {"id": "object-orange", "name": "橙色矩形", "color": "#f39530"}
    ]))
    hidden = [frame for frame in range(FRAME_COUNT) if visible_geometry(frame) is None]
    assert hidden == list(range(hidden[0], hidden[-1] + 1))
    document = service.segmentation(project_id, video_id)
    document["tracks"] = [{
        "id": TRACK_ID, "name": "橙色矩形 01", "label_id": "object-orange",
        "start_frame": 0, "end_frame": FRAME_COUNT - 1, "reviewed": False,
        "keyframes": [
            {"frame_index": frame, "visibility": "visible", "geometry": visible_geometry(frame), "reviewed": False}
            for frame in KEYFRAMES
        ],
        "visibility_ranges": [{"start_frame": hidden[0], "end_frame": hidden[-1],
                               "visibility": "occluded", "reviewed": False}],
    }]
    service.save_segmentation(project_id, video_id, document)
    annotations = service.annotations(project_id, video_id)
    annotations["events"] = [{
        "id": "event-pass-occluder", "kind": "interval", "start": index["timestamps"][16],
        "end": index["timestamps"][33], "label_id": service.labels(project_id)[0]["id"],
        "text": "橙色矩形向右移动，经过灰蓝色遮挡物后重新出现。", "track_ids": [TRACK_ID], "reviewed": False,
    }]
    annotations["captions"] = [{
        "id": "caption-scene", "text": "橙色矩形从左向右移动，经过固定遮挡物时先部分遮挡、完全遮挡，再重新出现。",
        "basis": "visual", "reviewed": False,
    }]
    service.save_annotations(project_id, video_id, annotations)
    return {
        "project_id": project_id, "video_id": video_id, "source": str(source),
        "workspace": str(workspace.resolve()),
        "open_path": f"/video?project={project_id}&video={video_id}",
        "keyframes_zero_based": list(KEYFRAMES), "occluded_frames_zero_based": [hidden[0], hidden[-1]],
        "reviewed": False,
        "note": "仅预置部分可见帧和完全遮挡区间；其余帧保持未标注，所有内容均待审核。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"),
                        help="Workspace containing projects/ (default: workspace)")
    parser.add_argument("--output", type=Path,
                        help="Source video folder (default: <workspace>/cache/video-segmentation-demo)")
    args = parser.parse_args()
    result = build_demo(args.workspace, args.output or args.workspace / "cache/video-segmentation-demo")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
