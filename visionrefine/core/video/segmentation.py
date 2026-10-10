"""Sparse manual video tracks: shape validation, explicit coverage and edit operations."""
from __future__ import annotations

import copy
import json
import uuid

from .models import SegmentationDocument
from ..dataset_io.models import Annotation

MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
MAX_KEYFRAMES = 20000
MAX_TOTAL_RUNS = 2_000_000


def geometry_annotation(geometry, *, object_id="shape", label="object", category_id=0):
    values = geometry.model_dump() if hasattr(geometry, "model_dump") else geometry
    return Annotation.model_validate({**values, "id": object_id, "category_id": category_id, "label": label,
                                      "source": "imported_coarse"})


def geometry_values(annotation):
    return {"kind": annotation.kind, "polygons": annotation.polygons,
            "mask": annotation.mask.model_dump() if annotation.mask else None}


def covered(track, start, end):
    """All intersecting frames need explicit reviewed keyframes or reviewed state ranges."""
    start, end = max(start, track.start_frame), min(end, track.end_frame)
    if start > end:
        return True
    ranges = [(key.frame_index, key.frame_index) for key in track.keyframes if key.reviewed]
    ranges += [(state.start_frame, state.end_frame) for state in track.visibility_ranges if state.reviewed]
    cursor = start
    for left, right in sorted(ranges):
        if right < cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right + 1)
        if cursor > end:
            return True
    return False


def validate_document(payload, video, labels):
    if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()) > MAX_DOCUMENT_BYTES:
        raise ValueError("A segmentation document exceeds the 32 MB editing limit")
    document = SegmentationDocument.model_validate(payload)
    allowed = {label["id"] for label in labels}
    ids = [track.id for track in document.tracks]
    if len(ids) != len(set(ids)):
        raise ValueError("Track IDs must be unique within a video")
    if sum(len(track.keyframes) + len(track.visibility_ranges) for track in document.tracks) > MAX_KEYFRAMES:
        raise ValueError("A video supports at most 20000 explicit keyframes and visibility ranges")
    runs = 0
    for track in document.tracks:
        if track.label_id not in allowed:
            raise ValueError(f"Unknown object label: {track.label_id}")
        if track.end_frame >= video["frame_count"]:
            raise ValueError("A track extends beyond the video frame count")
        spans = []
        for key in track.keyframes:
            if not track.start_frame <= key.frame_index <= track.end_frame:
                raise ValueError("Keyframes must fall within their object's lifetime")
            spans.append((key.frame_index, key.frame_index))
            if key.geometry:
                shape = geometry_annotation(key.geometry)
                x1, y1, x2, y2 = shape.bbox
                if x1 < 0 or y1 < 0 or x2 > video["width"] or y2 > video["height"]:
                    raise ValueError("Segmentation geometry exceeds the original displayed video bounds")
                if shape.mask:
                    runs += sum(len(tile.counts) for tile in shape.mask.tiles)
                key.geometry = type(key.geometry).model_validate(geometry_values(shape))
        for state in track.visibility_ranges:
            if state.start_frame < track.start_frame or state.end_frame > track.end_frame:
                raise ValueError("Visibility ranges must fall within their object's lifetime")
            spans.append((state.start_frame, state.end_frame))
        ordered = sorted(spans)
        if any(right >= next_left for (_, right), (next_left, _) in zip(ordered, ordered[1:])):
            raise ValueError("An object cannot have overlapping keyframes or visibility states")
        if track.reviewed and not covered(track, track.start_frame, track.end_frame):
            raise ValueError("Review every frame in the object's lifetime before marking the track reviewed")
        track.keyframes.sort(key=lambda key: key.frame_index)
        track.visibility_ranges.sort(key=lambda state: state.start_frame)
    if runs > MAX_TOTAL_RUNS:
        raise ValueError("A video document exceeds the two-million mask-run editing limit")
    previous_end = -1
    document.reviewed_ranges.sort(key=lambda row: row.start_frame)
    for region in document.reviewed_ranges:
        if region.end_frame >= video["frame_count"]:
            raise ValueError("A reviewed range extends beyond the video")
        if region.start_frame <= previous_end:
            raise ValueError("Reviewed ranges cannot overlap")
        previous_end = region.end_frame
        if any(not covered(track, region.start_frame, region.end_frame) for track in document.tracks):
            raise ValueError("Reviewed ranges need explicit reviewed coverage for every active object; keyframe gaps remain unreviewed")
    return document


def _clip_ranges(ranges, start, end):
    return [{**state, "start_frame": max(state["start_frame"], start), "end_frame": min(state["end_frame"], end)}
            for state in ranges if state["start_frame"] <= end and state["end_frame"] >= start]


def operate(document, annotations, operation, index):
    result, events = copy.deepcopy(document), copy.deepcopy(annotations)
    track = next((track for track in result["tracks"] if track["id"] == operation.track_id), None)
    if track is None:
        raise ValueError("Selected object does not exist")
    changed_events = False
    if operation.operation == "split":
        frame = operation.frame_index
        if frame is None or not track["start_frame"] < frame <= track["end_frame"]:
            raise ValueError("Split at a frame after the object's start and within its lifetime")
        new_id = operation.new_track_id or f"track-{uuid.uuid4().hex[:16]}"
        if any(item["id"] == new_id for item in result["tracks"]):
            raise ValueError("The new track ID already exists")
        new_track = copy.deepcopy(track)
        new_track.update(id=new_id, name=f"{track['name']} (split)"[:200], start_frame=frame, reviewed=False)
        old_end = track["end_frame"]
        track.update(end_frame=frame - 1, reviewed=False)
        new_track["keyframes"] = [key for key in track["keyframes"] if key["frame_index"] >= frame]
        track["keyframes"] = [key for key in track["keyframes"] if key["frame_index"] < frame]
        new_track["visibility_ranges"] = _clip_ranges(track["visibility_ranges"], frame, old_end)
        track["visibility_ranges"] = _clip_ranges(track["visibility_ranges"], track["start_frame"], frame - 1)
        result["tracks"].append(new_track)
        boundary = index["timestamps"][frame]
        for event in events["events"]:
            if track["id"] not in event.get("track_ids", []):
                continue
            refs = list(event["track_ids"])
            if event["start"] >= boundary:
                refs.remove(track["id"])
            overlaps_new = event["end"] > boundary if event["kind"] == "interval" else event["start"] >= boundary
            if overlaps_new:
                refs.append(new_id)
            refs = list(dict.fromkeys(refs))
            if refs != event["track_ids"]:
                event.update(track_ids=refs, reviewed=False)
                changed_events = True
    elif operation.operation == "merge":
        other = next((row for row in result["tracks"] if row["id"] == operation.other_track_id), None)
        if other is None or other["id"] == track["id"]:
            raise ValueError("Choose a different object to merge")
        if other["label_id"] != track["label_id"]:
            raise ValueError("Assign the same object category before merging tracks")
        track.update(start_frame=min(track["start_frame"], other["start_frame"]), end_frame=max(track["end_frame"], other["end_frame"]), reviewed=False)
        track["keyframes"].extend(other["keyframes"])
        track["visibility_ranges"].extend(other["visibility_ranges"])
        result["tracks"] = [row for row in result["tracks"] if row["id"] != other["id"]]
        for event in events["events"]:
            if other["id"] in event.get("track_ids", []):
                event.update(track_ids=list(dict.fromkeys(track["id"] if ref == other["id"] else ref for ref in event["track_ids"])), reviewed=False)
                changed_events = True
    else:
        start, end = operation.start_frame, operation.end_frame
        if start is None or end is None or start > end or end >= index["frame_count"]:
            raise ValueError("Choose a valid inclusive frame range")
        track["keyframes"] = [key for key in track["keyframes"] if not start <= key["frame_index"] <= end]
        track["visibility_ranges"] = (_clip_ranges(track["visibility_ranges"], 0, start - 1)
                                      + _clip_ranges(track["visibility_ranges"], end + 1, index["frame_count"] - 1))
        track["reviewed"] = False
    result["reviewed_ranges"] = []
    if changed_events:
        events["review"]["events"] = "unreviewed"
    return result, events, changed_events
