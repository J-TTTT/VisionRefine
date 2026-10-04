"""Exact video indexing and bounded, browser-compatible media derivatives.

PyAV is optional so existing image-only installations keep working. Frame identity
comes from decoded presentation timestamps, never from ``time * average_fps``.
All persisted times are seconds from the first decoded video frame.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Callable
import hashlib
import json
import os
import shutil
import tempfile

from PIL import Image

CACHE_VERSION = 2
MAX_SOURCE_BYTES = 8 * 1024**3
MAX_FRAME_PIXELS = 67_108_864
MAX_FRAME_COUNT = 250_000
MAX_DURATION_SECONDS = 6 * 60 * 60
PREVIEW_EDGE = 1280
TIME_BASE = Fraction(1, 1_000_000)


class VideoMediaError(ValueError):
    """An actionable media validation or conversion failure."""


class VideoCancelledError(VideoMediaError):
    """The caller requested cancellation; partial derivatives are discarded."""


Progress = Callable[[dict], None]
Cancel = Callable[[], bool]


def _av():
    try:
        import av
    except ImportError as exc:
        raise VideoMediaError('视频功能需要 PyAV：请运行 pip install -e ".[video]"。') from exc
    return av


def capabilities() -> dict:
    try:
        av = _av()
        for codec in ("libx264", "aac", "libvpx", "libopus"):
            av.codec.Codec(codec, "w")
        return {"available": True, "version": av.__version__, "exact_timestamps": True,
                "preview": "H.264/AAC MP4 + VP8/Opus WebM", "error": None}
    except Exception as exc:
        return {"available": False, "exact_timestamps": False, "error": str(exc)}


def _check(cancel: Cancel | None):
    if cancel and cancel():
        raise VideoCancelledError("视频处理已取消。")


def _progress(callback: Progress | None, current: int, total: int, message: str):
    if callback:
        callback({"current": current, "total": total, "message": message})


def _signature(source: Path) -> dict:
    stat = source.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _sha256(source: Path, cancel: Cancel | None) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            _check(cancel)
            digest.update(block)
    return digest.hexdigest()


def _rotation(frame, stream) -> int:
    angle = int(getattr(frame, "rotation", 0) or stream.metadata.get("rotate", 0))
    angle %= 360
    if angle not in (0, 90, 180, 270):
        raise VideoMediaError("暂不支持非直角的视频旋转；请先将画面旋转烘焙到视频中。")
    return angle


def _image(frame, angle: int) -> Image.Image:
    image = frame.to_image()
    if angle:
        image = image.rotate(angle, expand=True)
    return image


def _index(source: Path, cancel: Cancel | None, progress: Progress | None) -> dict:
    av = _av()
    with av.open(str(source), mode="r") as container:
        if not container.streams.video:
            raise VideoMediaError("文件没有可解码的视频轨道。")
        stream = container.streams.video[0]
        if stream.width * stream.height > MAX_FRAME_PIXELS:
            raise VideoMediaError("视频分辨率超过当前处理上限（约 6700 万像素）。")
        rate = float(stream.average_rate) if stream.average_rate else None
        time_base = Fraction(stream.time_base)
        pts_values: list[int] = []
        final_duration = Fraction(0)
        angle = 0
        size = None
        for frame in container.decode(stream):
            _check(cancel)
            if frame.width * frame.height > MAX_FRAME_PIXELS:
                raise VideoMediaError("视频帧分辨率超过处理上限。")
            if frame.pts is None or frame.time_base is None:
                raise VideoMediaError("视频缺少帧时间戳，无法提供可靠的逐帧标注；请先转换视频。")
            pts = Fraction(frame.pts) * frame.time_base / time_base
            if pts.denominator != 1:
                raise VideoMediaError("视频帧时间基准发生变化，无法建立精确索引。")
            pts = int(pts)
            if pts_values and pts <= pts_values[-1]:
                raise VideoMediaError("视频含重复或倒退的显示时间戳，请先修复或转换视频。")
            frame_angle = _rotation(frame, stream)
            frame_size = (frame.height, frame.width) if frame_angle in (90, 270) else (frame.width, frame.height)
            if size is not None and (frame_size != size or frame_angle != angle):
                raise VideoMediaError("视频中途改变了尺寸或旋转方向，请拆分为尺寸固定的片段。")
            size, angle = frame_size, frame_angle
            pts_values.append(pts)
            final_duration = Fraction(getattr(frame, "duration", 0) or 0) * frame.time_base
            if len(pts_values) > MAX_FRAME_COUNT:
                raise VideoMediaError(f"视频超过 {MAX_FRAME_COUNT} 帧，请先分段导入。")
            if float((pts - pts_values[0]) * time_base) > MAX_DURATION_SECONDS:
                raise VideoMediaError("视频超过 6 小时，请先分段导入。")
            if len(pts_values) % 25 == 0:
                _progress(progress, len(pts_values), stream.frames or 0, "建立精确帧索引")
        if not pts_values or size is None:
            raise VideoMediaError("视频中没有可解码的画面。")
        start = pts_values[0] * time_base
        timestamps = [float((pts - pts_values[0]) * time_base) for pts in pts_values]
        duration_source = "last_frame_duration"
        if final_duration <= 0:
            declared_end = ((stream.start_time or 0) + (stream.duration or 0)) * time_base
            if declared_end > pts_values[-1] * time_base:
                final_duration = declared_end - pts_values[-1] * time_base
                duration_source = "stream_duration"
            elif len(pts_values) > 1:
                final_duration = (pts_values[-1] - pts_values[-2]) * time_base
                duration_source = "estimated_last_interval"
            elif rate and rate > 0:
                final_duration = Fraction(1 / rate)
                duration_source = "estimated_nominal_rate"
            else:
                raise VideoMediaError("无法确定单帧视频的显示时长。")
        duration = float((pts_values[-1] - pts_values[0]) * time_base + final_duration)
        return {
            "width": size[0], "height": size[1], "encoded_width": stream.width,
            "encoded_height": stream.height, "rotation": angle, "duration": duration,
            "duration_source": duration_source, "duration_estimated": duration_source.startswith("estimated"),
            "fps": rate, "frame_count": len(pts_values), "timestamps": timestamps,
            "source_pts": pts_values, "time_base": [time_base.numerator, time_base.denominator],
            "source_start_time": float(start), "has_audio": bool(container.streams.audio),
            "audio_track_count": len(container.streams.audio), "video_stream_index": stream.index,
            "codec": stream.codec_context.name,
        }


def _preview(source: Path, folder: Path, metadata: dict, cancel: Cancel | None,
             progress: Progress | None, *, webm: bool = False) -> list[dict]:
    av = _av()
    import numpy as np

    width, height = metadata["width"], metadata["height"]
    scale = min(1.0, PREVIEW_EDGE / max(width, height))
    preview_width = max(2, int(width * scale) // 2 * 2)
    preview_height = max(2, int(height * scale) // 2 * 2)
    frame_count = metadata["frame_count"]
    thumb_indices = set(round(i * (frame_count - 1) / max(1, min(12, frame_count) - 1))
                        for i in range(min(12, frame_count)))
    if webm:
        thumb_indices = set()
    thumbnails: list[dict] = []
    source_tb = Fraction(*metadata["time_base"])
    source_start = metadata["source_pts"][0] * source_tb
    video_end = Fraction(str(metadata["duration"]))

    with av.open(str(source), "r") as source_container, av.open(
            str(folder / ("preview.webm" if webm else "preview.mp4")), "w",
            format="webm" if webm else "mp4",
            options={} if webm else {"movflags": "+faststart"}) as output:
        video_in = source_container.streams.video[0]
        video_out = output.add_stream("libvpx" if webm else "libx264", rate=Fraction(str(metadata["fps"] or 25)).limit_denominator(100_000))
        video_out.width, video_out.height = preview_width, preview_height
        video_out.pix_fmt = "yuv420p"
        video_out.time_base = TIME_BASE
        video_out.codec_context.time_base = TIME_BASE
        video_out.codec_context.max_b_frames = 0
        video_out.codec_context.gop_size = 30
        video_out.options = ({"deadline": "realtime", "cpu-used": "5", "crf": "12"}
                             if webm else {"preset": "veryfast", "crf": "20"})
        if webm:
            video_out.bit_rate = max(300_000, preview_width * preview_height * 8)
        preview_times = [round((pts - metadata["source_pts"][0]) * source_tb / TIME_BASE)
                         for pts in metadata["source_pts"]]
        end_time = round(video_end / TIME_BASE)
        preview_durations = {pts: (preview_times[i + 1] if i + 1 < frame_count else end_time) - pts
                             for i, pts in enumerate(preview_times)}

        def mux_video(packet):
            # Encoders may infer the final packet duration from nominal FPS.
            # Actual durations preserve the last frame and all VFR boundaries.
            pts = round(packet.pts * packet.time_base / TIME_BASE)
            packet.duration = round(preview_durations[pts] * TIME_BASE / packet.time_base)
            output.mux(packet)

        audio_in = source_container.streams.audio[0] if source_container.streams.audio else None
        audio_out = None
        resampler = None
        if audio_in:
            audio_out = output.add_stream("libopus" if webm else "aac", rate=48_000)
            audio_out.layout = "stereo"
            audio_out.bit_rate = 128_000
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48_000)

        def mux_audio(resampled):
            # Align sound to the same video origin, trimming samples before/after
            # the video interval. Positive offsets remain intact.
            if resampled.pts is None or resampled.time_base is None:
                raise VideoMediaError("音轨缺少时间戳，无法保证音画同步；请先转换视频。")
            start_sample = round((resampled.pts * resampled.time_base - source_start) * 48_000)
            left = max(0, -start_sample)
            right = min(resampled.samples, round(video_end * 48_000) - start_sample)
            if right <= left:
                return
            if left or right < resampled.samples:
                data = np.ascontiguousarray(resampled.to_ndarray()[:, left:right])
                resampled = av.AudioFrame.from_ndarray(data, format="fltp", layout="stereo")
                resampled.sample_rate = 48_000
            resampled.pts = start_sample + left
            resampled.time_base = Fraction(1, 48_000)
            for packet in audio_out.encode(resampled):
                output.mux(packet)

        streams = (video_in, audio_in) if audio_in else (video_in,)
        index = 0
        for packet in source_container.demux(*streams):
            _check(cancel)
            for frame in packet.decode():
                _check(cancel)
                if packet.stream.type == "audio":
                    for resampled in resampler.resample(frame):
                        mux_audio(resampled)
                    continue
                image = _image(frame, metadata["rotation"])
                if index in thumb_indices:
                    thumb = image.copy()
                    thumb.thumbnail((240, 160), Image.Resampling.LANCZOS)
                    name = f"thumb-{index:08d}.jpg"
                    thumb.save(folder / name, quality=82)
                    thumbnails.append({"frame_index": index, "timestamp": metadata["timestamps"][index], "file": name})
                    if index == 0:
                        poster = image.copy()
                        poster.thumbnail((640, 640), Image.Resampling.LANCZOS)
                        poster.save(folder / "poster.jpg", quality=88)
                if image.size != (preview_width, preview_height):
                    image = image.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
                encoded_frame = av.VideoFrame.from_image(image)
                encoded_frame.pts = round((metadata["source_pts"][index] - metadata["source_pts"][0]) * source_tb / TIME_BASE)
                encoded_frame.time_base = TIME_BASE
                for encoded_packet in video_out.encode(encoded_frame):
                    mux_video(encoded_packet)
                index += 1
                if index % 10 == 0 or index == frame_count:
                    _progress(progress, index, frame_count, "生成 WebM 兼容预览" if webm else "生成播放预览与缩略图")
        if resampler:
            for resampled in resampler.resample(None):
                mux_audio(resampled)
        for packet in video_out.encode():
            mux_video(packet)
        if audio_out:
            for packet in audio_out.encode():
                output.mux(packet)
    return thumbnails


def prepare_video(source: Path, cache: Path, progress: Progress | None = None,
                  cancel: Cancel | None = None) -> dict:
    """Index and transcode a local video; return JSON-serializable metadata.

    Results include exact ``timestamps`` and original ``source_pts``. Playback
    uses normalized MP4/WebM previews; annotations/extraction use the source index.
    Cache files become usable only once the final index.json is atomically saved.
    """
    _av()
    source, cache = Path(source).resolve(), Path(cache).resolve()
    if not source.is_file():
        raise VideoMediaError("视频源文件不存在。")
    signature = _signature(source)
    if signature["size"] > MAX_SOURCE_BYTES:
        raise VideoMediaError("视频超过 8 GB，请先分段导入。")
    _check(cancel)
    _progress(progress, 0, 0, "校验视频源文件")
    digest = _sha256(source, cancel)
    cached_index = cache / "index.json"
    if cached_index.is_file():
        try:
            saved = json.loads(cached_index.read_text(encoding="utf-8"))
            if (saved.get("cache_version") == CACHE_VERSION and saved.get("sha256") == digest
                    and saved.get("source_signature") == signature
                    and (cache / "preview.mp4").is_file() and (cache / "preview.webm").is_file()
                    and (cache / "poster.jpg").is_file()):
                return saved
        except (OSError, ValueError, KeyError):
            pass
    cache.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=cache))
    try:
        _progress(progress, 0, 0, "建立精确帧索引")
        metadata = _index(source, cancel, progress)
        thumbnails = _preview(source, staging, metadata, cancel, progress)
        _preview(source, staging, metadata, cancel, progress, webm=True)
        _check(cancel)
        if _signature(source) != signature:
            raise VideoMediaError("处理期间视频源文件发生变化，请重新导入。")
        metadata.update({
            "cache_version": CACHE_VERSION, "sha256": digest, "source_signature": signature,
            "preview_path": str(cache / "preview.mp4"), "preview_webm_path": str(cache / "preview.webm"),
            "poster_path": str(cache / "poster.jpg"),
            "thumbnails": [{**thumb, "path": str(cache / thumb["file"])} for thumb in thumbnails],
        })
        for item in staging.iterdir():
            os.replace(item, cache / item.name)
        index_tmp = staging / "index.json"
        index_tmp.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        os.replace(index_tmp, cached_index)
        _progress(progress, metadata["frame_count"], metadata["frame_count"], "视频已就绪")
        return metadata
    except VideoMediaError:
        raise
    except Exception as exc:
        raise VideoMediaError(f"视频处理失败：{exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def frame_image(source: Path, cache: Path, metadata: dict, frame_index: int,
                cancel: Cancel | None = None) -> Path:
    """Return a cached original-resolution JPEG for the exact decoded frame.

    Seeking starts at an earlier keyframe; the frame is selected by its original
    PTS. Source stat changes invalidate extraction, avoiding stale frame identity.
    """
    av = _av()
    source, cache = Path(source).resolve(), Path(cache).resolve()
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 0 <= frame_index < metadata["frame_count"]:
        raise VideoMediaError("帧序号超出视频范围。")
    if not source.is_file() or _signature(source) != metadata.get("source_signature"):
        raise VideoMediaError("视频源文件已移动或修改，请重新导入。")
    _check(cancel)
    folder = cache / "frames" / metadata["sha256"][:16]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{frame_index:08d}.jpg"
    if target.is_file():
        return target
    source_tb = Fraction(*metadata["time_base"])
    desired_time = metadata["source_pts"][frame_index] * source_tb
    image = None
    # Some formats do not support seeking. A complete decode still gives exact
    # results and is practical for the short clips targeted by the first release.
    for seek in (True, False):
        try:
            with av.open(str(source), "r") as container:
                stream = container.streams.video[0]
                if seek:
                    container.seek(int(desired_time / stream.time_base), stream=stream, any_frame=False, backward=True)
                for frame in container.decode(stream):
                    _check(cancel)
                    time = frame.pts * frame.time_base if frame.pts is not None else None
                    if time == desired_time:
                        image = _image(frame, metadata["rotation"])
                        break
                    if time is not None and time > desired_time:
                        break
            if image is not None:
                break
        except VideoCancelledError:
            raise
        except Exception:
            if not seek:
                raise
    if image is None:
        raise VideoMediaError("未能解码指定帧；源视频可能已损坏。")
    _check(cancel)
    fd, temp_name = tempfile.mkstemp(prefix=".frame-", suffix=".jpg", dir=folder)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        image.save(temp_path, format="JPEG", quality=95, subsampling=0)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target
