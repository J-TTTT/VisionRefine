"""Exercise media fidelity with real, small video fixtures (optional PyAV extra)."""
from fractions import Fraction
from pathlib import Path
import os
import struct

import pytest
from PIL import Image

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from visionrefine.core.video.media import (
    VideoCancelledError, VideoMediaError, capabilities, frame_image, prepare_video,
)


def make_video(path: Path, times=(0, 40, 160, 200), audio=False, rotation=0):
    """Create VFR H.264 with distinguishable frames and optionally offset audio."""
    with av.open(str(path), "w", format="mp4") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height = 96, 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.max_b_frames = 0
        sound = output.add_stream("aac", rate=48_000) if audio else None
        if sound:
            sound.layout = "stereo"
        for index, timestamp in enumerate(times):
            data = np.full((64, 96, 3), 35 + 45 * index, dtype=np.uint8)
            # An asymmetric red region verifies orientation across extraction.
            data[:16, :24] = [240, 10, 10]
            frame = av.VideoFrame.from_ndarray(data, format="rgb24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        if sound:
            # Starts 80 ms after the first video frame; alignment must survive.
            start = (times[0] + 80) * 48
            end = (times[-1] + 40) * 48
            for timestamp in range(start, end, 480):
                values = (0.25 * np.sin(2 * np.pi * 440 * np.arange(480) / 48_000)).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(np.stack([values, values]), format="fltp", layout="stereo")
                frame.pts, frame.time_base, frame.sample_rate = timestamp, Fraction(1, 48_000), 48_000
                for packet in sound.encode(frame):
                    output.mux(packet)
            for packet in sound.encode():
                output.mux(packet)
    if rotation:
        data = bytearray(path.read_bytes())
        header = data.index(b"tkhd") + 4
        assert data[header] == 0
        assert rotation == 90
        struct.pack_into(">9i", data, header + 40, 0, -65536, 0, 65536, 0, 0, 0, 0, 1073741824)
        path.write_bytes(data)
    return path


def test_vfr_pts_origin_exact_extraction_and_preview(tmp_path):
    source = make_video(tmp_path / "source.mp4", times=(2000, 2040, 2160, 2200))
    cache = tmp_path / "cache"
    metadata = prepare_video(source, cache)
    assert capabilities()["available"]
    assert metadata["timestamps"] == pytest.approx([0, .04, .16, .20])
    assert metadata["duration"] == pytest.approx(.24)
    assert not metadata["duration_estimated"]
    assert metadata["source_start_time"] == pytest.approx(2)
    assert metadata["frame_count"] == 4
    # Average-FPS arithmetic would choose the wrong time for frame 2.
    assert abs(2 / metadata["fps"] - metadata["timestamps"][2]) > .02
    image = Image.open(frame_image(source, cache, metadata, 2))
    assert image.size == (96, 64)
    assert image.getpixel((70, 40))[0] == pytest.approx(125, abs=6)
    with av.open(metadata["preview_path"]) as preview:
        frames = list(preview.decode(video=0))
        assert [frame.time for frame in frames] == pytest.approx(metadata["timestamps"], abs=.000002)
        assert float(preview.streams.video[0].duration * preview.streams.video[0].time_base) == pytest.approx(.24, abs=.000002)
    assert len(metadata["thumbnails"]) == 4
    # Reusing a valid cache does not transcode again.
    before = Path(metadata["preview_path"]).stat().st_mtime_ns
    assert prepare_video(source, cache) == metadata
    assert Path(metadata["preview_path"]).stat().st_mtime_ns == before


def test_rotation_matches_preview_original_and_poster(tmp_path):
    source = make_video(tmp_path / "rotated.mp4", rotation=90)
    metadata = prepare_video(source, tmp_path / "cache")
    assert metadata["rotation"] == 90
    assert (metadata["width"], metadata["height"]) == (64, 96)
    original = Image.open(frame_image(source, tmp_path / "cache", metadata, 0))
    assert original.size == (64, 96)
    # The encoded top-left red block rotates into the bottom-left corner.
    assert original.getpixel((6, 88))[0] > 200
    assert original.getpixel((6, 88))[1] < 40
    with av.open(metadata["preview_path"]) as preview:
        frame = next(preview.decode(video=0))
        assert (frame.width, frame.height) == (64, 96)
        assert frame.rotation == 0
        assert frame.to_image().getpixel((6, 88))[0] > 200


def test_audio_preserves_relative_offset_and_is_aac(tmp_path):
    source = make_video(tmp_path / "audio.mp4", times=(2000, 2040, 2160, 2200), audio=True)
    metadata = prepare_video(source, tmp_path / "cache")
    assert metadata["has_audio"] is True
    with av.open(str(source)) as original:
        first_source_audio = next(original.decode(audio=0))
        expected = first_source_audio.time - metadata["source_start_time"]
    with av.open(metadata["preview_path"]) as preview:
        assert len(preview.streams.audio) == 1
        assert preview.streams.audio[0].codec_context.name == "aac"
        decoded = list(preview.decode(audio=0))
        # One AAC encoder priming frame may precede the sound; never reset the
        # track to zero and lose the original positive offset.
        assert decoded[0].time == pytest.approx(expected, abs=.023)
        assert decoded[0].time > .01
        assert sum(frame.samples for frame in decoded) > 1000


def test_cancellation_discards_incomplete_cache(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    messages = []
    def progress(item):
        messages.append(item["message"])
    def cancel():
        return "建立精确帧索引" in messages
    with pytest.raises(VideoCancelledError):
        prepare_video(source, tmp_path / "cache", progress=progress, cancel=cancel)
    assert not (tmp_path / "cache" / "index.json").exists()
    assert not list((tmp_path / "cache").glob(".prepare-*"))


def test_invalid_indices_and_source_mutation_are_rejected(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    metadata = prepare_video(source, tmp_path / "cache")
    for index in (-1, 4, True, 1.1):
        with pytest.raises(VideoMediaError, match="帧序号"):
            frame_image(source, tmp_path / "cache", metadata, index)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    with pytest.raises(VideoMediaError, match="修改"):
        frame_image(source, tmp_path / "cache", metadata, 0)


def test_no_video_track_is_actionable(tmp_path):
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"not a video")
    with pytest.raises(VideoMediaError, match="视频处理失败"):
        prepare_video(source, tmp_path / "cache")


def test_webm_fallback_preserves_vfr_rotation_and_audio_offset(tmp_path):
    source = make_video(tmp_path / "portable.mp4", times=(2000, 2040, 2160, 2200), audio=True, rotation=90)
    metadata = prepare_video(source, tmp_path / "cache")
    with av.open(metadata["preview_webm_path"]) as preview:
        assert preview.streams.video[0].codec_context.name == "vp8"
        assert preview.streams.audio[0].codec_context.name == "opus"
        frames = list(preview.decode(video=0))
        assert [frame.time for frame in frames] == pytest.approx(metadata["timestamps"], abs=.001)
        assert (frames[0].width, frames[0].height) == (64, 96)
        assert frames[0].rotation == 0
        assert frames[0].to_image().getpixel((6, 88))[0] > 200
    with av.open(str(source)) as original:
        expected_start = next(original.decode(audio=0)).time - metadata["source_start_time"]
    with av.open(metadata["preview_webm_path"]) as preview:
        frames = list(preview.decode(audio=0))
        # WebM uses milliseconds; Opus pre-skip can add a rounding quantum.
        assert frames[0].time == pytest.approx(expected_start, abs=.00101)
        assert frames[0].time > .01
        assert sum(frame.samples for frame in frames) > 1000


def test_missing_webm_cache_is_rebuilt(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    cache = tmp_path / "cache"
    metadata = prepare_video(source, cache)
    assert Path(metadata["preview_webm_path"]).is_file()
    Path(metadata["preview_webm_path"]).unlink()
    restored = prepare_video(source, cache)
    assert Path(restored["preview_webm_path"]).is_file()
    assert restored["timestamps"] == metadata["timestamps"]
    assert restored["sha256"] == metadata["sha256"]


def test_cancel_during_webm_discards_both_partial_previews(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    seen_webm = False
    def progress(item):
        nonlocal seen_webm
        if item["message"] == "生成 WebM 兼容预览":
            seen_webm = True
    with pytest.raises(VideoCancelledError):
        prepare_video(source, tmp_path / "cache", progress=progress, cancel=lambda: seen_webm)
    assert seen_webm
    assert not (tmp_path / "cache" / "index.json").exists()
    assert not (tmp_path / "cache" / "preview.mp4").exists()
    assert not (tmp_path / "cache" / "preview.webm").exists()
    assert not list((tmp_path / "cache").glob(".prepare-*"))
