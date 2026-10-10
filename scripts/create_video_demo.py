#!/usr/bin/env python3
"""Create small, original demo clips for the video annotation workbench.

Usage: .venv/bin/python scripts/create_video_demo.py --output /tmp/visionrefine-video-demo
Requires the optional video extra. No downloads or external FFmpeg binary.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import math
import struct

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_clip(path: Path, variable_rate=False, rotate=False, audio=True):
    duration = 8 if not variable_rate else 4
    timestamps = [Fraction(i, 24) for i in range(duration * 24)]
    if variable_rate:
        timestamps = [t for i, t in enumerate(timestamps) if i % 5 not in (2, 3)]
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    with av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"}) as output:
        video = output.add_stream("libx264", rate=24)
        video.width, video.height, video.pix_fmt = 640, 360, "yuv420p"
        video.time_base = Fraction(1, 24_000)
        video.codec_context.time_base = Fraction(1, 24_000)
        video.codec_context.max_b_frames = 0
        video.options = {"preset": "fast", "crf": "18"}
        sound = output.add_stream("aac", rate=48_000) if audio else None
        if sound:
            sound.layout = "stereo"
        for index, timestamp in enumerate(timestamps):
            time = float(timestamp)
            normalized = time / duration
            image = Image.new("RGB", (640, 360), (19, 26, 43))
            draw = ImageDraw.Draw(image)
            draw.text((24, 20), "VisionRefine video annotation demo", font=font, fill=(233, 240, 250))
            draw.text((24, 58), f"frame {index:03d} | time {time:05.2f} s | {'VFR' if variable_rate else '24 FPS'}", font=small, fill=(145, 161, 181))
            draw.line((30, 265, 610, 265), fill=(72, 87, 107), width=2)
            if normalized < .25:
                phase = "ENTER"
                x = -30 + normalized * 1200
            elif normalized < .50:
                phase = "MOVE"
                x = 270 + (normalized - .25) * 500
            elif normalized < .75:
                phase = "PAUSE"
                x = 395
            else:
                phase = "EXIT"
                x = 395 + (normalized - .75) * 1200
            y = 210 - (abs(math.sin(time * 3)) * 22 if phase in ("ENTER", "MOVE", "EXIT") else 0)
            draw.ellipse((x - 32, y - 32, x + 32, y + 32), fill=(67, 205, 189))
            draw.rectangle((470, 197, 520, 263), fill=(236, 174, 62))
            draw.text((24, 290), f"Event: {phase}", font=font, fill=(233, 240, 250))
            draw.line((24, 341, 24 + int(592 * normalized), 341), fill=(67, 205, 189), width=5)
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = int(timestamp * 24_000), Fraction(1, 24_000)
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
        if sound:
            for start in range(0, duration * 48_000, 1024):
                count = min(1024, duration * 48_000 - start)
                times = (start + np.arange(count)) / 48_000
                beep = ((times >= duration / 4) & (times < duration / 4 + .15)) | ((times >= duration * .75) & (times < duration * .75 + .15))
                values = (np.sin(2 * np.pi * 440 * times) * .2 * beep).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(np.stack((values, values)), format="fltp", layout="stereo")
                frame.pts, frame.time_base, frame.sample_rate = start, Fraction(1, 48_000), 48_000
                for packet in sound.encode(frame):
                    output.mux(packet)
            for packet in sound.encode():
                output.mux(packet)
    if rotate:
        data = bytearray(path.read_bytes())
        header = data.index(b"tkhd") + 4
        if data[header] != 0:
            raise RuntimeError("Unexpected MP4 track header version")
        struct.pack_into(">9i", data, header + 40, 0, -65536, 0, 65536, 0, 0, 0, 0, 1073741824)
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/visionrefine-video-demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, settings in (
        ("events-and-audio-8s.mp4", {}),
        ("variable-frame-rate-4s.mp4", {"variable_rate": True, "audio": False}),
        ("rotated-video-4s.mp4", {"variable_rate": True, "rotate": True, "audio": False}),
    ):
        path = args.output / name
        create_clip(path, **settings)
        print(path.resolve())


if __name__ == "__main__":
    main()
