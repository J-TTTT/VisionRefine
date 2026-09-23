"""Create an editable synthetic segmentation demo without replacing prior edits.

Run from the repository: .venv/bin/python -m scripts.create_segmentation_demo
"""
from pathlib import Path
import json
import uuid

from PIL import Image, ImageDraw

from visionrefine import server


def main():
    for project in server.store.list():
        if project.get("demo_key") == "instance-segmentation-v1":
            print(json.dumps({"project_id": project["id"], "name": project["name"]}, ensure_ascii=False))
            return
    root = server.WORKSPACE_ROOT / "cache" / f"segmentation-demo-{uuid.uuid4().hex[:8]}" / "images"
    root.mkdir(parents=True)
    image = Image.new("RGB", (1200, 800), "#edf2ef")
    draw = ImageDraw.Draw(image)
    for x in range(0, 1200, 50):
        draw.line([(x, 0), (x, 800)], fill="#dce5df")
    for y in range(0, 800, 50):
        draw.line([(0, y), (1200, y)], fill="#dce5df")
    parts = [
        [[110, 170], [340, 170], [410, 300], [300, 440], [110, 390]],
        [[430, 180], [515, 180], [515, 265], [430, 265]],
    ]
    for part in parts:
        draw.polygon([tuple(p) for p in part], fill="#d7a96a", outline="#926329", width=3)
    draw.ellipse((690, 170, 1010, 490), fill="#94bcd1", outline="#46798d", width=3)
    draw.ellipse((785, 265, 915, 395), fill="#edf2ef", outline="#46798d", width=2)
    draw.text((100, 70), "POLYGON: move vertices / add a disconnected part", fill="#284638")
    draw.text((690, 70), "MASK: brush / erase / undo", fill="#284638")
    draw.text((100, 580), "Save, reload, and compare. These are synthetic practice shapes.", fill="#284638")
    image.save(root / "01-polygon-and-mask.png")
    large = Image.new("RGB", (4096, 3072), "#f1f3eb")
    large_draw = ImageDraw.Draw(large)
    for x in range(0, 4096, 256):
        large_draw.line([(x, 0), (x, 3072)], fill="#d1ded1", width=2)
    for y in range(0, 3072, 256):
        large_draw.line([(0, y), (4096, y)], fill="#d1ded1", width=2)
    large_draw.polygon([(1600, 1000), (2350, 1100), (2200, 1700), (1800, 1800)], fill="#c69f69")
    large_draw.ellipse((2700, 1800, 3300, 2400), fill="#8cb5c8")
    large.save(root / "02-large-image-practice.png")
    p = server.create_project(server.ProjectInput(name="[体验] 实例分割：多边形与画笔", task="instance_segmentation",
        dataset_path=str(root), labels=["component", "ring"], model_max_side=1536))
    server.analyze_project(p["id"])
    # Sparse mask tiles constructed from pixel-center circle membership.
    tiles = []
    for y in range(128, 512, 128):
        for x in range(640, 1024, 128):
            counts, bit, count, area = [], 0, 0, 0
            for py in range(128):
                for px in range(128):
                    distance = (x + px + .5 - 850) ** 2 + (y + py + .5 - 330) ** 2
                    value = int(65 ** 2 <= distance <= 160 ** 2)
                    area += value
                    if value == bit:
                        count += 1
                    else:
                        counts.append(count)
                        bit, count = value, 1
            counts.append(count)
            if area:
                tiles.append(dict(x=x, y=y, counts=counts))
    server.save_annotations(p["id"], server.AnnotationInput(image="01-polygon-and-mask.png", objects=[
        dict(id="demo-component", kind="polygon", label="component", polygons=parts),
        dict(id="demo-ring", kind="mask", label="ring", mask=dict(tiles=tiles)),
    ]))
    p = server.store.get(p["id"])
    p["demo_key"] = "instance-segmentation-v1"
    server.store.save(p)
    print(json.dumps({"project_id": p["id"], "name": p["name"], "images": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
