"""Create a reusable 125-image paging demo without replacing user edits.

Run: .venv/bin/python -m scripts.create_browsing_demo
"""
import json
import math
import uuid

from PIL import Image, ImageDraw
from visionrefine import server


def main():
    key = "browsing-demo-v1"
    for project in server.store.list():
        if project.get("demo_key") == key:
            print(json.dumps({"id": project["id"], "name": project["name"]}, ensure_ascii=False))
            return
    root = server.WORKSPACE_ROOT / "cache" / f"browsing-demo-{uuid.uuid4().hex[:8]}"
    images = root / "images"
    images.mkdir(parents=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "component"}, {"id": 2, "name": "cell"}]}
    for i in range(125):
        name = f"sample-{i + 1:03}.png"
        pixels = Image.new("RGB", (960, 640), "#edf2ef")
        draw = ImageDraw.Draw(pixels)
        for x in range(0, 960, 40):
            draw.line([(x, 0), (x, 640)], fill="#dce5df")
        for y in range(0, 640, 40):
            draw.line([(0, y), (960, y)], fill="#dce5df")
        dx, dy = (i % 5) * 12, (i % 7) * 8
        polygon = [[130 + dx, 180 + dy], [300 + dx, 150 + dy], [370 + dx, 300 + dy], [280 + dx, 410 + dy], [100 + dx, 350 + dy]]
        radius = 75 + (i % 6) * 7
        circle = [[680 + radius * math.cos(j * math.tau / 32), 300 + radius * math.sin(j * math.tau / 32)] for j in range(32)]
        for category, points, color in [(1, polygon, "#d4a569"), (2, circle, "#8fbdd0")]:
            draw.polygon([tuple(p) for p in points], fill=color)
            coco["annotations"].append(dict(id=len(coco["annotations"]) + 1, image_id=i + 1, category_id=category,
                segmentation=[[v for point in points for v in point]], iscrowd=0))
        draw.text((40, 40), f"VISIONREFINE  /  SYNTHETIC SAMPLE {i + 1:03} OF 125", fill="#284638", font_size=25)
        draw.text((40, 565), "Try paging, search, vertex editing, brush and human review.", fill="#284638", font_size=22)
        pixels.save(images / name)
        coco["images"].append(dict(id=i + 1, file_name=name, width=960, height=640))
    annotations = root / "instances.json"
    annotations.write_text(json.dumps(coco), encoding="utf-8")
    project = server.create_project(server.ProjectInput(name="[体验] 分页与分割 · 125 张合成图", task="instance_segmentation",
        dataset_path=str(images), annotation_path=str(annotations), dataset_format="coco_segmentation"))
    project = server.analyze_project(project["id"])
    project["demo_key"] = key
    server.store.save(project)
    print(json.dumps({"id": project["id"], "name": project["name"], "image_count": 125}, ensure_ascii=False))


if __name__ == "__main__":
    main()
