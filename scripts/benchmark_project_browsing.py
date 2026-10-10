"""Benchmark project browsing with synthetic records; never opens real datasets.

Run: .venv/bin/python scripts/benchmark_project_browsing.py --images 10000 50000
Each logical image references a single generated 128px PNG. This measures record
scaling, not cold filesystem scans or decoding large source images.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import visionrefine.server as server
from visionrefine.core.dataset_io.models import Annotation, Category, Dataset, DatasetSource, ImageRecord
from visionrefine.core.dataset_io.service import persist_import
from visionrefine.core.router import plan_image
from visionrefine.core.store import ProjectStore


def synthetic_project(store, root: Path, count: int, *, task="instance_segmentation"):
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), "#a0bdac").save(root / "sample.png")
    project = store.create(dict(name=f"Synthetic {count}", task=task, labels=["cell"], dataset_path=str(root)))
    shape = dict(kind="polygon", polygons=[[[10, 10], [50, 10], [50, 50], [10, 50]]]) if task == "instance_segmentation" else dict(bbox=[10, 10, 50, 50])
    obj = Annotation(id="object-0", label="cell", category_id=0, **shape)
    records = [ImageRecord(id=f"synthetic/{i:06}.png", path=f"synthetic/{i:06}.png", width=128, height=128,
                           source_id="synthetic", source_path="sample.png", split="val" if i % 2 else "train",
                           status="imported_coarse", objects=[obj]) for i in range(count)]
    dataset = Dataset(task=task, categories=[Category(id=0, name="cell")], images=records,
                      sources=[DatasetSource(id="synthetic", root=str(root), format="images")],
                      provenance={"format": "mixed"})
    dataset.report.image_count = count
    dataset.report.object_count = count
    persist_import(store, project, dataset, hash_images=False)
    route = plan_image(128, 128).to_dict()
    project["analysis"] = dict(image_count=count, unreadable_count=0, has_coarse_annotations=True,
        average_megapixels=0.02, max_width=128, max_height=128, routes={"direct": count}, truncated=False, errors=[],
        images=[dict(path=r.path, split=r.split, width=128, height=128, megapixels=0.02, route=route) for r in records])
    store.save(project)
    return project


def benchmark(count: int, repeats: int):
    with tempfile.TemporaryDirectory(prefix="visionrefine-browse-") as folder:
        root = Path(folder)
        previous_store, previous_workspace = server.store, server.WORKSPACE_ROOT
        try:
            server.store = ProjectStore(root / "workspace/projects")
            server.WORKSPACE_ROOT = root / "workspace"
            start = time.perf_counter()
            project = synthetic_project(server.store, root / "images", count)
            setup_seconds = time.perf_counter() - start
            # Fresh store: normal browsing must not depend on a warm Python cache.
            server.store = ProjectStore(server.store.root)
            base = f"/api/projects/{project['id']}"
            cases = {
                "project_list": ("/api/projects", {}),
                "project_detail": (base, {}),
                "first_page": (base + "/images", {}),
                "last_page": (base + "/images", {"offset": max(0, count - 50)}),
                "search": (base + "/images", {"q": f"{count - 1:06}"}),
                "annotations": (base + "/annotations", {"image": f"synthetic/{count - 1:06}.png"}),
                "thumbnail": (base + f"/thumbnail/synthetic/{count - 1:06}.png", {}),
                "crop": (base + f"/crop/synthetic/{count - 1:06}.png", dict(x=0, y=0, width=64, height=64)),
            }
            result = dict(images=count, synthetic_setup_seconds=round(setup_seconds, 3), requests={})
            with TestClient(server.app) as client:
                for name, (url, params) in cases.items():
                    times = []
                    for _ in range(repeats):
                        start = time.perf_counter()
                        response = client.get(url, params=params)
                        times.append((time.perf_counter() - start) * 1000)
                        response.raise_for_status()
                    result["requests"][name] = dict(first_ms=round(times[0], 2), median_ms=round(statistics.median(times), 2),
                                                    max_ms=round(max(times), 2), response_bytes=len(response.content))
            return result
        finally:
            server.store, server.WORKSPACE_ROOT = previous_store, previous_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, nargs="+", default=[10000, 50000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.images) < 1 or args.repeats < 1:
        parser.error("image counts and repeats must be positive")
    report = dict(python=platform.python_version(), repeats=args.repeats,
                  scope="Synthetic metadata and a shared generated 128px PNG; local TestClient, no real dataset.",
                  runs=[benchmark(count, args.repeats) for count in args.images])
    content = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
