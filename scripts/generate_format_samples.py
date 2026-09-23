"""Generate synthetic, reviewed packages for every registered detection exporter.

Run from repository root: python -m scripts.generate_format_samples /tmp/new-samples
The destination must not exist. No real datasets are read or changed.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

from visionrefine.core.dataset_io import registry
from visionrefine.core.dataset_io.common import dump_json
from visionrefine.core.dataset_io.service import package_snapshot
from visionrefine.core.dataset_io.testing import detection_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=False)
    dataset = detection_fixture(root / "input")
    project = dict(id="samples", task="detection", dataset_path=str(root / "input"))
    store = SimpleNamespace(root=root / "projects")
    packages = {}
    for spec in registry.capabilities():
        if not spec["can_export"] or "detection" not in spec["tasks"]:
            continue
        name = spec["id"]
        report = package_snapshot(store, project, dataset.model_copy(deep=True), name,
                                  "reviewed", True, dict(skipped_images=[], revisions=[]))
        folder = Path("projects/samples/exports") / report["export_id"]
        packages[name] = dict(directory=folder.as_posix(), archive=folder.with_suffix(".zip").as_posix(),
                              image_paths=report["image_paths"], version=spec["version"])
    dump_json(root / "index.json", packages)
    print(root / "index.json")


if __name__ == "__main__":
    main()
