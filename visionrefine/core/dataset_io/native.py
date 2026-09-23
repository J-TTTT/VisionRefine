"""Portable dataset snapshot, not a project backup or an arbitrary file loader."""
import json

from .common import dump_json, finish, read_image
from .models import Dataset, contained_path
from .portable import file_digest, portable_dataset


class NativeDataset:
    def read(self, root, source, labels, split, *, trust_reviewed=False):
        source = source or root / "dataset.json"
        package = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(package, dict) or package.get("format") != "visionrefine.dataset" or package.get("version") != "1.0" or "dataset" not in package:
            raise ValueError("Expected VisionRefine dataset package version 1.0")
        dataset = Dataset.model_validate(package["dataset"])
        if dataset.task != "detection":
            raise ValueError("Native package v1 currently supports detection snapshots only")
        # Bind every file against the selected local root. Never follow embedded source roots.
        for record in dataset.images:
            path = contained_path(root, record.path)
            actual = read_image(root, record.path, record.split, dataset.report)
            if actual is None or (actual.width, actual.height) != (record.width, record.height):
                raise ValueError(f"Native image missing or dimensions changed: {record.path}")
            if record.sha256 and file_digest(path) != record.sha256:
                raise ValueError(f"Native image checksum differs: {record.path}")
            record.source_id = record.source_path = None
            record.provenance["original_review_status"] = record.status
            if not trust_reviewed and record.status != "unreviewed":
                record.status = "imported_coarse"
                for obj in record.objects:
                    obj.provenance["original_review_status"] = obj.source
                    obj.source = "imported_coarse"
        dataset.sources = []
        dataset.provenance["review_status_trusted"] = trust_reviewed
        return finish(dataset)

    def write(self, dataset, output):
        paths = {i.path: f"images/{i.path}" for i in dataset.images}
        dump_json(output / "dataset.json", dict(format="visionrefine.dataset", version="1.0",
                                               dataset=portable_dataset(dataset, paths).model_dump()))
        return dict(image_paths=paths, category_mapping=[c.model_dump(include={"id", "name"}) for c in dataset.categories], warnings=[])
