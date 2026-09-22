"""Copy into visionrefine/core/dataset_io/your_format.py and implement both ends.

Register Format(..., version=..., geometries=("bbox",), attributes=(...),
input={...}, status="experimental", limitations=(...)) in __init__.py.
See format-center.md for the complete contract and acceptance checklist.
"""
from visionrefine.core.dataset_io.models import Dataset


class YourDetectionAdapter:
    def read(self, root, source, labels, split) -> Dataset:
        # Resolve files with contained_path; never write to root/source or fetch URLs.
        # Build pixel xyxy boxes, canonical classes, coarse (NOT human) revisions.
        # Report every skipped image/object and unsupported field in dataset.report.
        # Return finish(dataset) for complete validation and recomputed counts.
        raise NotImplementedError("Implement the format-specific reader")

    def preflight(self, dataset, include_images):
        # Optional extra checks beyond registry capabilities. Return issue(...) rows;
        # use loss/conversion for explicit consent and blocking for invalid output.
        return []

    def write(self, dataset, output):
        # dataset is an immutable-in-spirit, sanitized snapshot; output is fresh.
        # Write annotation/config files only. Packaging copies images and writes
        # portable visionrefine.json + export-report.json, fingerprints and ZIPs.
        # Return {"image_paths": {logical_path: relative_output_path},
        #         "category_mapping": [...], "warnings": [...]}; optional
        # image_encoding="jpeg" requests real conversion, not extension renaming.
        raise NotImplementedError("Implement the format-specific writer")
