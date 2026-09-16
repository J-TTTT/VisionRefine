from pathlib import Path

from visionrefine.core.store import ProjectStore


def test_delete_moves_only_project_metadata_to_trash(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source_image = dataset / "keep.jpg"
    source_image.write_bytes(b"not-an-image")

    store = ProjectStore(tmp_path / "workspace" / "projects")
    project = store.create({
        "name": "Delete test",
        "task": "detection",
        "dataset_path": str(dataset),
        "model_max_side": 1536,
    })
    assert project["labels"] == ["person"]
    destination = store.delete(project["id"])

    assert destination.parent == tmp_path / "workspace" / "trash"
    assert (destination / "project.json").exists()
    assert source_image.exists()
    assert store.list() == []
