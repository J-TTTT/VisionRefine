import io
import json
import stat
import threading
import time
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import visionrefine.server as server
from visionrefine.core.store import ProjectStore
from visionrefine.core.dataset_io import registry
from visionrefine.core.dataset_io.exports import ExportCenter
from visionrefine.core.dataset_io.models import Dataset, Category, ImageRecord, Annotation
from visionrefine.core.dataset_io.native import NativeDataset
from visionrefine.core.dataset_io.portable import extract_native
from visionrefine.core.dataset_io.tool_formats import CvatDetection, LabelmeDetection, LabelStudioDetection

FORMATS = [f["id"] for f in registry.capabilities() if f["can_export"] and "detection" in f["tasks"]]


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    Image.new("RGB", (100, 80), "white").save(root / "测试.jpg")
    Image.new("RGB", (100, 80), "red").save(root / "empty.png")
    source = root / "input.json"
    source.write_text(json.dumps(dict(images=[dict(id=1,file_name="测试.jpg",width=100,height=80),dict(id=2,file_name="empty.png",width=100,height=80)],
        categories=[dict(id=7,name="cell"),dict(id=9,name="unused")],
        annotations=[dict(id=1,image_id=1,category_id=7,bbox=[10.25,20.5,30.5,40],iscrowd=1)])))
    monkeypatch.setattr(server,"store",ProjectStore(tmp_path / "workspace/projects"))
    monkeypatch.setattr(server,"WORKSPACE_ROOT",tmp_path / "workspace")
    client=TestClient(server.app)
    response=client.post("/api/projects",json=dict(name="formats",task="detection",dataset_path=str(root),annotation_path=str(source),dataset_format="coco_detection",split="val"))
    assert response.status_code==201,response.text
    p=response.json()
    base=f"/api/projects/{p['id']}"
    client.post(base+"/analyze")
    objects=client.get(base+"/annotations",params={"image":"测试.jpg"}).json()["objects"]
    objects[0]["confidence"] = .85
    objects[0]["provenance"].update(api_key="DO-NOT-EXPORT",source_file="/private/secret/image.json")
    for name, rows in (("测试.jpg",objects),("empty.png",[])):
        response=client.put(base+"/annotations",json=dict(image=name,objects=rows))
        assert response.status_code==200,response.text
    return client,p,base,root


def preflight(project, formats, **options):
    client,p,base,root=project
    response=client.post(base+"/dataset/export/preview",json=dict(formats=formats,include_images=True,**options))
    assert response.status_code==200,response.text
    return response.json()


def start(project, preview, ack=True):
    client,p,base,root=project
    acknowledgements={f["format"]:[i["code"] for i in f["issues"] if i["requires_ack"]] for f in preview["formats"]} if ack else {}
    response=client.post(base+"/dataset/export/jobs",json=dict(preview_id=preview["preview_id"],acknowledgements=acknowledgements))
    return response


def wait_job(project, job):
    client,p,base,root=project
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        result=client.get(base+"/dataset/export/jobs/"+job["job_id"]).json()
        if result["status"] not in {"queued","running"}:
            return result
        time.sleep(.01)
    pytest.fail(f"Export job timed out: {result}")


def unpack(project, report, destination):
    client=project[0]
    response=client.get(report["download_url"])
    assert response.status_code==200,response.text
    with ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(destination)
    return destination


def test_capabilities_metadata(project):
    formats=project[0].get("/api/dataset-formats").json()
    assert len(formats)==9
    assert all("version" in f and "status" in f and "input" in f for f in formats)
    assert all(f["geometries"]==["bbox"] for f in formats if f["can_export"] and "detection" in f["tasks"])


def test_preflight_requires_each_conversion_confirmation(project):
    preview=preflight(project,["voc_detection","yolo_detection"])
    assert preview["image_count"]==2 and preview["object_count"]==1
    voc=next(f for f in preview["formats"] if f["format"]=="voc_detection")
    assert {"integer_boxes","jpeg_conversion","attributes_sidecar","confidence_sidecar"} <= {i["code"] for i in voc["issues"]}
    response=start(project,preview,ack=False)
    assert response.status_code==400
    assert project[0].get(project[2]+"/dataset/export/jobs").json()==[]


@pytest.mark.parametrize("format_id",FORMATS)
def test_all_formats_package_and_portable_sidecar(project,tmp_path,format_id):
    preview=preflight(project,[format_id])
    response=start(project,preview)
    assert response.status_code==202,response.text
    job=wait_job(project,response.json())
    assert job["status"]=="completed",job
    report=job["formats"][format_id]["report"]
    assert (report["image_count"],report["object_count"])==(2,1)
    root=unpack(project,report,tmp_path / "out")
    public=(root/"visionrefine.json").read_text()
    assert "DO-NOT-EXPORT" not in public and "/private/" not in public and str(tmp_path) not in public
    dataset=Dataset.model_validate_json(public)
    assert all((root/i.path).is_file() for i in dataset.images)
    if format_id=="cvat_detection":
        restored=CvatDetection().read(root/"images",root/"annotations.xml",[],"val")
    elif format_id=="label_studio_detection":
        assert not (root / "label_config.xml").read_text().startswith("<?xml")
        restored=LabelStudioDetection().read(root,root/"annotations.json",[],"val")
    elif format_id=="labelme_detection":
        restored=LabelmeDetection().read(root,root/"images",[],"val")
    elif format_id=="visionrefine":
        restored=NativeDataset().read(root,None,[],"val")
        assert all(i.status=="imported_coarse" for i in restored.images)
        trusted=NativeDataset().read(root,None,[],"val",trust_reviewed=True)
        assert all(i.status=="human_reviewed" for i in trusted.images)
    else:
        return
    assert len(restored.images)==2 and sum(len(i.objects) for i in restored.images)==1
    obj=next(i.objects[0] for i in restored.images if i.objects)
    assert obj.label=="cell" and obj.bbox==pytest.approx([10.25,20.5,40.75,60.5])


def test_multiformat_fixed_snapshot_and_replay(project,tmp_path):
    client,p,base,root=project
    preview=preflight(project,["coco_detection","labelme_detection","visionrefine"])
    client.put(base+"/annotations",json=dict(image="测试.jpg",objects=[]))
    first=start(project,preview)
    assert first.status_code==202,first.text
    again=start(project,preview)
    assert again.json()["job_id"]==first.json()["job_id"]
    job=wait_job(project,first.json())
    assert job["status"]=="completed",job
    assert all(r["report"]["object_count"]==1 for r in job["formats"].values())
    response=client.get(job["download_url"])
    with ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist())=={"batch-report.json","coco_detection.zip","labelme_detection.zip","visionrefine.zip"}


def test_native_zip_rebind_and_trust(project,tmp_path):
    preview=preflight(project,["visionrefine"])
    job=wait_job(project,start(project,preview).json())
    zip_file=tmp_path/"native.zip"
    zip_file.write_bytes(project[0].get(job["formats"]["visionrefine"]["report"]["download_url"]).content)
    for trusted in (False,True):
        source=dict(id="native",root=str(tmp_path/"unused"),format="visionrefine",annotation_path=str(zip_file),trust_reviewed=trusted)
        response=project[0].post("/api/dataset-imports/preview",json=dict(name="native",sources=[source]))
        assert response.status_code==200,response.text
        result=project[0].post(f"/api/dataset-imports/{response.json()['preview_id']}/commit").json()
        dataset=project[0].get(f"/api/projects/{result['id']}/dataset").json()
        assert {i["status"] for i in dataset["images"]}==({"human_reviewed"} if trusted else {"imported_coarse"})
        assert all("/cache/native-imports/" in s["root"] for s in dataset["sources"])


@pytest.mark.parametrize("change",["missing","modified"])
def test_changed_images_cannot_be_exported(project,change):
    preview=preflight(project,["coco_detection"])
    image=project[3]/"测试.jpg"
    if change=="missing":image.unlink()
    else:Image.new("RGB",(100,80),"blue").save(image)
    job=wait_job(project,start(project,preview).json())
    assert job["status"]=="failed" and not job.get("download_url")
    assert project[0].get(project[2]+f"/dataset/export/jobs/{job['job_id']}/download").status_code==409
    new=preflight(project,["coco_detection"])
    assert new["formats"][0]["blocked"]


def test_partial_failure_preserves_success_and_retry(project,monkeypatch):
    original=CvatDetection.write
    def fail(*args):raise ValueError("deliberate test failure")
    preview=preflight(project,["coco_detection","cvat_detection"])
    monkeypatch.setattr(CvatDetection,"write",fail)
    job=wait_job(project,start(project,preview).json())
    assert job["status"]=="partial",job
    assert job["formats"]["coco_detection"]["status"]=="completed"
    assert job["formats"]["cvat_detection"]["status"]=="failed"
    monkeypatch.setattr(CvatDetection,"write",original)
    retried=project[0].post(project[2]+f"/dataset/export/jobs/{job['job_id']}/retry")
    assert retried.status_code==202
    assert wait_job(project,retried.json())["status"]=="completed"


def test_cancel_and_restart_recovery(project,monkeypatch):
    import visionrefine.core.dataset_io.exports as exports
    entered,release=threading.Event(),threading.Event()
    original=exports.package_snapshot
    def slow(*args,**kwargs):
        entered.set()
        assert release.wait(5)
        kwargs["checkpoint"]()
        return original(*args,**kwargs)
    monkeypatch.setattr(exports,"package_snapshot",slow)
    job=start(project,preflight(project,["coco_detection"])).json()
    try:
        assert entered.wait(5)
        response=project[0].post(project[2]+f"/dataset/export/jobs/{job['job_id']}/cancel")
        assert response.json()["cancel_requested"]
    finally:release.set()
    assert wait_job(project,job)["status"]=="cancelled"
    path=server.store.root/project[1]["id"]/"export-jobs"/f"{job['job_id']}.json"
    data=json.loads(path.read_text());data["status"]="running"
    path.write_text(json.dumps(data))
    restarted=ExportCenter(server.store)
    assert restarted.get(project[1]["id"],job["job_id"])["status"]=="interrupted"


def test_presets_persist_and_no_implicit_ack(project):
    options=dict(formats=["cvat_detection","visionrefine"],include_images=True)
    response=project[0].post(project[2]+"/dataset/export/presets",json=dict(name="常用",options=options))
    assert response.status_code==200,response.text
    assert ExportCenter(server.store).presets(project[1]["id"])[0]["name"]=="常用"
    assert "acknowledgements" not in response.text


@pytest.mark.parametrize("name",["../escape","/absolute","C:/file","a\\file"])
def test_native_zip_rejects_unsafe_paths(tmp_path,name):
    archive=tmp_path/"bad.zip"
    with ZipFile(archive,"w") as z:
        z.writestr("dataset.json","{}")
        z.writestr(name,"bad")
    with pytest.raises(ValueError):extract_native(archive,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_native_zip_rejects_links_duplicates_and_limits(tmp_path):
    for case in ("symlink","duplicate","limit"):
        archive=tmp_path/f"{case}.zip"
        with ZipFile(archive,"w") as z:
            z.writestr("dataset.json","{}")
            if case=="symlink":
                info=ZipInfo("link");info.external_attr=(stat.S_IFLNK|0o777)<<16
                z.writestr(info,"/private")
            elif case=="duplicate":z.writestr("DATASET.JSON","{}")
            else:z.writestr("big","long")
        with pytest.raises(ValueError):extract_native(archive,tmp_path/case,max_bytes=3 if case=="limit" else 10000)


def test_cvat_rejects_video_and_entities(tmp_path):
    source=tmp_path/"cvat.xml"
    for text in ('<!DOCTYPE annotations><annotations/>','<annotations><version>1.1</version><track/></annotations>'):
        source.write_text(text)
        with pytest.raises(ValueError):CvatDetection().read(tmp_path,source,[],"train")


def test_label_studio_multiple_versions_are_not_silently_selected(tmp_path):
    Image.new("RGB",(100,80)).save(tmp_path/"a.jpg")
    source=tmp_path/"tasks.json"
    source.write_text(json.dumps([dict(data={"image":"a.jpg"},annotations=[{"result":[]},{"result":[]}])]))
    with pytest.raises(ValueError):LabelStudioDetection().read(tmp_path,source,["cell"],"train")


def test_labelme_relative_path_fallback_and_bad_shapes(tmp_path):
    folder = tmp_path / "annotations"
    folder.mkdir()
    Image.new("RGB", (100, 80)).save(tmp_path / "a.jpg")
    source = folder / "a.json"
    source.write_text(json.dumps(dict(imagePath="../a.jpg", shapes=[
        dict(label="cell", shape_type="rectangle", points=[[40, 50], [10, 20]]),
        dict(label="cell", shape_type="polygon", points=[[1, 1], [5, 5], [9, 9]]), None])))
    dataset = LabelmeDetection().read(tmp_path, folder, ["person"], "train")
    assert [c.name for c in dataset.categories] == ["cell"]
    assert dataset.images[0].objects[0].bbox == [10, 20, 40, 50]
    assert dataset.report.skipped_objects == 2
    source.write_text(json.dumps(dict(imagePath="../../outside.jpg", shapes=[])))
    with pytest.raises(ValueError):
        LabelmeDetection().read(tmp_path, folder, ["cell"], "train")


def test_label_studio_bad_rows_and_rotations_are_reported(tmp_path):
    Image.new("RGB", (100, 80)).save(tmp_path / "a.jpg")
    source = tmp_path / "tasks.json"
    result = dict(type="rectanglelabels", value=dict(x=10, y=20, width=30, height=40, rotation=0, rectanglelabels=["cell"]))
    rotated = json.loads(json.dumps(result))
    rotated["value"]["rotation"] = 10
    source.write_text(json.dumps([None, dict(data={"image": 4}), dict(data={"image": "a.jpg"},
        annotations=[dict(result=[result, rotated, None])])]))
    dataset = LabelStudioDetection().read(tmp_path, source, ["person"], "train")
    assert [c.name for c in dataset.categories] == ["cell"]
    assert dataset.report.skipped_images == 2 and dataset.report.skipped_objects == 2
    assert dataset.images[0].objects[0].bbox == pytest.approx([10, 16, 40, 48])


def test_metadata_redaction_never_changes_category_names():
    from visionrefine.core.dataset_io.portable import portable_dataset
    dataset = Dataset(categories=[Category(id=0, name="/m/01g317")], images=[ImageRecord(id="a.jpg", path="a.jpg",
        width=20, height=20, objects=[Annotation(id="1", category_id=0, label="/m/01g317", bbox=[1, 2, 3, 4],
        attributes={"api_key": "private", "custom": "/private/path"})])])
    result = portable_dataset(dataset, {"a.jpg": "images/a.jpg"})
    assert result.categories[0].name == result.images[0].objects[0].label == "/m/01g317"
    assert result.images[0].objects[0].attributes == {"custom": "[local-path]"}


def test_trusted_native_append_never_replaces_existing_human(project, tmp_path):
    preview = preflight(project, ["visionrefine"])
    job = wait_job(project, start(project, preview).json())
    archive = tmp_path / "native.zip"
    archive.write_bytes(project[0].get(job["formats"]["visionrefine"]["report"]["download_url"]).content)
    client, p, base, root = project
    assert client.put(base + "/annotations", json=dict(image="测试.jpg", objects=[])).status_code == 200
    response = client.post("/api/dataset-imports/preview", json=dict(project_id=p["id"], duplicate_policy="update_coarse",
        sources=[dict(id="native", root=str(tmp_path), annotation_path=str(archive), format="visionrefine", trust_reviewed=True)]))
    assert response.status_code == 200, response.text
    result = client.post(f"/api/dataset-imports/{response.json()['preview_id']}/commit")
    assert result.status_code == 201, result.text
    annotation = client.get(base + "/annotations", params=dict(image="测试.jpg")).json()
    assert annotation["status"] == "human_reviewed" and annotation["objects"] == []
    history = client.get(base + "/dataset/history").json()
    assert history[0]["operation"]["sources"][0]["trust_reviewed"] is True


def test_invalid_native_archive_returns_client_error(project, tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"not a ZIP")
    response = project[0].post("/api/dataset-imports/preview", json=dict(sources=[
        dict(id="native", root=str(tmp_path), annotation_path=str(archive), format="visionrefine")]))
    assert response.status_code == 400 and "ZIP" in response.text


def test_labelme_invalid_attributes_block_preflight(project):
    client, p, base, root = project
    annotation = client.get(base + "/annotations", params=dict(image="测试.jpg")).json()
    annotation["objects"][0]["attributes"]["flags"] = ["not a dictionary"]
    assert client.put(base + "/annotations", json=dict(image="测试.jpg", objects=annotation["objects"])).status_code == 200
    preview = preflight(project, ["labelme_detection"])
    assert preview["formats"][0]["blocked"]
    assert start(project, preview).status_code == 400


def test_adapter_developer_fixture_and_semantic_helper(tmp_path):
    from types import SimpleNamespace
    from visionrefine.core.dataset_io.service import package_snapshot
    from visionrefine.core.dataset_io.testing import detection_fixture, assert_detection_equivalent
    original = detection_fixture(tmp_path / "input")
    store = SimpleNamespace(root=tmp_path / "projects")
    p = dict(id="helper", task="detection", dataset_path=str(tmp_path / "input"))
    report = package_snapshot(store, p, original.model_copy(deep=True), "labelme_detection", "reviewed", True,
                              dict(skipped_images=[], revisions=[]))
    root = store.root / "helper/exports" / report["export_id"]
    restored = LabelmeDetection().read(root, root / "images", [], "val")
    assert_detection_equivalent(original, restored, report["image_paths"], compare_splits=True)


def test_exif_orientation_and_disguised_encoding_are_checked(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (100, 80), "white").save(root / "rotated.jpg", exif=exif)
    Image.new("RGB", (100, 80), "red").save(root / "disguised.jpg", format="PNG")
    monkeypatch.setattr(server, "store", ProjectStore(tmp_path / "workspace/projects"))
    client = TestClient(server.app)
    response = client.post("/api/projects", json=dict(name="exif", task="detection", dataset_path=str(root), labels=["cell"]))
    assert response.status_code == 201, response.text
    p = response.json()
    base = f"/api/projects/{p['id']}"
    assert client.post(base + "/analyze").status_code == 200
    for name in ("rotated.jpg", "disguised.jpg"):
        assert client.put(base + "/annotations", json=dict(image=name, objects=[])).status_code == 200
    response = client.post(base + "/dataset/export/preview", json=dict(formats=["voc_detection", "visionrefine"]))
    assert response.status_code == 200, response.text
    voc, native = response.json()["formats"]
    assert voc["blocked"] and not native["blocked"]
    assert {"exif_orientation", "jpeg_conversion"} <= {i["code"] for i in voc["issues"]}
