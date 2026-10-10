"""Run actual default models through the application API in isolated demo projects.

Requires the web server on --base-url and the worker on its configured address.
This creates NEW projects, leaves all suggestions pending and writes a JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request

from scripts.create_video_segmentation_demo import build_demo

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8022")
    parser.add_argument("--workspace", type=Path, default=ROOT / "workspace")
    parser.add_argument("--capabilities", nargs="+", choices=["video_segmentation", "image_segmentation", "image_detection", "video_caption", "video_events"],
                        help="Run only the selected capabilities; completed results are reused")
    parser.add_argument("--rerun", action="store_true", help="Repeat selected capabilities and retain previous results in the report")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = workspace / "cache/ai/validation"
    output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path, payload=None, method="POST"):
        request = urllib.request.Request(args.base_url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else method)
        with opener.open(request, timeout=30) as response:
            return json.load(response)

    manifest_path = output / "demo.json"
    if manifest_path.exists():
        demo = json.loads(manifest_path.read_text())
    else:
        demo = build_demo(workspace, output / "source")
        from visionrefine.core.store import ProjectStore
        store = ProjectStore(workspace / "projects")
        project = store.get(demo["project_id"])
        project["name"] = "[第三版测试] AI 辅助视频标注"
        store.save(project)
        manifest_path.write_text(json.dumps(demo, ensure_ascii=False, indent=2))
    pid, vid = demo["project_id"], demo["video_id"]
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"video": demo, "runs": []}

    def run(project_id, request):
        print("Starting " + request["capability"], flush=True)
        started = time.monotonic()
        result = api(f"/api/ai/projects/{project_id}/jobs", request)
        jid = result["id"]
        deadline = time.monotonic() + 900
        previous = None
        while time.monotonic() < deadline:
            job = api(f"/api/ai/projects/{project_id}/jobs/{jid}")
            status = (job["status"], job["progress"]["message"])
            if status != previous:
                print(request["capability"] + ": " + str(status), flush=True)
                previous = status
            if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                break
            time.sleep(1)
        else:
            api(f"/api/ai/projects/{project_id}/jobs/{jid}/cancel", {})
            raise RuntimeError("Smoke job exceeded 15 minutes")
        row = {"capability": request["capability"], "project_id": project_id, "job_id": jid,
               "status": job["status"], "elapsed_seconds": round(time.monotonic() - started, 2),
               "model": job["provider"]["model"], "items": len(job["items"]), "error": job["error"]}
        if request["capability"] in {"video_caption", "video_events"}:
            row["values"] = [item["value"] for item in job["items"]]
        report["runs"].append(row)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if job["status"] != "completed":
            raise RuntimeError(job["error"])
        return job

    completed = set() if args.rerun else {row["capability"] for row in report["runs"] if row["status"] == "completed"}
    def pending(cap):
        return cap not in completed and (args.capabilities is None or cap in args.capabilities)
    if pending("video_segmentation"):
        run(pid, {"capability": "video_segmentation", "video_id": vid, "track_id": "demo-orange-object",
                  "start_frame": 0, "end_frame": 47, "seed_frame": 0})
    image_root = output / "images"
    image_root.mkdir(exist_ok=True)
    frame_url = args.base_url + f"/api/video/projects/{pid}/videos/{vid}/frames/0"
    with opener.open(frame_url) as response:
        (image_root / "frame.jpg").write_bytes(response.read())
    image_projects = report.setdefault("image_projects", {})
    for cap, task in [("image_segmentation", "instance_segmentation"), ("image_detection", "detection")]:
        if not pending(cap):
            continue
        if cap not in image_projects:
            project = api("/api/projects", {"name": "[第三版测试] " + cap, "task": task,
                          "dataset_path": str(image_root), "labels": ["orange rectangle"], "model_max_side": 1536})
            image_projects[cap] = project["id"]
            api(f"/api/projects/{project['id']}/analyze", {})
        request = {"capability": cap, "image": "frame.jpg"}
        if cap == "image_segmentation":
            request.update(label="orange rectangle", prompt={"points": [[30, 120]], "point_labels": [1], "box": [16, 90, 52, 150]})
        run(image_projects[cap], request)
    for cap in ("video_caption", "video_events"):
        if pending(cap):
            run(pid, {"capability": cap, "video_id": vid, "sample_count": 12,
                      "instruction": "描述可见的形状、颜色和运动。只描述橙色矩形与遮挡物的关系。"})
    print("Selected capabilities completed. Suggestions remain pending in the demo projects.", flush=True)


if __name__ == "__main__":
    main()
