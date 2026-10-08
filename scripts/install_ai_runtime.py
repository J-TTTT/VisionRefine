"""Install pinned CUDA wheels from official sources without pip's single stream bottleneck."""
import email
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import urllib.request

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

from download_ai_file import download

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "workspace/cache/ai"
WHEELS = CACHE / "wheels"


def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def install():
    from importlib.metadata import version, PackageNotFoundError
    try:
        if version("torch") == "2.8.0+cu126" and version("torchvision") == "0.23.0+cu126":
            print("Pinned GPU runtime is already installed", flush=True)
            return
    except PackageNotFoundError:
        pass
    supported = set(sys_tags())
    selected = []
    small = []
    for name, version in (("torch", "2.8.0+cu126"), ("torchvision", "0.23.0+cu126")):
        listing = fetch(f"https://download.pytorch.org/whl/cu126/{name}/").decode()
        found = []
        for link in re.findall(r'href="([^"]+)"', listing):
            url = link.split("#")[0]
            filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
            if not filename.endswith(".whl"):
                continue
            pkg, ver, _, tags = parse_wheel_filename(filename)
            if str(ver) == version and supported.intersection(tags):
                found.append((url, filename, link.split("sha256=")[-1] if "sha256=" in link else None))
        if not found:
            raise RuntimeError(f"No compatible official wheel: {name} {version}")
        url, filename, sha = found[0]
        url = url.replace("download-r2.pytorch.org", "download.pytorch.org")
        metadata = email.message_from_bytes(fetch(url.replace("download-r2.pytorch.org", "download.pytorch.org") + ".metadata"))
        for line in metadata.get_all("Requires-Dist", []):
            req = Requirement(line)
            if req.marker and not req.marker.evaluate():
                continue
            if req.name in {"torch", "torchvision"}:
                continue
            if req.name.startswith("nvidia-") or req.name == "triton":
                spec = str(req.specifier)
                if not spec.startswith("=="):
                    raise ValueError("Expected pinned CUDA dependency")
                package = json.loads(fetch(f"https://pypi.org/pypi/{req.name}/{spec[2:]}/json"))
                for item in package["urls"]:
                    if item["filename"].endswith(".whl") and supported.intersection(parse_wheel_filename(item["filename"])[3]):
                        selected.append((item["url"], item["filename"], item["digests"]["sha256"]))
                        break
                else:
                    raise RuntimeError(f"No compatible wheel: {req.name}")
            else:
                small.append(str(req))
        selected.append((url, filename, sha))
    selected = list({row[1]: row for row in selected}.values())
    for url, filename, sha in selected:
        download(url, WHEELS / filename, sha, workers=16)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *small], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", *[str(WHEELS / row[1]) for row in selected]], check=True)
    for _, filename, _ in selected:
        (WHEELS / filename).unlink()


if __name__ == "__main__":
    install()
