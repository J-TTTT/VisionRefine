"""Download a pinned model/package file with resumable HTTP ranges and SHA-256.

Only used by installation scripts; never called on a user-supplied web request.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def download(url, destination, sha256=None, workers=12, *, direct=False):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    def checksum(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    if target.exists() and (not sha256 or checksum(target) == sha256):
        print(f'Already downloaded: {target.name}', flush=True)
        return target
    with opener.open(urllib.request.Request(url, headers={'Range': 'bytes=0-0'}), timeout=60) as response:
        content_range = response.headers.get('Content-Range')
        if not content_range:
            raise ValueError('Download server does not support HTTP ranges')
        size = int(content_range.rsplit('/', 1)[1])
    partial = target.with_name(target.name + '.partial')
    journal = target.with_name(target.name + '.progress.json')
    chunk = 8 * 1024 * 1024
    meta = dict(url=url, size=size, chunk=chunk, sha256=sha256)
    completed = set()
    if journal.exists() and partial.exists():
        old = json.loads(journal.read_text())
        if all(old.get(k) == v for k,v in meta.items()):
            completed = set(old['completed'])
    descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(descriptor, size)
    lock = threading.Lock()
    count = (size + chunk - 1) // chunk
    def part(index):
        if index in completed:
            return
        start, end = index * chunk, min(size, (index + 1) * chunk) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={'Range':f'bytes={start}-{end}'})
                with opener.open(req, timeout=120) as response:
                    if response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                        raise ValueError('Server returned incorrect byte range')
                    block = response.read(end-start+2)
                if len(block) != end-start+1:
                    raise ValueError('Incomplete range')
                written = 0
                while written < len(block):
                    written += os.pwrite(descriptor, block[written:], start+written)
                with lock:
                    completed.add(index)
                    temp = journal.with_suffix('.tmp')
                    temp.write_text(json.dumps({**meta,'completed':sorted(completed)}))
                    temp.replace(journal)
                    if len(completed) % 10 == 0 or len(completed) == count:
                        print(f'{target.name}: {len(completed)}/{count} chunks', flush=True)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(8, attempt + 1))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(part, range(count)))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if sha256 and checksum(partial) != sha256:
        journal.unlink(missing_ok=True)
        raise ValueError(f'SHA-256 mismatch: {target.name}; rerun to download again')
    partial.replace(target)
    journal.unlink(missing_ok=True)
    print(f'Verified: {target.name}', flush=True)
    return target


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url');parser.add_argument('destination');parser.add_argument('--sha256');parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    download(args.url,args.destination,args.sha256,args.workers)
