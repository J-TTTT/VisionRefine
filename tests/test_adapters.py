import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from visionrefine.core.adapters import normalize_base_url, test_openai_compatible as check_adapter


class ModelHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps({"data": [{"id": "test-vlm"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def test_normalize_base_url():
    assert normalize_base_url("http://localhost:8000/v1/") == "http://localhost:8000/v1"
    with pytest.raises(ValueError):
        normalize_base_url("not-a-url")


def test_openai_compatible_model_discovery(monkeypatch):
    # Local fixture traffic must not go through a user's HTTP proxy.
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = check_adapter(f"http://127.0.0.1:{server.server_port}/v1")
        assert result.ok
        assert result.models == ["test-vlm"]
    finally:
        server.shutdown()
        server.server_close()
