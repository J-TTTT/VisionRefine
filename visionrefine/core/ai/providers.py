"""Network adapters. Provider errors never silently switch services."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

MAX_RESPONSE = 64 * 1024 * 1024


class Cancelled(Exception):
    pass


def request(provider, path, payload=None, *, timeout=None):
    headers = {"Content-Type": "application/json"}
    variable = provider.get("api_key_env")
    if variable:
        token = os.environ.get(variable)
        if not token:
            raise ValueError(f"Model service key environment variable is not set: {variable}")
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = urllib.request.Request(provider["base_url"] + path, data=data, headers=headers)
    # Local inference must not accidentally traverse a corporate HTTP proxy.
    from urllib.parse import urlsplit
    import ipaddress
    host = urlsplit(provider["base_url"]).hostname
    try:
        local = ipaddress.ip_address(host).is_private
    except ValueError:
        local = host == "localhost"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if local else urllib.request.build_opener()
    try:
        with opener.open(req, timeout=timeout or provider.get("timeout", 600)) as response:
            raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise ValueError("Model response exceeds the 64 MB limit")
        return json.loads(raw)
    except urllib.error.HTTPError as exc:
        # The upstream body may include request content or credentials. Keep it private.
        raise ValueError(f"Model service returned HTTP {exc.code}; check its logs and configuration") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError("Cannot reach the model service or the request timed out; check its address and startup status") from None


def check(provider):
    path = "/models" if provider["protocol"] == "openai_compatible" else "/capabilities"
    result = request(provider, path, timeout=10)
    if provider["protocol"] == "openai_compatible":
        ids = [item.get("id") for item in result.get("data", [])]
        if provider["model"] not in ids:
            raise ValueError("Service is reachable, but the configured model is missing from /models")
        return {"reachable": True, "model": provider["model"], "capabilities": provider["capabilities"]}
    models = result.get("models", [])
    model = next((m for m in models if m.get("id") == provider["model"]), None)
    if not model or not set(provider["capabilities"]).issubset(set(model.get("capabilities", []))):
        raise ValueError("Service does not advertise this model and its configured capabilities")
    return {"reachable": True, "model": provider["model"], "ready": model.get("ready", False),
            "capabilities": model["capabilities"], "detail": model.get("detail", "")}


def instruction(payload):
    cap = payload["capability"]
    shared = ("You annotate only visible evidence. Do not infer audio, hidden objects or unobserved actions. "
              "Return one JSON object, no markdown. Use Chinese descriptions unless the user requests another language. "
              "All coordinates are original image pixels. Times are seconds on the supplied original presentation timeline. ")
    if cap == "image_detection":
        schema = '{"objects":[{"label":"exact supplied label", "bbox":[x1,y1,x2,y2],"confidence":0.8}]}'
    elif cap == "video_caption":
        shared += ("The input is one video in chronological order. Compare the beginning, middle and end. "
                   "Summarize observed movement and changes over the requested interval, not just one frame. ")
        schema = '{"captions":[{"text":"description supported by the frames"}]}'
    else:
        shared += ("The input is one video in chronological order. Identify meaningful actions or changes across frames. "
                   "Merge consecutive observations of the same action into one interval. "
                   "Do not repeat a static scene description for each sampled frame. ")
        schema = '{"events":[{"kind":"interval", "start":0.0,"end":1.0,"label_id":null,"text":"action"}]}'
    return (shared + "Required schema: " + schema + ". Return an empty list if no relevant evidence. "
            + "Use only supplied label IDs (or null for free descriptions). Do not invent track IDs. "
            + json.dumps({k: payload.get(k) for k in ("labels", "start", "end", "width", "height", "instruction")}, ensure_ascii=False))


def parse_json(content):
    if not isinstance(content, str):
        raise ValueError("Model returned non-text content")
    value = content.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(value)
    except (ValueError, IndexError):
        raise ValueError("Model did not return valid JSON; adjust the model or instruction and retry") from None
    if not isinstance(result, dict):
        raise ValueError("Model output must be a JSON object")
    return result


def infer(provider, payload, checkpoint, progress):
    checkpoint()
    if provider["protocol"] == "openai_compatible":
        content = [{"type": "text", "text": instruction(payload)}]
        for frame in payload["frames"]:
            content += [{"type": "text", "text": f'Frame {frame["frame_index"]}, timestamp {frame["timestamp"]:.6f}s'},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame["jpeg"]}}]
        progress("模型正在分析画面", 0, 1)
        result = request(provider, "/chat/completions", {"model": provider["model"], "messages": [{"role": "user", "content": content}],
                                                       "max_tokens": 2048, "temperature": 0})
        checkpoint()
        try:
            return parse_json(result["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            raise ValueError("Unexpected chat completion response") from None
    remote = request(provider, "/jobs", {**payload, "model": provider["model"]}, timeout=30)
    jid = remote.get("id", "")
    import re
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", jid):
        raise ValueError("Worker returned an invalid job ID")
    deadline = time.monotonic() + provider.get("timeout", 600)
    try:
        while True:
            checkpoint()
            result = request(provider, f"/jobs/{jid}", timeout=15)
            if result.get("status") == "completed":
                return result["result"]
            if result.get("status") in {"failed", "cancelled", "interrupted"}:
                raise ValueError("Model worker could not complete the task: " + str(result.get("error") or result["status"])[:1000])
            status = result.get("progress") or {}
            progress(str(status.get("message", "模型正在处理")), status.get("current", 0), status.get("total", 1))
            if time.monotonic() > deadline:
                raise ValueError("Model job timed out; check model worker logs or increase its timeout")
            time.sleep(0.5)
    except Exception:
        try:
            request(provider, f"/jobs/{jid}/cancel", {}, timeout=5)
        except Exception:
            pass
        raise
