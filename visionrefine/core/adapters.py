from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class AdapterCheck:
    ok: bool
    message: str
    models: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Endpoint must be a valid HTTP or HTTPS URL")
    return url


def test_openai_compatible(
    base_url: str,
    *,
    api_key_env: str | None = None,
    timeout: float = 8.0,
) -> AdapterCheck:
    """Check an OpenAI-compatible `/models` endpoint without storing secrets."""
    base_url = normalize_base_url(base_url)
    url = base_url + "/models"
    headers = {"Accept": "application/json"}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            return AdapterCheck(False, f"Environment variable {api_key_env} is not set.", [])
        headers["Authorization"] = f"Bearer {key}"

    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return AdapterCheck(False, f"Model service returned HTTP {exc.code}.", [])
    except (URLError, TimeoutError) as exc:
        return AdapterCheck(False, f"Cannot reach model service: {exc.reason if isinstance(exc, URLError) else exc}", [])
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        return AdapterCheck(False, f"Invalid model service response: {exc}", [])

    models = [str(row.get("id")) for row in payload.get("data", []) if row.get("id")]
    if not models:
        return AdapterCheck(False, "Connected, but the service returned no model IDs.", [])
    return AdapterCheck(True, f"Connected. Found {len(models)} model(s).", models)


def create_chat_completion(
    adapter: dict,
    messages: list[dict],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 90.0,
) -> dict:
    base_url = normalize_base_url(adapter["base_url"])
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    api_key_env = adapter.get("api_key_env")
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set.")
        headers["Authorization"] = f"Bearer {key}"
    payload = json.dumps({
        "model": adapter["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    try:
        with urlopen(
            Request(base_url + "/chat/completions", data=payload, headers=headers),
            timeout=timeout,
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Model service returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        raise RuntimeError(f"Cannot reach model service: {reason}") from exc

