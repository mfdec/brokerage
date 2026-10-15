"""Minimal Claude Messages API client (standard library only) with a tool-use loop.

The transport is injectable so the whole loop is tested against a scripted fake server; only the final network hop is untested
without a real key.  Nothing is sent anywhere except ``api.anthropic.com``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

MODELS = [("Claude Sonnet 5.5 (balanced, recommended)", "claude-sonnet-5-5"),
          ("Claude Opus 5.5 (most thorough)", "claude-opus-5-5"),
          ("Claude Fable 5.1", "claude-fable-5-1"),
          ("Claude Haiku 5.5 (fastest, cheapest)", "claude-haiku-5-5")]

RETRY_STATUS = {429, 500, 502, 503, 529}


class AiError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Cancelled(AiError):
    pass


@dataclass
class AiResult:
    text: str
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    stop_reason: str = ""


Transport = Callable[[str, dict, bytes, float], "tuple[int, dict, bytes]"]


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:                    # noqa: S310 - fixed https URL
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AiError(f"Could not reach api.anthropic.com ({getattr(e, 'reason', e)}). Check the internet connection / firewall.") from None


def _error_text(status: int, body: bytes) -> str:
    try:
        msg = json.loads(body)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        msg = body[:200].decode("utf-8", "replace")
    hint = {401: "The API key was rejected - check it in the console.",
            403: "This key may not be allowed to use that model.",
            404: "Model not found - pick another one in the list.",
            413: "The request was too large.",
            429: "Rate limit or spending limit reached.",
            529: "The API is overloaded; try again in a minute."}.get(status, "")
    return f"Claude API error {status}: {msg} {hint}".strip()


class ClaudeClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5", transport: Transport | None = None, timeout: float = 120.0,
                 max_tokens: int = 4096, sleep: Callable[[float], None] = time.sleep):
        self.api_key, self.model = api_key, model
        self.transport = transport or urllib_transport
        self.timeout, self.max_tokens, self.sleep = timeout, max_tokens, sleep

    # ------------------------------------------------------------------ one request
    def _post(self, payload: dict) -> dict:
        headers = {"x-api-key": self.api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(4):
            status, hdr, raw = self.transport(API_URL, headers, body, self.timeout)
            if status == 200:
                try:
                    return json.loads(raw)
                except ValueError:
                    raise AiError("The API returned something that is not JSON.") from None
            if status in RETRY_STATUS and attempt < 3:
                try:
                    wait = float({k.lower(): v for k, v in hdr.items()}.get("retry-after", ""))
                except ValueError:
                    wait = 2.0 * 2 ** attempt
                self.sleep(min(max(wait, 1.0), 30.0))
                continue
            raise AiError(_error_text(status, raw), status)
        raise AiError("Gave up after repeated API errors.")                           # pragma: no cover

    # ------------------------------------------------------------------ the loop
    def run(self, system: str, user_content: list[dict] | str, tools: list[dict] | None = None,
            handler: Callable[[str, dict], object] | None = None, max_turns: int = 8,
            progress: Callable[[str], None] | None = None, cancelled: Callable[[], bool] | None = None) -> AiResult:
        msgs: list[dict] = [{"role": "user", "content": user_content}]
        res = AiResult("")
        say = progress or (lambda s: None)
        for turn in range(max_turns):
            if cancelled and cancelled():
                raise Cancelled("Cancelled.")
            say("Asking Claude..." if turn == 0 else f"Claude is checking the data (step {turn + 1})...")
            payload = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": msgs}
            if tools:
                payload["tools"] = tools
            r = self._post(payload)
            res.turns += 1
            u = r.get("usage") or {}
            res.input_tokens += int(u.get("input_tokens", 0))
            res.output_tokens += int(u.get("output_tokens", 0))
            content = r.get("content") or []
            res.stop_reason = r.get("stop_reason", "")
            res.text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            if res.stop_reason != "tool_use":
                if res.stop_reason == "refusal":
                    raise AiError("Claude declined to answer this request.")
                return res
            msgs.append({"role": "assistant", "content": content})
            results = []
            for b in content:
                if b.get("type") != "tool_use":
                    continue
                name, args = b.get("name", ""), b.get("input") or {}
                res.tool_calls.append((name, args))
                say(f"Claude asked for: {name}")
                try:
                    if handler is None:
                        raise ValueError("no tools available")
                    out, err = json.dumps(handler(name, args), default=float), False
                except Exception as e:                                               # noqa: BLE001 - reported back to the model
                    out, err = f"error: {e}", True
                results.append({"type": "tool_result", "tool_use_id": b.get("id", ""), "content": out, **({"is_error": True} if err else {})})
            msgs.append({"role": "user", "content": results})
        raise AiError("Claude kept asking for more data; stopped after the step limit.")
