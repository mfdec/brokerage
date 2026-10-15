"""AI analyst: key storage, the Messages-API tool loop (scripted fake server), evidence, tools and report parsing."""

import json

import numpy as np
import pytest

from roomscape.ai.analyst import (
    TOOLS, ToolBox, build_evidence, decay_map_summary, parse_report, user_message,
)
from roomscape.ai.client import AiError, Cancelled, ClaudeClient
from roomscape.ai.keystore import KeyStore, _MemoryBackend
from roomscape.core.decaymap import DecayFields
from roomscape.core.fieldmap import FieldMap
from roomscape.core.modes import RoomDims
from roomscape.core.roomfield import RoomConfig

KEY = "sk-ant-api03-" + "x" * 30


class FakeServer:
    """Scripted replies; records every request so the tests can look at what was sent."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, url, headers, body, timeout):
        self.requests.append((url, headers, json.loads(body)))
        r = self.replies.pop(0)
        if isinstance(r, tuple):
            return r
        return 200, {}, json.dumps(r).encode()


def text_reply(text, **kw):
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": {"input_tokens": 100, "output_tokens": 20}, **kw}


def tool_reply(name, args, tid="tu_1"):
    return {"content": [{"type": "text", "text": "looking"}, {"type": "tool_use", "id": tid, "name": name, "input": args}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 50, "output_tokens": 10}}


def fields(dims=RoomDims(5, 4, 2.5)):
    n_t = 31
    edt, t20, edc = (FieldMap(dims, np.array([0.0]), 0.1, 0.22), FieldMap(dims, np.array([0.0]), 0.1, 0.22),
                     FieldMap(dims, np.arange(n_t, dtype=float), 0.1, 0.22))
    f = DecayFields(dims, 63.0, 1 / 3, edt, t20, edc, np.arange(n_t) * 0.05)
    for x in np.arange(0.5, 4.6, 0.4):
        for y in np.arange(0.5, 3.6, 0.4):
            e = 0.9 + 0.5 * (x > 3.5)                       # a slow corner at the back
            edt.add(x, y, np.array([e]), 1.0, 0.3)
            t20.add(x, y, np.array([e * 1.1]), 1.0, 0.3)
            edc.add(x, y, 10 ** (-(np.arange(n_t) * 0.05) * 30 / e / 10), 1.0, 0.3)
            f.n_frames += 1
            f.n_used += 1
    return f


# ------------------------------------------------------------------ key store
def test_keystore_validates_and_roundtrips(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ks = KeyStore(_MemoryBackend())
    assert ks.get() == (None, "")
    for bad in ("", "hello", "sk-ant-short", "sk-ant-api03-" + "x" * 10 + " " + "y" * 10):
        with pytest.raises(ValueError):
            ks.set(bad)
    ks.set("  " + KEY + "\n")
    assert ks.get() == (KEY, "memory (this session only)")
    ks.forget()
    assert ks.get()[0] is None


def test_keystore_env_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    key, src = KeyStore(_MemoryBackend()).get()
    assert key == KEY and "ANTHROPIC_API_KEY" in src


# ------------------------------------------------------------------ client
def test_plain_answer_and_request_shape():
    srv = FakeServer(text_reply("hello"))
    r = ClaudeClient(KEY, "claude-sonnet-5-5", transport=srv).run("sys", "hi")
    assert r.text == "hello" and r.turns == 1 and (r.input_tokens, r.output_tokens) == (100, 20)
    url, hdr, body = srv.requests[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert hdr["x-api-key"] == KEY and hdr["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-sonnet-5-5" and body["system"] == "sys" and body["messages"][0] == {"role": "user", "content": "hi"}
    assert "tools" not in body


def test_tool_loop_passes_results_back():
    srv = FakeServer(tool_reply("add", {"a": 2, "b": 3}), text_reply("5"))
    seen = []
    r = ClaudeClient(KEY, transport=srv).run("s", "q", tools=[{"name": "add"}], handler=lambda n, a: seen.append((n, a)) or {"sum": 5})
    assert r.text == "5" and r.turns == 2 and r.tool_calls == [("add", {"a": 2, "b": 3})] and seen == [("add", {"a": 2, "b": 3})]
    msgs = srv.requests[1][2]["messages"]
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"][1]["type"] == "tool_use"
    tr = msgs[2]["content"][0]
    assert tr == {"type": "tool_result", "tool_use_id": "tu_1", "content": '{"sum": 5}'}


def test_tool_errors_go_back_to_the_model_not_the_user():
    def boom(n, a):
        raise ValueError("no map yet")
    srv = FakeServer(tool_reply("x", {}), text_reply("ok"))
    r = ClaudeClient(KEY, transport=srv).run("s", "q", tools=[{"name": "x"}], handler=boom)
    assert r.text == "ok"
    tr = srv.requests[1][2]["messages"][2]["content"][0]
    assert tr["is_error"] is True and "no map yet" in tr["content"]


def test_step_limit():
    srv = FakeServer(*[tool_reply("x", {}) for _ in range(3)])
    with pytest.raises(AiError, match="step limit"):
        ClaudeClient(KEY, transport=srv).run("s", "q", tools=[{"name": "x"}], handler=lambda n, a: 1, max_turns=3)


@pytest.mark.parametrize("status,needle", [(401, "rejected"), (404, "Model not found"), (400, "bad thing")])
def test_http_errors_are_readable_and_never_leak_the_key(status, needle):
    body = json.dumps({"error": {"type": "x", "message": "bad thing"}}).encode()
    with pytest.raises(AiError) as ei:
        ClaudeClient(KEY, transport=FakeServer((status, {}, body))).run("s", "q")
    assert "bad thing" in str(ei.value) and KEY not in str(ei.value) and ei.value.status == status
    if status != 400:
        assert needle in str(ei.value)


def test_retries_on_overload_then_succeeds():
    waits = []
    srv = FakeServer((529, {"retry-after": "3"}, b"{}"), (429, {}, b"{}"), text_reply("fine"))
    r = ClaudeClient(KEY, transport=srv, sleep=waits.append).run("s", "q")
    assert r.text == "fine" and waits == [3.0, 4.0]


def test_gives_up_after_repeated_overload():
    srv = FakeServer(*[(529, {}, b"{}")] * 4)
    with pytest.raises(AiError, match="overloaded"):
        ClaudeClient(KEY, transport=srv, sleep=lambda s: None).run("s", "q")


def test_refusal_and_cancel():
    with pytest.raises(AiError, match="declined"):
        ClaudeClient(KEY, transport=FakeServer(text_reply("", stop_reason="refusal"))).run("s", "q")
    with pytest.raises(Cancelled):
        ClaudeClient(KEY, transport=FakeServer()).run("s", "q", cancelled=lambda: True)


# ------------------------------------------------------------------ evidence and tools
def test_evidence_is_json_and_compact():
    cfg = RoomConfig()
    ev = build_evidence(cfg, [], fields(), "why is the back corner boomy?")
    text = json.dumps(ev)
    assert len(text) < 12000
    dm = ev["decay_map"]
    assert dm["bursts_used"] > 50 and dm["edt_s"]["max"] > dm["edt_s"]["min"] + 0.3
    worst = dm["edt_longest_spots"][0]
    assert worst["x"] > 3.5 and worst["edt_s"] > 1.3                 # the slow corner is found and named
    assert ev["modes"]["first"][0]["f_hz"] < 40 and ev["user_question"].startswith("why")
    assert "api" not in text.lower()


def test_evidence_without_a_map():
    assert build_evidence(RoomConfig(), [], None)["decay_map"] is None


def test_toolbox_answers_and_refuses():
    cfg = RoomConfig()
    tb = ToolBox(cfg, [], fields_for=lambda f: fields())
    r = tb("decay_at", {"x": 4.2, "y": 2.0})
    assert r["edt_s"] == pytest.approx(1.4, abs=0.05) and r["energy_left_db"]["1s"] < r["energy_left_db"]["0.25s"]
    g = tb("decay_grid", {"cell_m": 1.0})
    assert len(g["edt_s"]) == 4 and len(g["edt_s"][0]) == 5
    m = tb("mode_pressure_at", {"x": 0.0, "y": 0.0, "f_hz": 34.3})
    assert m["modes"] and all(0.0 <= v["abs_psi"] <= 1.0 + 1e-9 for v in m["modes"])
    assert tb("list_modes", {"f_max": 100})["count"] > 3
    with pytest.raises(ValueError, match="unknown tool"):
        tb("rm_rf", {})
    with pytest.raises(ValueError, match="no decay map"):
        ToolBox(cfg)("decay_at", {"x": 1, "y": 1})
    with pytest.raises(ValueError, match="out of range"):
        tb("measurement_bands", {"index": 2})
    assert {t["name"] for t in TOOLS} == {"list_modes", "decay_at", "decay_grid", "mode_pressure_at", "measurement_bands"}


def test_tool_args_are_filtered_to_known_parameters():
    tb = ToolBox(RoomConfig(), [], fields_for=lambda f: fields())
    assert tb("list_modes", {"f_max": 80, "__class__": 1, "evil": "x"})["count"] > 0


# ------------------------------------------------------------------ report parsing
DIMS = RoomDims(5, 4, 2.5)


def test_parse_report_full():
    txt = "Here you go:\n" + json.dumps({
        "summary": "Back corner rings.", "findings": [{"title": "Slow corner", "detail": "d", "confidence": "HIGH", "evidence": ["EDT 1.4 s"],
        "marks": [{"x": 4.4, "y": 3.5, "label": "slow"}, {"x": 9, "y": 1, "label": "outside"}, {"x": "a", "y": 1}]}],
        "suggestions": ["trap in the corner"], "limits": ["few bursts"]})
    r = parse_report(txt, DIMS)
    assert r.parsed and r.findings[0]["confidence"] == "high"
    assert [(m.x, m.y, m.label) for m in r.marks] == [(4.4, 3.5, "slow")]
    out = r.to_text()
    assert "Slow corner" in out and "EDT 1.4 s" in out and "trap in the corner" in out and "few bursts" in out


def test_parse_report_caps_marks_and_survives_junk():
    many = {"summary": "s", "findings": [{"title": str(i), "marks": [{"x": 1, "y": 1, "label": "m"}] * 6} for i in range(5)]}
    assert len(parse_report(json.dumps(many), DIMS).marks) == 12
    for junk in ("no json at all", "{broken", "[1,2,3]"):
        r = parse_report(junk, DIMS)
        assert not r.parsed and r.to_text() == junk.strip()


def test_user_message_image_is_optional():
    ev = {"a": 1}
    assert [p["type"] for p in user_message(ev)] == ["text"]
    p = user_message(ev, "QUJD")
    assert p[0]["type"] == "image" and p[0]["source"]["data"] == "QUJD" and "attached image" in p[1]["text"]


def test_end_to_end_with_scripted_model():
    cfg = RoomConfig()
    tb = ToolBox(cfg, [], fields_for=lambda f: fields())
    final = json.dumps({"summary": "ok", "findings": [{"title": "t", "confidence": "low", "marks": [{"x": 4.4, "y": 2, "label": "a"}]}]})
    srv = FakeServer(tool_reply("decay_grid", {"cell_m": 1.0}), text_reply(final))
    res = ClaudeClient(KEY, transport=srv).run("sys", user_message(build_evidence(cfg, [], fields())), tools=TOOLS, handler=tb)
    rep = parse_report(res.text, cfg.dims)
    assert rep.summary == "ok" and len(rep.marks) == 1
    grid = json.loads(srv.requests[1][2]["messages"][2]["content"][0]["content"])
    assert grid["band_centre_hz"] == 63.0
