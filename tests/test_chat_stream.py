"""
chat 流式接口红绿灯测试（已合并到 hermes_cli/web_server.py 中）。

运行：conda activate deepagent && pytest tests/test_chat_stream.py -v
"""

import asyncio
import json
import sys
import time
import types
import uuid

import httpx
import pytest

# 注意：web_server 的鉴权 middleware 会拦 /api/chat/* 路由，
# 测试里通过往 _SESSION_TOKEN 注入固定值绕过即可。
from hermes_cli import web_server as srv


# ---------------------------------------------------------------------------
# 把真实 AIAgent 换成一个最小桩
# ---------------------------------------------------------------------------
class _StubAgent:
    """模拟 AIAgent：调用回调发若干 token，然后返回 final。"""

    def __init__(self, *args, **kwargs):
        self.session_id = kwargs.get("session_id") or uuid.uuid4().hex
        self.stream_delta_callback = None

    def run_conversation(self, user_message: str, *args, **kwargs):
        cb = self.stream_delta_callback
        tokens = ["你", "好", "，", "世", "界"]
        for t in tokens:
            if cb is not None:
                cb(t)
            time.sleep(0.02)
        return {"final_response": "".join(tokens)}


_TEST_TOKEN = "test-token-fixed"
_AUTH_HEADERS = {"Authorization": f"Bearer {_TEST_TOKEN}"}


@pytest.fixture(autouse=True)
def _patch_agent(monkeypatch, tmp_path):
    """把 web_server 的 AIAgent 换成桩，并清理 chat 全局单例。"""
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "test.db")
    monkeypatch.setattr(srv, "_chat_session_db", db, raising=False)
    monkeypatch.setattr(srv, "_chat_agent", None, raising=False)
    monkeypatch.setattr(srv, "_chat_session_id", None, raising=False)
    monkeypatch.setattr(srv, "_chat_session_locks", {}, raising=False)
    monkeypatch.setattr(srv, "AIAgent", _StubAgent)
    monkeypatch.setattr(srv, "load_config", lambda: {"model": {"default": "stub"}})
    monkeypatch.setattr(srv, "_SESSION_TOKEN", _TEST_TOKEN, raising=False)
    yield
    db.close()


def _parse_sse(raw: str):
    events = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        ev, data, eid = None, None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[7:].strip()
            elif line.startswith("data: "):
                data = json.loads(line[6:])
            elif line.startswith("id: "):
                eid = line[4:].strip()
        events.append((ev, data, eid))
    return events


async def _drain(resp) -> str:
    return "".join([c async for c in resp.aiter_text()])


def _run(coro):
    """用一个独立事件循环跑协程，避免依赖 pytest-asyncio。"""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 用例 1：stream
# ---------------------------------------------------------------------------
def test_chat_stream_meta_tokens_done():
    async def _go():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=10.0, headers=_AUTH_HEADERS
        ) as client:
            async with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                return await _drain(resp)

    raw = _run(_go())
    events = _parse_sse(raw)
    assert events[0][0] == "meta" and "message_id" in events[0][1]
    tokens = [e for e in events if e[0] == "token"]
    assert len(tokens) == 5
    ids = [int(e[2]) for e in tokens]
    assert ids == [0, 1, 2, 3, 4]
    assert events[-1][0] == "done"
    assert events[-1][1]["final"] == "你好，世界"


# ---------------------------------------------------------------------------
# 用例 2：resume
# ---------------------------------------------------------------------------
def test_chat_resume_after_partial():
    msg_id = uuid.uuid4().hex

    async def _go():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=10.0, headers=_AUTH_HEADERS
        ) as client:
            async with client.stream(
                "POST", "/api/chat/stream", json={"message": "hi", "message_id": msg_id}
            ) as resp:
                await _drain(resp)
            async with client.stream(
                "GET", "/api/chat/resume", params={"message_id": msg_id, "last_seq": 1}
            ) as resp:
                assert resp.status_code == 200
                return await _drain(resp)

    raw = _run(_go())
    events = _parse_sse(raw)
    assert events[0][0] == "meta" and events[0][1].get("resume") is True
    tokens = [e for e in events if e[0] == "token"]
    assert [int(e[2]) for e in tokens] == [2, 3, 4]
    assert events[-1][0] == "done"
    assert events[-1][1]["final"] == "你好，世界"


# ---------------------------------------------------------------------------
# 用例 3：未鉴权应当被 middleware 拦下
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 用例 4：voice WebSocket —— 短回复也必须返回音频和 done
# ---------------------------------------------------------------------------
def test_voice_websocket_short_reply_emits_audio_and_done(monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    class _StubVADEngine:
        def __init__(self, silence_threshold_ms=800):
            self.calls = 0

        def process_chunk(self, chunk: bytes):
            self.calls += 1
            return "speech_start" if self.calls == 1 else "speech_end"

        def close(self):
            return None

    # 用一个很短的回复，确保到不了 80 字阈值，能稳定复现“只出文字不出音频”的问题
    def _stub_run_agent_sync(user_text: str, sid: str, delta_queue, interrupt_event):
        delta_queue.put(("token", "你好"))
        return {"final_response": "你好"}

    def _stub_run_stt(pcm_bytes: bytes):
        return {"success": True, "transcript": "测试一下"}

    def _stub_run_tts(text: str):
        return {"success": True, "audio_data": b"fake-mp3", "format": "mp3"}

    fake_vad_module = types.ModuleType("tools.vad_engine")
    fake_vad_module.VADEngine = _StubVADEngine
    monkeypatch.setitem(sys.modules, "tools.vad_engine", fake_vad_module)
    monkeypatch.setattr(srv, "_MIN_SPEECH_BYTES", 0, raising=False)
    monkeypatch.setattr(srv, "_MIN_SPEECH_RMS", 0, raising=False)

    # 直接替换嵌套函数用到的全局名字，让逻辑专注在“结束时是否 flush TTS”
    setattr(srv, "_voice_test_stub_run_agent_sync", _stub_run_agent_sync)
    setattr(srv, "_voice_test_stub_run_stt", _stub_run_stt)
    setattr(srv, "_voice_test_stub_run_tts", _stub_run_tts)

    # 通过覆盖函数源码中会读取的全局引用，最小化干预现有实现
    orig_to_thread = srv._asyncio.to_thread

    async def _patched_to_thread(func, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name == "_run_stt":
            return _stub_run_stt(*args, **kwargs)
        if name == "_run_tts":
            return _stub_run_tts(*args, **kwargs)
        if name == "_run_agent_sync":
            return _stub_run_agent_sync(*args, **kwargs)
        return await orig_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(srv._asyncio, "to_thread", _patched_to_thread)

    client = TestClient(srv.app)
    client.headers["Authorization"] = f"Bearer {_TEST_TOKEN}"

    with client.websocket_connect(
        "/api/voice/ws",
        headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
    ) as ws:
        first = ws.receive_json()
        assert first == {"type": "status", "status": "idle"}

        ws.send_json({"type": "voice_start"})
        listening = ws.receive_json()
        assert listening == {"type": "status", "status": "listening"}

        # 发送 2 帧 PCM：第 1 帧触发 speech_start，第 2 帧触发 speech_end
        ws.send_bytes(b"\x01\x00" * 512)
        vad_start = ws.receive_json()
        assert vad_start == {"type": "vad", "status": "speech_start"}
        status_listening = ws.receive_json()
        assert status_listening == {"type": "status", "status": "listening"}

        ws.send_bytes(b"\x01\x00" * 512)
        vad_end = ws.receive_json()
        assert vad_end == {"type": "vad", "status": "speech_end"}
        thinking = ws.receive_json()
        assert thinking == {"type": "status", "status": "thinking"}
        transcription = ws.receive_json()
        assert transcription == {"type": "transcription", "text": "测试一下"}
        speaking = ws.receive_json()
        assert speaking == {"type": "status", "status": "speaking"}
        token = ws.receive_json()
        assert token == {"type": "token", "delta": "你好"}

        # 这里是本次回归的关键断言：短回复也应该发音频和 done
        audio = ws.receive_bytes()
        assert audio == b"fake-mp3"
        done = ws.receive_json()
        assert done == {"type": "done", "final": "你好"}
        back_to_listening = ws.receive_json()
        assert back_to_listening == {"type": "status", "status": "listening"}
