"""
会话接口红绿灯测试（GET/DELETE /api/sessions/*）。

运行：conda activate deepagent && pytest tests/test_sessions_api.py -v
"""

import json
import time

import httpx
import pytest

from hermes_cli import web_server as srv

# ---------------------------------------------------------------------------
# 测试用常量
# ---------------------------------------------------------------------------
_TEST_TOKEN = "test-token-fixed"
_AUTH_HEADERS = {"Authorization": f"Bearer {_TEST_TOKEN}"}

# ---------------------------------------------------------------------------
# SessionDB 桩数据
# ---------------------------------------------------------------------------
_SAMPLE_SESSIONS = [
    {
        "id": "sess-001",
        "source": "cli",
        "model": "deepseek-v4-pro",
        "title": "测试会话一",
        "started_at": 1714600000.0,
        "ended_at": None,
        "message_count": 12,
        "preview": "帮我写一个 Python 脚本",
        "last_active": time.time() - 60,
    },
    {
        "id": "sess-002",
        "source": "telegram",
        "model": "anthropic/claude-sonnet-4",
        "title": "测试会话二",
        "started_at": 1714610000.0,
        "ended_at": 1714615000.0,
        "message_count": 5,
        "preview": "今天天气怎么样",
        "last_active": 1714615000.0,
    },
]

_SAMPLE_MESSAGES = [
    {"role": "user", "content": "帮我写一个 Python 脚本", "timestamp": 1714600000.0},
    {"role": "assistant", "content": "好的，我来帮你写...", "timestamp": 1714600010.0},
    {"role": "user", "content": "能加点注释吗", "timestamp": 1714600100.0},
    {"role": "assistant", "content": "当然可以...", "timestamp": 1714600110.0},
]

_SAMPLE_SEARCH_RESULTS = [
    {
        "session_id": "sess-001",
        "snippet": "...帮我写一个 <b>Python</b> 脚本...",
        "role": "user",
        "source": "cli",
        "model": "deepseek-v4-pro",
        "session_started": 1714600000.0,
    },
]


class _StubSessionDB:
    """模拟 SessionDB，返回可控测试数据。"""

    def __init__(self, db_path=None):
        self.db_path = db_path

    def close(self):
        pass

    def list_sessions_rich(self, limit=20, offset=0, **kwargs):
        return _SAMPLE_SESSIONS[offset : offset + limit]

    def session_count(self):
        return len(_SAMPLE_SESSIONS)

    def search_messages(self, query="", limit=20, **kwargs):
        if not query:
            return []
        return _SAMPLE_SEARCH_RESULTS[:limit]

    def resolve_session_id(self, session_id):
        for s in _SAMPLE_SESSIONS:
            if s["id"] == session_id or s["id"].startswith(session_id):
                return s["id"]
        return None

    def get_session(self, sid):
        for s in _SAMPLE_SESSIONS:
            if s["id"] == sid:
                return dict(s)
        return None

    def get_messages(self, sid):
        if sid == "sess-001":
            return _SAMPLE_MESSAGES
        return []

    def delete_session(self, session_id):
        sid = self.resolve_session_id(session_id)
        return sid is not None


# ---------------------------------------------------------------------------
# 全局 fixture：替换 SessionDB、token 和 config 读写
# ---------------------------------------------------------------------------
_CONFIG_STORE = {"model": "stub", "agent": {"system_prompt": ""}}


@pytest.fixture(autouse=True)
def _patch_session_db(monkeypatch):
    """把所有 /api/sessions 和 config 用到的模块替换为桩。"""
    monkeypatch.setattr(srv, "_SESSION_TOKEN", _TEST_TOKEN, raising=False)
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _StubSessionDB)
    monkeypatch.setattr(srv, "SessionDB", _StubSessionDB)

    def _mock_load_config():
        return dict(_CONFIG_STORE)

    def _mock_save_config(config):
        _CONFIG_STORE.clear()
        _CONFIG_STORE.update(config)

    monkeypatch.setattr(srv, "load_config", _mock_load_config)
    monkeypatch.setattr(srv, "save_config", _mock_save_config)


def _get(path: str, params=None):
    """同步 GET 辅助函数。"""
    import asyncio

    async def _go():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            timeout=10.0,
            headers=_AUTH_HEADERS,
        ) as client:
            resp = await client.get(path, params=params)
            return resp.status_code, resp.json()

    return asyncio.new_event_loop().run_until_complete(_go())


def _delete(path: str):
    """同步 DELETE 辅助函数。"""
    import asyncio

    async def _go():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            timeout=10.0,
            headers=_AUTH_HEADERS,
        ) as client:
            resp = await client.delete(path)
            return resp.status_code, resp.json()

    return asyncio.new_event_loop().run_until_complete(_go())


# ===================================================================
# 用例 1：GET /api/sessions —— 列表 + 分页
# ===================================================================
def test_list_sessions_returns_paginated():
    code, body = _get("/api/sessions", {"limit": 1, "offset": 0})
    assert code == 200
    assert body["total"] == 2
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert len(body["sessions"]) == 1
    assert body["sessions"][0]["id"] == "sess-001"


def test_list_sessions_offset():
    code, body = _get("/api/sessions", {"limit": 20, "offset": 1})
    assert code == 200
    assert len(body["sessions"]) == 1
    assert body["sessions"][0]["id"] == "sess-002"


def test_list_sessions_marks_active():
    code, body = _get("/api/sessions")
    assert code == 200
    # sess-001 最近 60 秒内活跃，应为 active
    assert body["sessions"][0]["is_active"] is True
    # sess-002 已结束且 last_active 超过 300 秒，应为 inactive
    assert body["sessions"][1]["is_active"] is False


# ===================================================================
# 用例 2：GET /api/sessions/search —— 全文搜索
# ===================================================================
def test_search_sessions_returns_results():
    code, body = _get("/api/sessions/search", {"q": "Python", "limit": 5})
    assert code == 200
    assert len(body["results"]) == 1
    assert body["results"][0]["session_id"] == "sess-001"
    assert "Python" in body["results"][0]["snippet"]


def test_search_sessions_empty_query():
    code, body = _get("/api/sessions/search", {"q": ""})
    assert code == 200
    assert body["results"] == []


def test_search_sessions_whitespace_query():
    code, body = _get("/api/sessions/search", {"q": "   "})
    assert code == 200
    assert body["results"] == []


# ===================================================================
# 用例 3：POST /api/config —— 确认 system_prompt 可正常保存
# ===================================================================
def test_save_system_prompt_persists():
    """验证 agent.system_prompt 保存后能正确回传。"""
    import asyncio

    async def _go():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            timeout=10.0,
            headers=_AUTH_HEADERS,
        ) as client:
            # 写 —— body 需要包裹在 config 键下（ConfigUpdate 模型）
            payload = {"config": {"agent": {"system_prompt": "你是一个严谨的代码助手"}}}
            resp = await client.put("/api/config", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is True
            # 确认 get 也能读到新值（mock 了 load_config 返回同一份数据）
            resp2 = await client.get("/api/config")
            assert resp2.status_code == 200
            agent2 = resp2.json().get("agent") or {}
            assert agent2.get("system_prompt") == "你是一个严谨的代码助手"

    asyncio.new_event_loop().run_until_complete(_go())


# ===================================================================
# 用例 4：GET /api/sessions/{id} —— 会话详情
# ===================================================================
def test_get_session_detail_found():
    code, body = _get("/api/sessions/sess-001")
    assert code == 200
    assert body["id"] == "sess-001"
    assert body["title"] == "测试会话一"
    assert body["source"] == "cli"


def test_get_session_detail_not_found():
    code, body = _get("/api/sessions/nonexistent-id")
    assert code == 404
    assert "not found" in body.get("detail", "").lower()


# ===================================================================
# 用例 5：GET /api/sessions/{id}/messages —— 会话消息列表
# ===================================================================
def test_get_session_messages_found():
    code, body = _get("/api/sessions/sess-001/messages")
    assert code == 200
    assert body["session_id"] == "sess-001"
    assert len(body["messages"]) == 4
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"


def test_get_session_messages_empty():
    code, body = _get("/api/sessions/sess-002/messages")
    assert code == 200
    assert body["session_id"] == "sess-002"
    assert body["messages"] == []


def test_get_session_messages_not_found():
    code, body = _get("/api/sessions/nonexistent-id/messages")
    assert code == 404


# ===================================================================
# 用例 6：DELETE /api/sessions/{id} —— 删除会话
# ===================================================================
def test_delete_session_success():
    code, body = _delete("/api/sessions/sess-001")
    assert code == 200
    assert body["ok"] is True


def test_delete_session_not_found():
    code, body = _delete("/api/sessions/nonexistent-id")
    assert code == 404
    # 前端依赖这个 404 来触发 toast 提示
    assert "not found" in body.get("detail", "").lower()


# ===================================================================
# 用例 7：鉴权 —— 无 token 时返回 401
# ===================================================================
def test_sessions_require_auth():
    import asyncio

    async def _go():
        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            timeout=10.0,
        ) as client:
            resp = await client.get("/api/sessions")
            return resp.status_code, resp.json()

    code, body = asyncio.new_event_loop().run_until_complete(_go())
    assert code == 401
