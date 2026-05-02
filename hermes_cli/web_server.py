"""
Hermes Agent — Web UI server.

Provides a FastAPI backend serving the Vite/React frontend and REST API
endpoints for managing configuration, environment variables, and sessions.

Usage:
    python -m hermes_cli.main web          # Start on http://127.0.0.1:9119
    python -m hermes_cli.main web --port 8080
"""

import asyncio
import hmac
import importlib.util
import json
import logging
import os
import secrets
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli import __version__, __release_date__
from hermes_cli.config import (
    DEFAULT_CONFIG,
    OPTIONAL_ENV_VARS,
    get_config_path,
    get_env_path,
    get_hermes_home,
    load_config,
    load_env,
    save_config,
    save_env_value,
    remove_env_value,
    check_config_version,
    redact_key,
)
from gateway.status import get_running_pid, read_runtime_status

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    raise SystemExit(
        "Web UI requires fastapi and uvicorn.\n"
        f"Install with: {sys.executable} -m pip install 'fastapi' 'uvicorn[standard]'"
    )

WEB_DIST = Path(os.environ["HERMES_WEB_DIST"]) if "HERMES_WEB_DIST" in os.environ else Path(__file__).parent / "web_dist"
_log = logging.getLogger(__name__)

app = FastAPI(title="Hermes Agent", version=__version__)

# ---------------------------------------------------------------------------
# Session token for protecting sensitive endpoints (reveal).
# Generated fresh on every server start — dies when the process exits.
# Injected into the SPA HTML so only the legitimate web UI can use it.
# ---------------------------------------------------------------------------
_SESSION_TOKEN = secrets.token_urlsafe(32)

# Simple rate limiter for the reveal endpoint
_reveal_timestamps: List[float] = []
_REVEAL_MAX_PER_WINDOW = 5
_REVEAL_WINDOW_SECONDS = 30

# CORS: restrict to localhost origins only.  The web UI is intended to run
# locally; binding to 0.0.0.0 with allow_origins=["*"] would let any website
# read/modify config and secrets.

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints that do NOT require the session token.  Everything else under
# /api/ is gated by the auth middleware below.  Keep this list minimal —
# only truly non-sensitive, read-only endpoints belong here.
# ---------------------------------------------------------------------------
_PUBLIC_API_PATHS: frozenset = frozenset({
    "/api/status",
    "/api/config/defaults",
    "/api/config/schema",
    "/api/model/info",
    "/api/dashboard/themes",
    "/api/dashboard/plugins",
    "/api/dashboard/plugins/rescan",
})


def _require_token(request: Request) -> None:
    """Validate the ephemeral session token.  Raises 401 on mismatch.

    Uses ``hmac.compare_digest`` to prevent timing side-channels.
    """
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {_SESSION_TOKEN}"
    if not hmac.compare_digest(auth.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require the session token on all /api/ routes except the public list."""
    path = request.url.path
    if path.startswith("/api/") and path not in _PUBLIC_API_PATHS and not path.startswith("/api/plugins/"):
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {_SESSION_TOKEN}"
        if not hmac.compare_digest(auth.encode(), expected.encode()):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Config schema — auto-generated from DEFAULT_CONFIG
# ---------------------------------------------------------------------------

# Manual overrides for fields that need select options or custom types
_SCHEMA_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "model": {
        "type": "string",
        "description": "Default model (e.g. anthropic/claude-sonnet-4.6)",
        "category": "general",
    },
    "model_context_length": {
        "type": "number",
        "description": "Context window override (0 = auto-detect from model metadata)",
        "category": "general",
    },
    "terminal.backend": {
        "type": "select",
        "description": "Terminal execution backend",
        "options": ["local", "docker", "ssh", "modal", "daytona", "singularity"],
    },
    "terminal.modal_mode": {
        "type": "select",
        "description": "Modal sandbox mode",
        "options": ["sandbox", "function"],
    },
    "tts.provider": {
        "type": "select",
        "description": "Text-to-speech provider",
        "options": ["edge", "aliyun", "elevenlabs", "openai", "neutts"],
    },
    "tts.edge.voice": {
        "type": "select",
        "description": "Edge TTS 发音人",
        "options": [
            "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-YunjianNeural",
            "zh-CN-XiaoyiNeural", "zh-CN-YunyangNeural", "zh-CN-XiaochenNeural",
            "zh-CN-XiaohanNeural", "zh-CN-XiaomengNeural", "zh-CN-XiaomoNeural",
            "zh-CN-XiaoqiuNeural", "zh-CN-XiaoruiNeural", "zh-CN-XiaoshuangNeural",
            "zh-CN-XiaoxuanNeural", "zh-CN-XiaoyanNeural", "zh-CN-XiaoyouNeural",
            "zh-CN-XiaozhenNeural", "en-US-AriaNeural", "en-US-JennyNeural",
        ],
    },
    "tts.openai.voice": {
        "type": "select",
        "description": "OpenAI TTS 发音人",
        "options": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    },
    "tts.openai.model": {
        "type": "select",
        "description": "OpenAI TTS 模型",
        "options": ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
    },
    "voice.aliyun.stt_model": {
        "type": "select",
        "description": "阿里云 STT 模型",
        "options": ["sensevoice-v1"],
    },
    "voice.aliyun.tts_model": {
        "type": "select",
        "description": "阿里云 TTS 模型",
        "options": ["cosyvoice-v1", "cosyvoice-v2"],
    },
    "voice.aliyun.tts_voice": {
        "type": "select",
        "description": "阿里云 TTS 发音人",
        "options": [
            "longxiaochun", "longxiaoxia", "longxiaobai", "longxiaochen",
            "longxiaozhi", "longxiaomei", "longxiaofei", "longxiaotong",
            "longxiaomeng", "longxiaojie", "longxiaorui", "longxiaoyu",
        ],
    },
    "stt.provider": {
        "type": "select",
        "description": "Speech-to-text provider",
        "options": ["local", "openai", "mistral"],
    },
    "display.skin": {
        "type": "select",
        "description": "CLI visual theme",
        "options": ["default", "ares", "mono", "slate"],
    },
    "dashboard.theme": {
        "type": "select",
        "description": "Web dashboard visual theme",
        "options": ["default", "midnight", "ember", "mono", "cyberpunk", "rose"],
    },
    "display.resume_display": {
        "type": "select",
        "description": "How resumed sessions display history",
        "options": ["minimal", "full", "off"],
    },
    "display.busy_input_mode": {
        "type": "select",
        "description": "Input behavior while agent is running",
        "options": ["queue", "interrupt", "block"],
    },
    "memory.provider": {
        "type": "select",
        "description": "Memory provider plugin",
        "options": ["builtin", "honcho"],
    },
    "approvals.mode": {
        "type": "select",
        "description": "Dangerous command approval mode",
        "options": ["ask", "yolo", "deny"],
    },
    "context.engine": {
        "type": "select",
        "description": "Context management engine",
        "options": ["default", "custom"],
    },
    "human_delay.mode": {
        "type": "select",
        "description": "Simulated typing delay mode",
        "options": ["off", "typing", "fixed"],
    },
    "logging.level": {
        "type": "select",
        "description": "Log level for agent.log",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
    },
    "agent.service_tier": {
        "type": "select",
        "description": "API service tier (OpenAI/Anthropic)",
        "options": ["", "auto", "default", "flex"],
    },
    "agent.system_prompt": {
        "type": "text",
        "description": "自定义系统提示词，配置 LLM 的性格和回答风格。留空则使用默认身份（SOUL.md）",
        "category": "agent",
    },
    "delegation.reasoning_effort": {
        "type": "select",
        "description": "Reasoning effort for delegated subagents",
        "options": ["", "low", "medium", "high"],
    },
}

# Categories with fewer fields get merged into "general" to avoid tab sprawl.
_CATEGORY_MERGE: Dict[str, str] = {
    "privacy": "security",
    "context": "agent",
    "skills": "agent",
    "cron": "agent",
    "network": "agent",
    "checkpoints": "agent",
    "approvals": "security",
    "human_delay": "display",
    "smart_model_routing": "agent",
    "dashboard": "display",
}

# Display order for tabs — unlisted categories sort alphabetically after these.
_CATEGORY_ORDER = [
    "general", "agent", "terminal", "display", "delegation",
    "memory", "compression", "security", "browser", "voice",
    "tts", "stt", "logging", "discord", "auxiliary",
]


def _infer_type(value: Any) -> str:
    """Infer a UI field type from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def _build_schema_from_config(
    config: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Walk DEFAULT_CONFIG and produce a flat dot-path → field schema dict."""
    schema: Dict[str, Dict[str, Any]] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        # Skip internal / version keys
        if full_key in ("_config_version",):
            continue

        # Category is the first path component for nested keys, or "general"
        # for top-level scalar fields (model, toolsets, timezone, etc.).
        if prefix:
            category = prefix.split(".")[0]
        elif isinstance(value, dict):
            category = key
        else:
            category = "general"

        if isinstance(value, dict):
            # Recurse into nested dicts
            schema.update(_build_schema_from_config(value, full_key))
        else:
            entry: Dict[str, Any] = {
                "type": _infer_type(value),
                "description": full_key.replace(".", " → ").replace("_", " ").title(),
                "category": category,
            }
            # Apply manual overrides
            if full_key in _SCHEMA_OVERRIDES:
                entry.update(_SCHEMA_OVERRIDES[full_key])
            # Merge small categories
            entry["category"] = _CATEGORY_MERGE.get(entry["category"], entry["category"])
            schema[full_key] = entry
    return schema


CONFIG_SCHEMA = _build_schema_from_config(DEFAULT_CONFIG)

# Inject virtual fields that don't live in DEFAULT_CONFIG but are surfaced
# by the normalize/denormalize cycle.  Insert model_context_length right after
# the "model" key so it renders adjacent in the frontend.
# Also inject voice.aliyun.* fields so the Aliyun TTS/STT config is visible
# in ConfigPage when tts.provider is set to "aliyun".
_mcl_entry = _SCHEMA_OVERRIDES["model_context_length"]
_ordered_schema: Dict[str, Dict[str, Any]] = {}
for _k, _v in CONFIG_SCHEMA.items():
    _ordered_schema[_k] = _v
    if _k == "model":
        _ordered_schema["model_context_length"] = _mcl_entry
    if _k == "voice.silence_duration":
        for _vk, _cat in (("voice.aliyun.stt_model", "stt"), ("voice.aliyun.tts_model", "tts"), ("voice.aliyun.tts_voice", "tts")):
            _v = dict(_SCHEMA_OVERRIDES[_vk])
            _v.setdefault("category", _cat)
            _ordered_schema[_vk] = _v
CONFIG_SCHEMA = _ordered_schema


class ConfigUpdate(BaseModel):
    config: dict


class EnvVarUpdate(BaseModel):
    key: str
    value: str


class EnvVarDelete(BaseModel):
    key: str


class EnvVarReveal(BaseModel):
    key: str


_GATEWAY_HEALTH_URL = os.getenv("GATEWAY_HEALTH_URL")
_GATEWAY_HEALTH_TIMEOUT = float(os.getenv("GATEWAY_HEALTH_TIMEOUT", "3"))


def _probe_gateway_health() -> tuple[bool, dict | None]:
    """Probe the gateway via its HTTP health endpoint (cross-container).

    Uses ``/health/detailed`` first (returns full state), falling back to
    the simpler ``/health`` endpoint.  Returns ``(is_alive, body_dict)``.

    Accepts any of these as ``GATEWAY_HEALTH_URL``:
    - ``http://gateway:8642``                (base URL — recommended)
    - ``http://gateway:8642/health``         (explicit health path)
    - ``http://gateway:8642/health/detailed`` (explicit detailed path)

    This is a **blocking** call — run via ``run_in_executor`` from async code.
    """
    if not _GATEWAY_HEALTH_URL:
        return False, None

    # Normalise to base URL so we always probe the right paths regardless of
    # whether the user included /health or /health/detailed in the env var.
    base = _GATEWAY_HEALTH_URL.rstrip("/")
    if base.endswith("/health/detailed"):
        base = base[: -len("/health/detailed")]
    elif base.endswith("/health"):
        base = base[: -len("/health")]

    for path in (f"{base}/health/detailed", f"{base}/health"):
        try:
            req = urllib.request.Request(path, method="GET")
            with urllib.request.urlopen(req, timeout=_GATEWAY_HEALTH_TIMEOUT) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read())
                    return True, body
        except Exception:
            continue
    return False, None


@app.get("/api/status")
async def get_status():
    current_ver, latest_ver = check_config_version()

    # --- Gateway liveness detection ---
    # Try local PID check first (same-host).  If that fails and a remote
    # GATEWAY_HEALTH_URL is configured, probe the gateway over HTTP so the
    # dashboard works when the gateway runs in a separate container.
    gateway_pid = get_running_pid()
    gateway_running = gateway_pid is not None
    remote_health_body: dict | None = None

    if not gateway_running and _GATEWAY_HEALTH_URL:
        loop = asyncio.get_event_loop()
        alive, remote_health_body = await loop.run_in_executor(
            None, _probe_gateway_health
        )
        if alive:
            gateway_running = True
            # PID from the remote container (display only — not locally valid)
            if remote_health_body:
                gateway_pid = remote_health_body.get("pid")

    gateway_state = None
    gateway_platforms: dict = {}
    gateway_exit_reason = None
    gateway_updated_at = None
    configured_gateway_platforms: set[str] | None = None
    try:
        from gateway.config import load_gateway_config

        gateway_config = load_gateway_config()
        configured_gateway_platforms = {
            platform.value for platform in gateway_config.get_connected_platforms()
        }
    except Exception:
        configured_gateway_platforms = None

    # Prefer the detailed health endpoint response (has full state) when the
    # local runtime status file is absent or stale (cross-container).
    runtime = read_runtime_status()
    if runtime is None and remote_health_body and remote_health_body.get("gateway_state"):
        runtime = remote_health_body

    if runtime:
        gateway_state = runtime.get("gateway_state")
        gateway_platforms = runtime.get("platforms") or {}
        if configured_gateway_platforms is not None:
            gateway_platforms = {
                key: value
                for key, value in gateway_platforms.items()
                if key in configured_gateway_platforms
            }
        gateway_exit_reason = runtime.get("exit_reason")
        gateway_updated_at = runtime.get("updated_at")
        if not gateway_running:
            gateway_state = gateway_state if gateway_state in ("stopped", "startup_failed") else "stopped"
            gateway_platforms = {}
        elif gateway_running and remote_health_body is not None:
            # The health probe confirmed the gateway is alive, but the local
            # runtime status file may be stale (cross-container).  Override
            # stopped/None state so the dashboard shows the correct badge.
            if gateway_state in (None, "stopped"):
                gateway_state = "running"

    # If there was no runtime info at all but the health probe confirmed alive,
    # ensure we still report the gateway as running (no shared volume scenario).
    if gateway_running and gateway_state is None and remote_health_body is not None:
        gateway_state = "running"

    active_sessions = 0
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            sessions = db.list_sessions_rich(limit=50)
            now = time.time()
            active_sessions = sum(
                1 for s in sessions
                if s.get("ended_at") is None
                and (now - s.get("last_active", s.get("started_at", 0))) < 300
            )
        finally:
            db.close()
    except Exception:
        pass

    return {
        "version": __version__,
        "release_date": __release_date__,
        "hermes_home": str(get_hermes_home()),
        "config_path": str(get_config_path()),
        "env_path": str(get_env_path()),
        "config_version": current_ver,
        "latest_config_version": latest_ver,
        "gateway_running": gateway_running,
        "gateway_pid": gateway_pid,
        "gateway_health_url": _GATEWAY_HEALTH_URL,
        "gateway_state": gateway_state,
        "gateway_platforms": gateway_platforms,
        "gateway_exit_reason": gateway_exit_reason,
        "gateway_updated_at": gateway_updated_at,
        "active_sessions": active_sessions,
    }


@app.get("/api/sessions")
async def get_sessions(limit: int = 20, offset: int = 0):
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            sessions = db.list_sessions_rich(limit=limit, offset=offset)
            total = db.session_count()
            now = time.time()
            for s in sessions:
                s["is_active"] = (
                    s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
            return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}
        finally:
            db.close()
    except Exception as e:
        _log.exception("GET /api/sessions failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/sessions/search")
async def search_sessions(q: str = "", limit: int = 20):
    """Full-text search across session message content using FTS5."""
    if not q or not q.strip():
        return {"results": []}
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            # Auto-add prefix wildcards so partial words match
            # e.g. "nimb" → "nimb*" matches "nimby"
            # Preserve quoted phrases and existing wildcards as-is
            import re
            terms = []
            for token in re.findall(r'"[^"]*"|\S+', q.strip()):
                if token.startswith('"') or token.endswith("*"):
                    terms.append(token)
                else:
                    terms.append(token + "*")
            prefix_query = " ".join(terms)
            matches = db.search_messages(query=prefix_query, limit=limit)
            # Group by session_id — return unique sessions with their best snippet
            seen: dict = {}
            for m in matches:
                sid = m["session_id"]
                if sid not in seen:
                    seen[sid] = {
                        "session_id": sid,
                        "snippet": m.get("snippet", ""),
                        "role": m.get("role"),
                        "source": m.get("source"),
                        "model": m.get("model"),
                        "session_started": m.get("session_started"),
                    }
            return {"results": list(seen.values())}
        finally:
            db.close()
    except Exception:
        _log.exception("GET /api/sessions/search failed")
        raise HTTPException(status_code=500, detail="Search failed")


def _normalize_config_for_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config for the web UI.

    Hermes supports ``model`` as either a bare string (``"anthropic/claude-sonnet-4"``)
    or a dict (``{default: ..., provider: ..., base_url: ...}``).  The schema is built
    from DEFAULT_CONFIG where ``model`` is a string, but user configs often have the
    dict form.  Normalize to the string form so the frontend schema matches.

    Also surfaces ``model_context_length`` as a top-level field so the web UI can
    display and edit it.  A value of 0 means "auto-detect".
    """
    config = dict(config)  # shallow copy
    model_val = config.get("model")
    if isinstance(model_val, dict):
        # Extract context_length before flattening the dict
        ctx_len = model_val.get("context_length", 0)
        config["model"] = model_val.get("default", model_val.get("name", ""))
        config["model_context_length"] = ctx_len if isinstance(ctx_len, int) else 0
    else:
        config["model_context_length"] = 0
    return config


@app.get("/api/config")
async def get_config():
    config = _normalize_config_for_web(load_config())
    # Strip internal keys that the frontend shouldn't see or send back
    return {k: v for k, v in config.items() if not k.startswith("_")}


@app.get("/api/config/defaults")
async def get_defaults():
    return DEFAULT_CONFIG


@app.get("/api/config/schema")
async def get_schema():
    return {"fields": CONFIG_SCHEMA, "category_order": _CATEGORY_ORDER}


_EMPTY_MODEL_INFO: dict = {
    "model": "",
    "provider": "",
    "auto_context_length": 0,
    "config_context_length": 0,
    "effective_context_length": 0,
    "capabilities": {},
}


@app.get("/api/model/info")
def get_model_info():
    """Return resolved model metadata for the currently configured model.

    Calls the same context-length resolution chain the agent uses, so the
    frontend can display "Auto-detected: 200K" alongside the override field.
    Also returns model capabilities (vision, reasoning, tools) when available.
    """
    try:
        cfg = load_config()
        model_cfg = cfg.get("model", "")

        # Extract model name and provider from the config
        if isinstance(model_cfg, dict):
            model_name = model_cfg.get("default", model_cfg.get("name", ""))
            provider = model_cfg.get("provider", "")
            base_url = model_cfg.get("base_url", "")
            config_ctx = model_cfg.get("context_length")
        else:
            model_name = str(model_cfg) if model_cfg else ""
            provider = ""
            base_url = ""
            config_ctx = None

        if not model_name:
            return dict(_EMPTY_MODEL_INFO, provider=provider)

        # Resolve auto-detected context length (pass config_ctx=None to get
        # purely auto-detected value, then separately report the override)
        try:
            from agent.model_metadata import get_model_context_length
            auto_ctx = get_model_context_length(
                model=model_name,
                base_url=base_url,
                provider=provider,
                config_context_length=None,  # ignore override — we want auto value
            )
        except Exception:
            auto_ctx = 0

        config_ctx_int = 0
        if isinstance(config_ctx, int) and config_ctx > 0:
            config_ctx_int = config_ctx

        # Effective is what the agent actually uses
        effective_ctx = config_ctx_int if config_ctx_int > 0 else auto_ctx

        # Try to get model capabilities from models.dev
        caps = {}
        try:
            from agent.models_dev import get_model_capabilities
            mc = get_model_capabilities(provider=provider, model=model_name)
            if mc is not None:
                caps = {
                    "supports_tools": mc.supports_tools,
                    "supports_vision": mc.supports_vision,
                    "supports_reasoning": mc.supports_reasoning,
                    "context_window": mc.context_window,
                    "max_output_tokens": mc.max_output_tokens,
                    "model_family": mc.model_family,
                }
        except Exception:
            pass

        return {
            "model": model_name,
            "provider": provider,
            "auto_context_length": auto_ctx,
            "config_context_length": config_ctx_int,
            "effective_context_length": effective_ctx,
            "capabilities": caps,
        }
    except Exception:
        _log.exception("GET /api/model/info failed")
        return dict(_EMPTY_MODEL_INFO)


def _denormalize_config_from_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse _normalize_config_for_web before saving.

    Reconstructs ``model`` as a dict by reading the current on-disk config
    to recover model subkeys (provider, base_url, api_mode, etc.) that were
    stripped from the GET response.  The frontend only sees model as a flat
    string; the rest is preserved transparently.

    Also handles ``model_context_length`` — writes it back into the model dict
    as ``context_length``.  A value of 0 or absent means "auto-detect" (omitted
    from the dict so get_model_context_length() uses its normal resolution).
    """
    config = dict(config)
    # Remove any _model_meta that might have leaked in (shouldn't happen
    # with the stripped GET response, but be defensive)
    config.pop("_model_meta", None)

    # Extract and remove model_context_length before processing model
    ctx_override = config.pop("model_context_length", 0)
    if not isinstance(ctx_override, int):
        try:
            ctx_override = int(ctx_override)
        except (TypeError, ValueError):
            ctx_override = 0

    model_val = config.get("model")
    if isinstance(model_val, str) and model_val:
        # Read the current disk config to recover model subkeys
        try:
            disk_config = load_config()
            disk_model = disk_config.get("model")
            if isinstance(disk_model, dict):
                # Preserve all subkeys, update default with the new value
                disk_model["default"] = model_val
                # Write context_length into the model dict (0 = remove/auto)
                if ctx_override > 0:
                    disk_model["context_length"] = ctx_override
                else:
                    disk_model.pop("context_length", None)
                config["model"] = disk_model
            else:
                # Model was previously a bare string — upgrade to dict if
                # user is setting a context_length override
                if ctx_override > 0:
                    config["model"] = {
                        "default": model_val,
                        "context_length": ctx_override,
                    }
        except Exception:
            pass  # can't read disk config — just use the string form
    return config


@app.put("/api/config")
async def update_config(body: ConfigUpdate):
    try:
        save_config(_denormalize_config_from_web(body.config))
        return {"ok": True}
    except Exception as e:
        _log.exception("PUT /api/config failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/env")
async def get_env_vars():
    env_on_disk = load_env()
    result = {}
    for var_name, info in OPTIONAL_ENV_VARS.items():
        value = env_on_disk.get(var_name)
        result[var_name] = {
            "is_set": bool(value),
            "redacted_value": redact_key(value) if value else None,
            "description": info.get("description", ""),
            "url": info.get("url"),
            "category": info.get("category", ""),
            "is_password": info.get("password", False),
            "tools": info.get("tools", []),
            "advanced": info.get("advanced", False),
        }
    return result


@app.put("/api/env")
async def set_env_var(body: EnvVarUpdate):
    try:
        save_env_value(body.key, body.value)
        return {"ok": True, "key": body.key}
    except Exception as e:
        _log.exception("PUT /api/env failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/env")
async def remove_env_var(body: EnvVarDelete):
    try:
        removed = remove_env_value(body.key)
        if not removed:
            raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")
        return {"ok": True, "key": body.key}
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("DELETE /api/env failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/env/reveal")
async def reveal_env_var(body: EnvVarReveal, request: Request):
    """Return the real (unredacted) value of a single env var.

    Protected by:
    - Ephemeral session token (generated per server start, injected into SPA)
    - Rate limiting (max 5 reveals per 30s window)
    - Audit logging
    """
    # --- Token check ---
    _require_token(request)

    # --- Rate limit ---
    now = time.time()
    cutoff = now - _REVEAL_WINDOW_SECONDS
    _reveal_timestamps[:] = [t for t in _reveal_timestamps if t > cutoff]
    if len(_reveal_timestamps) >= _REVEAL_MAX_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many reveal requests. Try again shortly.")
    _reveal_timestamps.append(now)

    # --- Reveal ---
    env_on_disk = load_env()
    value = env_on_disk.get(body.key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")

    _log.info("env/reveal: %s", body.key)
    return {"key": body.key, "value": value}


# ---------------------------------------------------------------------------
# OAuth provider endpoints — status + disconnect (Phase 1)
# ---------------------------------------------------------------------------
#
# Phase 1 surfaces *which OAuth providers exist* and whether each is
# connected, plus a disconnect button. The actual login flow (PKCE for
# Anthropic, device-code for Nous/Codex) still runs in the CLI for now;
# Phase 2 will add in-browser flows. For unconnected providers we return
# the canonical ``hermes auth add <provider>`` command so the dashboard
# can surface a one-click copy.


def _truncate_token(value: Optional[str], visible: int = 6) -> str:
    """Return ``...XXXXXX`` (last N chars) for safe display in the UI.

    We never expose more than the trailing ``visible`` characters of an
    OAuth access token. JWT prefixes (the part before the first dot) are
    stripped first when present so the visible suffix is always part of
    the signing region rather than a meaningless header chunk.
    """
    if not value:
        return ""
    s = str(value)
    if "." in s and s.count(".") >= 2:
        # Looks like a JWT — show the trailing piece of the signature only.
        s = s.rsplit(".", 1)[-1]
    if len(s) <= visible:
        return s
    return f"…{s[-visible:]}"


def _anthropic_oauth_status() -> Dict[str, Any]:
    """Combined status across the three Anthropic credential sources we read.

    Hermes resolves Anthropic creds in this order at runtime:
    1. ``~/.hermes/.anthropic_oauth.json`` — Hermes-managed PKCE flow
    2. ``~/.claude/.credentials.json`` — Claude Code CLI credentials (auto)
    3. ``ANTHROPIC_TOKEN`` / ``ANTHROPIC_API_KEY`` env vars
    The dashboard reports the highest-priority source that's actually present.
    """
    try:
        from agent.anthropic_adapter import (
            read_hermes_oauth_credentials,
            read_claude_code_credentials,
            _HERMES_OAUTH_FILE,
        )
    except ImportError:
        read_claude_code_credentials = None  # type: ignore
        read_hermes_oauth_credentials = None  # type: ignore
        _HERMES_OAUTH_FILE = None  # type: ignore

    hermes_creds = None
    if read_hermes_oauth_credentials:
        try:
            hermes_creds = read_hermes_oauth_credentials()
        except Exception:
            hermes_creds = None
    if hermes_creds and hermes_creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "hermes_pkce",
            "source_label": f"Hermes PKCE ({_HERMES_OAUTH_FILE})",
            "token_preview": _truncate_token(hermes_creds.get("accessToken")),
            "expires_at": hermes_creds.get("expiresAt"),
            "has_refresh_token": bool(hermes_creds.get("refreshToken")),
        }

    cc_creds = None
    if read_claude_code_credentials:
        try:
            cc_creds = read_claude_code_credentials()
        except Exception:
            cc_creds = None
    if cc_creds and cc_creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "claude_code",
            "source_label": "Claude Code (~/.claude/.credentials.json)",
            "token_preview": _truncate_token(cc_creds.get("accessToken")),
            "expires_at": cc_creds.get("expiresAt"),
            "has_refresh_token": bool(cc_creds.get("refreshToken")),
        }

    env_token = os.getenv("ANTHROPIC_TOKEN") or os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
    if env_token:
        return {
            "logged_in": True,
            "source": "env_var",
            "source_label": "ANTHROPIC_TOKEN environment variable",
            "token_preview": _truncate_token(env_token),
            "expires_at": None,
            "has_refresh_token": False,
        }
    return {"logged_in": False, "source": None}


def _claude_code_only_status() -> Dict[str, Any]:
    """Surface Claude Code CLI credentials as their own provider entry.

    Independent of the Anthropic entry above so users can see whether their
    Claude Code subscription tokens are actively flowing into Hermes even
    when they also have a separate Hermes-managed PKCE login.
    """
    try:
        from agent.anthropic_adapter import read_claude_code_credentials
        creds = read_claude_code_credentials()
    except Exception:
        creds = None
    if creds and creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "claude_code_cli",
            "source_label": "~/.claude/.credentials.json",
            "token_preview": _truncate_token(creds.get("accessToken")),
            "expires_at": creds.get("expiresAt"),
            "has_refresh_token": bool(creds.get("refreshToken")),
        }
    return {"logged_in": False, "source": None}


# Provider catalog. The order matters — it's how we render the UI list.
# ``cli_command`` is what the dashboard surfaces as the copy-to-clipboard
# fallback while Phase 2 (in-browser flows) isn't built yet.
# ``flow`` describes the OAuth shape so the future modal can pick the
# right UI: ``pkce`` = open URL + paste callback code, ``device_code`` =
# show code + verification URL + poll, ``external`` = read-only (delegated
# to a third-party CLI like Claude Code or Qwen).
_OAUTH_PROVIDER_CATALOG: tuple[Dict[str, Any], ...] = (
    {
        "id": "anthropic",
        "name": "Anthropic (Claude API)",
        "flow": "pkce",
        "cli_command": "hermes auth add anthropic",
        "docs_url": "https://docs.claude.com/en/api/getting-started",
        "status_fn": _anthropic_oauth_status,
    },
    {
        "id": "claude-code",
        "name": "Claude Code (subscription)",
        "flow": "external",
        "cli_command": "claude setup-token",
        "docs_url": "https://docs.claude.com/en/docs/claude-code",
        "status_fn": _claude_code_only_status,
    },
    {
        "id": "nous",
        "name": "Nous Portal",
        "flow": "device_code",
        "cli_command": "hermes auth add nous",
        "docs_url": "https://portal.nousresearch.com",
        "status_fn": None,  # dispatched via auth.get_nous_auth_status
    },
    {
        "id": "openai-codex",
        "name": "OpenAI Codex (ChatGPT)",
        "flow": "device_code",
        "cli_command": "hermes auth add openai-codex",
        "docs_url": "https://platform.openai.com/docs",
        "status_fn": None,  # dispatched via auth.get_codex_auth_status
    },
    {
        "id": "qwen-oauth",
        "name": "Qwen (via Qwen CLI)",
        "flow": "external",
        "cli_command": "hermes auth add qwen-oauth",
        "docs_url": "https://github.com/QwenLM/qwen-code",
        "status_fn": None,  # dispatched via auth.get_qwen_auth_status
    },
)


def _resolve_provider_status(provider_id: str, status_fn) -> Dict[str, Any]:
    """Dispatch to the right status helper for an OAuth provider entry."""
    if status_fn is not None:
        try:
            return status_fn()
        except Exception as e:
            return {"logged_in": False, "error": str(e)}
    try:
        from hermes_cli import auth as hauth
        if provider_id == "nous":
            raw = hauth.get_nous_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": "nous_portal",
                "source_label": raw.get("portal_base_url") or "Nous Portal",
                "token_preview": _truncate_token(raw.get("access_token")),
                "expires_at": raw.get("access_expires_at"),
                "has_refresh_token": bool(raw.get("has_refresh_token")),
            }
        if provider_id == "openai-codex":
            raw = hauth.get_codex_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": raw.get("source") or "openai_codex",
                "source_label": raw.get("auth_mode") or "OpenAI Codex",
                "token_preview": _truncate_token(raw.get("api_key")),
                "expires_at": None,
                "has_refresh_token": False,
                "last_refresh": raw.get("last_refresh"),
            }
        if provider_id == "qwen-oauth":
            raw = hauth.get_qwen_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": "qwen_cli",
                "source_label": raw.get("auth_store_path") or "Qwen CLI",
                "token_preview": _truncate_token(raw.get("access_token")),
                "expires_at": raw.get("expires_at"),
                "has_refresh_token": bool(raw.get("has_refresh_token")),
            }
    except Exception as e:
        return {"logged_in": False, "error": str(e)}
    return {"logged_in": False}


@app.get("/api/providers/oauth")
async def list_oauth_providers():
    """Enumerate every OAuth-capable LLM provider with current status.

    Response shape (per provider):
        id              stable identifier (used in DELETE path)
        name            human label
        flow            "pkce" | "device_code" | "external"
        cli_command     fallback CLI command for users to run manually
        docs_url        external docs/portal link for the "Learn more" link
        status:
          logged_in        bool — currently has usable creds
          source           short slug ("hermes_pkce", "claude_code", ...)
          source_label     human-readable origin (file path, env var name)
          token_preview    last N chars of the token, never the full token
          expires_at       ISO timestamp string or null
          has_refresh_token bool
    """
    providers = []
    for p in _OAUTH_PROVIDER_CATALOG:
        status = _resolve_provider_status(p["id"], p.get("status_fn"))
        providers.append({
            "id": p["id"],
            "name": p["name"],
            "flow": p["flow"],
            "cli_command": p["cli_command"],
            "docs_url": p["docs_url"],
            "status": status,
        })
    return {"providers": providers}


@app.delete("/api/providers/oauth/{provider_id}")
async def disconnect_oauth_provider(provider_id: str, request: Request):
    """Disconnect an OAuth provider. Token-protected (matches /env/reveal)."""
    _require_token(request)

    valid_ids = {p["id"] for p in _OAUTH_PROVIDER_CATALOG}
    if provider_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {provider_id}. "
                   f"Available: {', '.join(sorted(valid_ids))}",
        )

    # Anthropic and claude-code clear the same Hermes-managed PKCE file
    # AND forget the Claude Code import. We don't touch ~/.claude/* directly
    # — that's owned by the Claude Code CLI; users can re-auth there if they
    # want to undo a disconnect.
    if provider_id in ("anthropic", "claude-code"):
        try:
            from agent.anthropic_adapter import _HERMES_OAUTH_FILE
            if _HERMES_OAUTH_FILE.exists():
                _HERMES_OAUTH_FILE.unlink()
        except Exception:
            pass
        # Also clear the credential pool entry if present.
        try:
            from hermes_cli.auth import clear_provider_auth
            clear_provider_auth("anthropic")
        except Exception:
            pass
        _log.info("oauth/disconnect: %s", provider_id)
        return {"ok": True, "provider": provider_id}

    try:
        from hermes_cli.auth import clear_provider_auth
        cleared = clear_provider_auth(provider_id)
        _log.info("oauth/disconnect: %s (cleared=%s)", provider_id, cleared)
        return {"ok": bool(cleared), "provider": provider_id}
    except Exception as e:
        _log.exception("disconnect %s failed", provider_id)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# OAuth Phase 2 — in-browser PKCE & device-code flows
# ---------------------------------------------------------------------------
#
# Two flow shapes are supported:
#
#   PKCE (Anthropic):
#     1. POST /api/providers/oauth/anthropic/start
#          → server generates code_verifier + challenge, builds claude.ai
#            authorize URL, stashes verifier in _oauth_sessions[session_id]
#          → returns { session_id, flow: "pkce", auth_url }
#     2. UI opens auth_url in a new tab. User authorizes, copies code.
#     3. POST /api/providers/oauth/anthropic/submit { session_id, code }
#          → server exchanges (code + verifier) → tokens at console.anthropic.com
#          → persists to ~/.hermes/.anthropic_oauth.json AND credential pool
#          → returns { ok: true, status: "approved" }
#
#   Device code (Nous, OpenAI Codex):
#     1. POST /api/providers/oauth/{nous|openai-codex}/start
#          → server hits provider's device-auth endpoint
#          → gets { user_code, verification_url, device_code, interval, expires_in }
#          → spawns background poller thread that polls the token endpoint
#            every `interval` seconds until approved/expired
#          → stores poll status in _oauth_sessions[session_id]
#          → returns { session_id, flow: "device_code", user_code,
#                      verification_url, expires_in, poll_interval }
#     2. UI opens verification_url in a new tab and shows user_code.
#     3. UI polls GET /api/providers/oauth/{provider}/poll/{session_id}
#          every 2s until status != "pending".
#     4. On "approved" the background thread has already saved creds; UI
#        refreshes the providers list.
#
# Sessions are kept in-memory only (single-process FastAPI) and time out
# after 15 minutes. A periodic cleanup runs on each /start call to GC
# expired sessions so the dict doesn't grow without bound.

_OAUTH_SESSION_TTL_SECONDS = 15 * 60
_oauth_sessions: Dict[str, Dict[str, Any]] = {}
_oauth_sessions_lock = threading.Lock()

# Import OAuth constants from canonical source instead of duplicating.
# Guarded so hermes web still starts if anthropic_adapter is unavailable;
# Phase 2 endpoints will return 501 in that case.
try:
    from agent.anthropic_adapter import (
        _OAUTH_CLIENT_ID as _ANTHROPIC_OAUTH_CLIENT_ID,
        _OAUTH_TOKEN_URL as _ANTHROPIC_OAUTH_TOKEN_URL,
        _OAUTH_REDIRECT_URI as _ANTHROPIC_OAUTH_REDIRECT_URI,
        _OAUTH_SCOPES as _ANTHROPIC_OAUTH_SCOPES,
        _generate_pkce as _generate_pkce_pair,
    )
    _ANTHROPIC_OAUTH_AVAILABLE = True
except ImportError:
    _ANTHROPIC_OAUTH_AVAILABLE = False
_ANTHROPIC_OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"


def _gc_oauth_sessions() -> None:
    """Drop expired sessions. Called opportunistically on /start."""
    cutoff = time.time() - _OAUTH_SESSION_TTL_SECONDS
    with _oauth_sessions_lock:
        stale = [sid for sid, sess in _oauth_sessions.items() if sess["created_at"] < cutoff]
        for sid in stale:
            _oauth_sessions.pop(sid, None)


def _new_oauth_session(provider_id: str, flow: str) -> tuple[str, Dict[str, Any]]:
    """Create + register a new OAuth session, return (session_id, session_dict)."""
    sid = secrets.token_urlsafe(16)
    sess = {
        "session_id": sid,
        "provider": provider_id,
        "flow": flow,
        "created_at": time.time(),
        "status": "pending",  # pending | approved | denied | expired | error
        "error_message": None,
    }
    with _oauth_sessions_lock:
        _oauth_sessions[sid] = sess
    return sid, sess


def _save_anthropic_oauth_creds(access_token: str, refresh_token: str, expires_at_ms: int) -> None:
    """Persist Anthropic PKCE creds to both Hermes file AND credential pool.

    Mirrors what auth_commands.add_command does so the dashboard flow leaves
    the system in the same state as ``hermes auth add anthropic``.
    """
    from agent.anthropic_adapter import _HERMES_OAUTH_FILE
    payload = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at_ms,
    }
    _HERMES_OAUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HERMES_OAUTH_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Best-effort credential-pool insert. Failure here doesn't invalidate
    # the file write — pool registration only matters for the rotation
    # strategy, not for runtime credential resolution.
    try:
        from agent.credential_pool import (
            PooledCredential,
            load_pool,
            AUTH_TYPE_OAUTH,
            SOURCE_MANUAL,
        )
        import uuid
        pool = load_pool("anthropic")
        # Avoid duplicate entries: delete any prior dashboard-issued OAuth entry
        existing = [e for e in pool.entries() if getattr(e, "source", "").startswith(f"{SOURCE_MANUAL}:dashboard_pkce")]
        for e in existing:
            try:
                pool.remove_entry(getattr(e, "id", ""))
            except Exception:
                pass
        entry = PooledCredential(
            provider="anthropic",
            id=uuid.uuid4().hex[:6],
            label="dashboard PKCE",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:dashboard_pkce",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at_ms=expires_at_ms,
        )
        pool.add_entry(entry)
    except Exception as e:
        _log.warning("anthropic pool add (dashboard) failed: %s", e)


def _start_anthropic_pkce() -> Dict[str, Any]:
    """Begin PKCE flow. Returns the auth URL the UI should open."""
    if not _ANTHROPIC_OAUTH_AVAILABLE:
        raise HTTPException(status_code=501, detail="Anthropic OAuth not available (missing adapter)")
    verifier, challenge = _generate_pkce_pair()
    sid, sess = _new_oauth_session("anthropic", "pkce")
    sess["verifier"] = verifier
    sess["state"] = verifier  # Anthropic round-trips verifier as state
    params = {
        "code": "true",
        "client_id": _ANTHROPIC_OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _ANTHROPIC_OAUTH_REDIRECT_URI,
        "scope": _ANTHROPIC_OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    auth_url = f"{_ANTHROPIC_OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    return {
        "session_id": sid,
        "flow": "pkce",
        "auth_url": auth_url,
        "expires_in": _OAUTH_SESSION_TTL_SECONDS,
    }


def _submit_anthropic_pkce(session_id: str, code_input: str) -> Dict[str, Any]:
    """Exchange authorization code for tokens. Persists on success."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess or sess["provider"] != "anthropic" or sess["flow"] != "pkce":
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    if sess["status"] != "pending":
        return {"ok": False, "status": sess["status"], "message": sess.get("error_message")}

    # Anthropic's redirect callback page formats the code as `<code>#<state>`.
    # Strip the state suffix if present (we already have the verifier server-side).
    parts = code_input.strip().split("#", 1)
    code = parts[0].strip()
    if not code:
        return {"ok": False, "status": "error", "message": "No code provided"}
    state_from_callback = parts[1] if len(parts) > 1 else ""

    exchange_data = json.dumps({
        "grant_type": "authorization_code",
        "client_id": _ANTHROPIC_OAUTH_CLIENT_ID,
        "code": code,
        "state": state_from_callback or sess["state"],
        "redirect_uri": _ANTHROPIC_OAUTH_REDIRECT_URI,
        "code_verifier": sess["verifier"],
    }).encode()
    req = urllib.request.Request(
        _ANTHROPIC_OAUTH_TOKEN_URL,
        data=exchange_data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "hermes-dashboard/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        sess["status"] = "error"
        sess["error_message"] = f"Token exchange failed: {e}"
        return {"ok": False, "status": "error", "message": sess["error_message"]}

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = int(result.get("expires_in") or 3600)
    if not access_token:
        sess["status"] = "error"
        sess["error_message"] = "No access token returned"
        return {"ok": False, "status": "error", "message": sess["error_message"]}

    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    try:
        _save_anthropic_oauth_creds(access_token, refresh_token, expires_at_ms)
    except Exception as e:
        sess["status"] = "error"
        sess["error_message"] = f"Save failed: {e}"
        return {"ok": False, "status": "error", "message": sess["error_message"]}
    sess["status"] = "approved"
    _log.info("oauth/pkce: anthropic login completed (session=%s)", session_id)
    return {"ok": True, "status": "approved"}


async def _start_device_code_flow(provider_id: str) -> Dict[str, Any]:
    """Initiate a device-code flow (Nous or OpenAI Codex).

    Calls the provider's device-auth endpoint via the existing CLI helpers,
    then spawns a background poller. Returns the user-facing display fields
    so the UI can render the verification page link + user code.
    """
    from hermes_cli import auth as hauth
    if provider_id == "nous":
        from hermes_cli.auth import _request_device_code, PROVIDER_REGISTRY
        import httpx
        pconfig = PROVIDER_REGISTRY["nous"]
        portal_base_url = (
            os.getenv("HERMES_PORTAL_BASE_URL")
            or os.getenv("NOUS_PORTAL_BASE_URL")
            or pconfig.portal_base_url
        ).rstrip("/")
        client_id = pconfig.client_id
        scope = pconfig.scope
        def _do_nous_device_request():
            with httpx.Client(timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}) as client:
                return _request_device_code(
                    client=client,
                    portal_base_url=portal_base_url,
                    client_id=client_id,
                    scope=scope,
                )
        device_data = await asyncio.get_event_loop().run_in_executor(None, _do_nous_device_request)
        sid, sess = _new_oauth_session("nous", "device_code")
        sess["device_code"] = str(device_data["device_code"])
        sess["interval"] = int(device_data["interval"])
        sess["expires_at"] = time.time() + int(device_data["expires_in"])
        sess["portal_base_url"] = portal_base_url
        sess["client_id"] = client_id
        threading.Thread(
            target=_nous_poller, args=(sid,), daemon=True, name=f"oauth-poll-{sid[:6]}"
        ).start()
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": str(device_data["user_code"]),
            "verification_url": str(device_data["verification_uri_complete"]),
            "expires_in": int(device_data["expires_in"]),
            "poll_interval": int(device_data["interval"]),
        }

    if provider_id == "openai-codex":
        # Codex uses fixed OpenAI device-auth endpoints; reuse the helper.
        sid, _ = _new_oauth_session("openai-codex", "device_code")
        # Use the helper but in a thread because it polls inline.
        # We can't extract just the start step without refactoring auth.py,
        # so we run the full helper in a worker and proxy the user_code +
        # verification_url back via the session dict. The helper prints
        # to stdout — we capture nothing here, just status.
        threading.Thread(
            target=_codex_full_login_worker, args=(sid,), daemon=True,
            name=f"oauth-codex-{sid[:6]}",
        ).start()
        # Block briefly until the worker has populated the user_code, OR error.
        deadline = time.time() + 10
        while time.time() < deadline:
            with _oauth_sessions_lock:
                s = _oauth_sessions.get(sid)
            if s and (s.get("user_code") or s["status"] != "pending"):
                break
            await asyncio.sleep(0.1)
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(sid, {})
        if s.get("status") == "error":
            raise HTTPException(status_code=500, detail=s.get("error_message") or "device-auth failed")
        if not s.get("user_code"):
            raise HTTPException(status_code=504, detail="device-auth timed out before returning a user code")
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": s["user_code"],
            "verification_url": s["verification_url"],
            "expires_in": int(s.get("expires_in") or 900),
            "poll_interval": int(s.get("interval") or 5),
        }

    raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support device-code flow")


def _nous_poller(session_id: str) -> None:
    """Background poller that drives a Nous device-code flow to completion."""
    from hermes_cli.auth import _poll_for_token, refresh_nous_oauth_from_state
    from datetime import datetime, timezone
    import httpx
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        return
    portal_base_url = sess["portal_base_url"]
    client_id = sess["client_id"]
    device_code = sess["device_code"]
    interval = sess["interval"]
    expires_in = max(60, int(sess["expires_at"] - time.time()))
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}) as client:
            token_data = _poll_for_token(
                client=client,
                portal_base_url=portal_base_url,
                client_id=client_id,
                device_code=device_code,
                expires_in=expires_in,
                poll_interval=interval,
            )
        # Same post-processing as _nous_device_code_login (mint agent key)
        now = datetime.now(timezone.utc)
        token_ttl = int(token_data.get("expires_in") or 0)
        auth_state = {
            "portal_base_url": portal_base_url,
            "inference_base_url": token_data.get("inference_base_url"),
            "client_id": client_id,
            "scope": token_data.get("scope"),
            "token_type": token_data.get("token_type", "Bearer"),
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "obtained_at": now.isoformat(),
            "expires_at": (
                datetime.fromtimestamp(now.timestamp() + token_ttl, tz=timezone.utc).isoformat()
                if token_ttl else None
            ),
            "expires_in": token_ttl,
        }
        full_state = refresh_nous_oauth_from_state(
            auth_state, min_key_ttl_seconds=300, timeout_seconds=15.0,
            force_refresh=False, force_mint=True,
        )
        from hermes_cli.auth import persist_nous_credentials
        persist_nous_credentials(full_state)
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: nous login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("nous device-code poll failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = str(e)


def _codex_full_login_worker(session_id: str) -> None:
    """Run the complete OpenAI Codex device-code flow.

    Codex doesn't use the standard OAuth device-code endpoints; it has its
    own ``/api/accounts/deviceauth/usercode`` (JSON body, returns
    ``device_auth_id``) and ``/api/accounts/deviceauth/token`` (JSON body
    polled until 200). On success the response carries an
    ``authorization_code`` + ``code_verifier`` that get exchanged at
    CODEX_OAUTH_TOKEN_URL with grant_type=authorization_code.

    The flow is replicated inline (rather than calling
    _codex_device_code_login) because that helper prints/blocks/polls in a
    single function — we need to surface the user_code to the dashboard the
    moment we receive it, well before polling completes.
    """
    try:
        import httpx
        from hermes_cli.auth import (
            CODEX_OAUTH_CLIENT_ID,
            CODEX_OAUTH_TOKEN_URL,
            DEFAULT_CODEX_BASE_URL,
        )
        issuer = "https://auth.openai.com"

        # Step 1: request device code
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.post(
                f"{issuer}/api/accounts/deviceauth/usercode",
                json={"client_id": CODEX_OAUTH_CLIENT_ID},
                headers={"Content-Type": "application/json"},
            )
        if resp.status_code != 200:
            raise RuntimeError(f"deviceauth/usercode returned {resp.status_code}")
        device_data = resp.json()
        user_code = device_data.get("user_code", "")
        device_auth_id = device_data.get("device_auth_id", "")
        poll_interval = max(3, int(device_data.get("interval", "5")))
        if not user_code or not device_auth_id:
            raise RuntimeError("device-code response missing user_code or device_auth_id")
        verification_url = f"{issuer}/codex/device"
        with _oauth_sessions_lock:
            sess = _oauth_sessions.get(session_id)
            if not sess:
                return
            sess["user_code"] = user_code
            sess["verification_url"] = verification_url
            sess["device_auth_id"] = device_auth_id
            sess["interval"] = poll_interval
            sess["expires_in"] = 15 * 60  # OpenAI's effective limit
            sess["expires_at"] = time.time() + sess["expires_in"]

        # Step 2: poll until authorized
        deadline = time.time() + sess["expires_in"]
        code_resp = None
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            while time.time() < deadline:
                time.sleep(poll_interval)
                poll = client.post(
                    f"{issuer}/api/accounts/deviceauth/token",
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Content-Type": "application/json"},
                )
                if poll.status_code == 200:
                    code_resp = poll.json()
                    break
                if poll.status_code in (403, 404):
                    continue  # user hasn't authorized yet
                raise RuntimeError(f"deviceauth/token poll returned {poll.status_code}")

        if code_resp is None:
            with _oauth_sessions_lock:
                sess["status"] = "expired"
                sess["error_message"] = "Device code expired before approval"
            return

        # Step 3: exchange authorization_code for tokens
        authorization_code = code_resp.get("authorization_code", "")
        code_verifier = code_resp.get("code_verifier", "")
        if not authorization_code or not code_verifier:
            raise RuntimeError("device-auth response missing authorization_code/code_verifier")
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            token_resp = client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": f"{issuer}/deviceauth/callback",
                    "client_id": CODEX_OAUTH_CLIENT_ID,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if token_resp.status_code != 200:
            raise RuntimeError(f"token exchange returned {token_resp.status_code}")
        tokens = token_resp.json()
        access_token = tokens.get("access_token", "")
        refresh_token = tokens.get("refresh_token", "")
        if not access_token:
            raise RuntimeError("token exchange did not return access_token")

        # Persist via credential pool — same shape as auth_commands.add_command
        from agent.credential_pool import (
            PooledCredential,
            load_pool,
            AUTH_TYPE_OAUTH,
            SOURCE_MANUAL,
        )
        import uuid as _uuid
        pool = load_pool("openai-codex")
        base_url = (
            os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
            or DEFAULT_CODEX_BASE_URL
        )
        entry = PooledCredential(
            provider="openai-codex",
            id=_uuid.uuid4().hex[:6],
            label="dashboard device_code",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:dashboard_device_code",
            access_token=access_token,
            refresh_token=refresh_token,
            base_url=base_url,
        )
        pool.add_entry(entry)
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: openai-codex login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("codex device-code worker failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(session_id)
            if s:
                s["status"] = "error"
                s["error_message"] = str(e)


@app.post("/api/providers/oauth/{provider_id}/start")
async def start_oauth_login(provider_id: str, request: Request):
    """Initiate an OAuth login flow. Token-protected."""
    _require_token(request)
    _gc_oauth_sessions()
    valid = {p["id"] for p in _OAUTH_PROVIDER_CATALOG}
    if provider_id not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown provider {provider_id}")
    catalog_entry = next(p for p in _OAUTH_PROVIDER_CATALOG if p["id"] == provider_id)
    if catalog_entry["flow"] == "external":
        raise HTTPException(
            status_code=400,
            detail=f"{provider_id} uses an external CLI; run `{catalog_entry['cli_command']}` manually",
        )
    try:
        if catalog_entry["flow"] == "pkce":
            return _start_anthropic_pkce()
        if catalog_entry["flow"] == "device_code":
            return await _start_device_code_flow(provider_id)
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("oauth/start %s failed", provider_id)
        raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=400, detail="Unsupported flow")


class OAuthSubmitBody(BaseModel):
    session_id: str
    code: str


@app.post("/api/providers/oauth/{provider_id}/submit")
async def submit_oauth_code(provider_id: str, body: OAuthSubmitBody, request: Request):
    """Submit the auth code for PKCE flows. Token-protected."""
    _require_token(request)
    if provider_id == "anthropic":
        return await asyncio.get_event_loop().run_in_executor(
            None, _submit_anthropic_pkce, body.session_id, body.code,
        )
    raise HTTPException(status_code=400, detail=f"submit not supported for {provider_id}")


@app.get("/api/providers/oauth/{provider_id}/poll/{session_id}")
async def poll_oauth_session(provider_id: str, session_id: str):
    """Poll a device-code session's status (no auth — read-only state)."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if sess["provider"] != provider_id:
        raise HTTPException(status_code=400, detail="Provider mismatch for session")
    return {
        "session_id": session_id,
        "status": sess["status"],
        "error_message": sess.get("error_message"),
        "expires_at": sess.get("expires_at"),
    }


@app.delete("/api/providers/oauth/sessions/{session_id}")
async def cancel_oauth_session(session_id: str, request: Request):
    """Cancel a pending OAuth session. Token-protected."""
    _require_token(request)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.pop(session_id, None)
    if sess is None:
        return {"ok": False, "message": "session not found"}
    return {"ok": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# Session detail endpoints
# ---------------------------------------------------------------------------


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        sid = db.resolve_session_id(session_id)
        session = db.get_session(sid) if sid else None
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    finally:
        db.close()


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = db.get_messages(sid)
        return {"session_id": sid, "messages": messages}
    finally:
        db.close()


@app.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        if not db.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Log viewer endpoint
# ---------------------------------------------------------------------------


@app.get("/api/logs")
async def get_logs(
    file: str = "agent",
    lines: int = 100,
    level: Optional[str] = None,
    component: Optional[str] = None,
    search: Optional[str] = None,
):
    from hermes_cli.logs import _read_tail, LOG_FILES

    log_name = LOG_FILES.get(file)
    if not log_name:
        raise HTTPException(status_code=400, detail=f"Unknown log file: {file}")
    log_path = get_hermes_home() / "logs" / log_name
    if not log_path.exists():
        return {"file": file, "lines": []}

    try:
        from hermes_logging import COMPONENT_PREFIXES
    except ImportError:
        COMPONENT_PREFIXES = {}

    # Normalize "ALL" / "all" / empty → no filter. _matches_filters treats an
    # empty tuple as "must match a prefix" (startswith(()) is always False),
    # so passing () instead of None silently drops every line.
    min_level = level if level and level.upper() != "ALL" else None
    if component and component.lower() != "all":
        comp_prefixes = COMPONENT_PREFIXES.get(component)
        if comp_prefixes is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown component: {component}. "
                       f"Available: {', '.join(sorted(COMPONENT_PREFIXES))}",
            )
    else:
        comp_prefixes = None

    has_filters = bool(min_level or comp_prefixes or search)
    result = _read_tail(
        log_path, min(lines, 500) if not search else 2000,
        has_filters=has_filters,
        min_level=min_level,
        component_prefixes=comp_prefixes,
    )
    # Post-filter by search term (case-insensitive substring match).
    # _read_tail doesn't support free-text search, so we filter here and
    # trim to the requested line count afterward.
    if search:
        needle = search.lower()
        result = [l for l in result if needle in l.lower()][-min(lines, 500):]
    return {"file": file, "lines": result}


# ---------------------------------------------------------------------------
# Cron job management endpoints
# ---------------------------------------------------------------------------


class CronJobCreate(BaseModel):
    prompt: str
    schedule: str
    name: str = ""
    deliver: str = "local"


class CronJobUpdate(BaseModel):
    updates: dict


@app.get("/api/cron/jobs")
async def list_cron_jobs():
    from cron.jobs import list_jobs
    return list_jobs(include_disabled=True)


@app.get("/api/cron/jobs/{job_id}")
async def get_cron_job(job_id: str):
    from cron.jobs import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs")
async def create_cron_job(body: CronJobCreate):
    from cron.jobs import create_job
    try:
        job = create_job(prompt=body.prompt, schedule=body.schedule,
                         name=body.name, deliver=body.deliver)
        return job
    except Exception as e:
        _log.exception("POST /api/cron/jobs failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/cron/jobs/{job_id}")
async def update_cron_job(job_id: str, body: CronJobUpdate):
    from cron.jobs import update_job
    job = update_job(job_id, body.updates)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/pause")
async def pause_cron_job(job_id: str):
    from cron.jobs import pause_job
    job = pause_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/resume")
async def resume_cron_job(job_id: str):
    from cron.jobs import resume_job
    job = resume_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/trigger")
async def trigger_cron_job(job_id: str):
    from cron.jobs import trigger_job
    job = trigger_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/api/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str):
    from cron.jobs import remove_job
    if not remove_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Skills & Tools endpoints
# ---------------------------------------------------------------------------


class SkillToggle(BaseModel):
    name: str
    enabled: bool


@app.get("/api/skills")
async def get_skills():
    from tools.skills_tool import _find_all_skills
    from hermes_cli.skills_config import get_disabled_skills
    config = load_config()
    disabled = get_disabled_skills(config)
    skills = _find_all_skills(skip_disabled=True)
    for s in skills:
        s["enabled"] = s["name"] not in disabled
    return skills


@app.put("/api/skills/toggle")
async def toggle_skill(body: SkillToggle):
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills
    config = load_config()
    disabled = get_disabled_skills(config)
    if body.enabled:
        disabled.discard(body.name)
    else:
        disabled.add(body.name)
    save_disabled_skills(config, disabled)
    return {"ok": True, "name": body.name, "enabled": body.enabled}


@app.get("/api/tools/toolsets")
async def get_toolsets():
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets,
        _get_platform_tools,
        _toolset_has_keys,
    )
    from toolsets import resolve_toolset

    config = load_config()
    enabled_toolsets = _get_platform_tools(
        config,
        "cli",
        include_default_mcp_servers=False,
    )
    result = []
    for name, label, desc in _get_effective_configurable_toolsets():
        try:
            tools = sorted(set(resolve_toolset(name)))
        except Exception:
            tools = []
        is_enabled = name in enabled_toolsets
        result.append({
            "name": name, "label": label, "description": desc,
            "enabled": is_enabled,
            "available": is_enabled,
            "configured": _toolset_has_keys(name, config),
            "tools": tools,
        })
    return result


# ---------------------------------------------------------------------------
# Raw YAML config endpoint
# ---------------------------------------------------------------------------


class RawConfigUpdate(BaseModel):
    yaml_text: str


@app.get("/api/config/raw")
async def get_config_raw():
    path = get_config_path()
    if not path.exists():
        return {"yaml": ""}
    return {"yaml": path.read_text(encoding="utf-8")}


@app.put("/api/config/raw")
async def update_config_raw(body: RawConfigUpdate):
    try:
        parsed = yaml.safe_load(body.yaml_text)
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="YAML must be a mapping")
        save_config(parsed)
        return {"ok": True}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")


# ---------------------------------------------------------------------------
# Token / cost analytics endpoint
# ---------------------------------------------------------------------------


@app.get("/api/analytics/usage")
async def get_usage_analytics(days: int = 30):
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        cutoff = time.time() - (days * 86400)
        cur = db._conn.execute("""
            SELECT date(started_at, 'unixepoch') as day,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions
            FROM sessions WHERE started_at > ?
            GROUP BY day ORDER BY day
        """, (cutoff,))
        daily = [dict(r) for r in cur.fetchall()]

        cur2 = db._conn.execute("""
            SELECT model,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COUNT(*) as sessions
            FROM sessions WHERE started_at > ? AND model IS NOT NULL
            GROUP BY model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, (cutoff,))
        by_model = [dict(r) for r in cur2.fetchall()]

        cur3 = db._conn.execute("""
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions
            FROM sessions WHERE started_at > ?
        """, (cutoff,))
        totals = dict(cur3.fetchone())

        return {"daily": daily, "by_model": by_model, "totals": totals, "period_days": days}
    finally:
        db.close()


def mount_spa(application: FastAPI):
    """Mount the built SPA. Falls back to index.html for client-side routing.

    The session token is injected into index.html via a ``<script>`` tag so
    the SPA can authenticate against protected API endpoints without a
    separate (unauthenticated) token-dispensing endpoint.
    """
    if not WEB_DIST.exists():
        @application.get("/{full_path:path}")
        async def no_frontend(full_path: str):
            return JSONResponse(
                {"error": "Frontend not built. Run: cd web && npm run build"},
                status_code=404,
            )
        return

    _index_path = WEB_DIST / "index.html"

    def _serve_index():
        """Return index.html with the session token injected."""
        html = _index_path.read_text()
        token_script = (
            f'<script>window.__HERMES_SESSION_TOKEN__="{_SESSION_TOKEN}";</script>'
        )
        html = html.replace("</head>", f"{token_script}</head>", 1)
        return HTMLResponse(
            html,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    application.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @application.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = WEB_DIST / full_path
        # Prevent path traversal via url-encoded sequences (%2e%2e/)
        if (
            full_path
            and file_path.resolve().is_relative_to(WEB_DIST.resolve())
            and file_path.exists()
            and file_path.is_file()
        ):
            return FileResponse(file_path)
        return _serve_index()


# ---------------------------------------------------------------------------
# Dashboard theme endpoints
# ---------------------------------------------------------------------------

# Built-in dashboard themes — label + description only.  The actual color
# definitions live in the frontend (web/src/themes/presets.ts).
_BUILTIN_DASHBOARD_THEMES = [
    {"name": "default",   "label": "Hermes Teal",  "description": "Classic dark teal — the canonical Hermes look"},
    {"name": "midnight",  "label": "Midnight",      "description": "Deep blue-violet with cool accents"},
    {"name": "ember",     "label": "Ember",          "description": "Warm crimson and bronze — forge vibes"},
    {"name": "mono",      "label": "Mono",           "description": "Clean grayscale — minimal and focused"},
    {"name": "cyberpunk", "label": "Cyberpunk",      "description": "Neon green on black — matrix terminal"},
    {"name": "rose",      "label": "Rosé",           "description": "Soft pink and warm ivory — easy on the eyes"},
]


def _discover_user_themes() -> list:
    """Scan ~/.hermes/dashboard-themes/*.yaml for user-created themes."""
    themes_dir = get_hermes_home() / "dashboard-themes"
    if not themes_dir.is_dir():
        return []
    result = []
    for f in sorted(themes_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("name"):
                result.append({
                    "name": data["name"],
                    "label": data.get("label", data["name"]),
                    "description": data.get("description", ""),
                })
        except Exception:
            continue
    return result


@app.get("/api/dashboard/themes")
async def get_dashboard_themes():
    """Return available themes and the currently active one."""
    config = load_config()
    active = config.get("dashboard", {}).get("theme", "default")
    user_themes = _discover_user_themes()
    # Merge built-in + user, user themes override built-in by name.
    seen = set()
    themes = []
    for t in _BUILTIN_DASHBOARD_THEMES:
        seen.add(t["name"])
        themes.append(t)
    for t in user_themes:
        if t["name"] not in seen:
            themes.append(t)
            seen.add(t["name"])
    return {"themes": themes, "active": active}


class ThemeSetBody(BaseModel):
    name: str


@app.put("/api/dashboard/theme")
async def set_dashboard_theme(body: ThemeSetBody):
    """Set the active dashboard theme (persists to config.yaml)."""
    config = load_config()
    if "dashboard" not in config:
        config["dashboard"] = {}
    config["dashboard"]["theme"] = body.name
    save_config(config)
    return {"ok": True, "theme": body.name}


# ---------------------------------------------------------------------------
# Dashboard plugin system
# ---------------------------------------------------------------------------

def _discover_dashboard_plugins() -> list:
    """Scan plugins/*/dashboard/manifest.json for dashboard extensions.

    Checks three plugin sources (same as hermes_cli.plugins):
    1. User plugins:    ~/.hermes/plugins/<name>/dashboard/manifest.json
    2. Bundled plugins: <repo>/plugins/<name>/dashboard/manifest.json  (memory/, etc.)
    3. Project plugins: ./.hermes/plugins/  (only if HERMES_ENABLE_PROJECT_PLUGINS)
    """
    plugins = []
    seen_names: set = set()

    search_dirs = [
        (get_hermes_home() / "plugins", "user"),
        (PROJECT_ROOT / "plugins" / "memory", "bundled"),
        (PROJECT_ROOT / "plugins", "bundled"),
    ]
    if os.environ.get("HERMES_ENABLE_PROJECT_PLUGINS"):
        search_dirs.append((Path.cwd() / ".hermes" / "plugins", "project"))

    for plugins_root, source in search_dirs:
        if not plugins_root.is_dir():
            continue
        for child in sorted(plugins_root.iterdir()):
            if not child.is_dir():
                continue
            manifest_file = child / "dashboard" / "manifest.json"
            if not manifest_file.exists():
                continue
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                name = data.get("name", child.name)
                if name in seen_names:
                    continue
                seen_names.add(name)
                plugins.append({
                    "name": name,
                    "label": data.get("label", name),
                    "description": data.get("description", ""),
                    "icon": data.get("icon", "Puzzle"),
                    "version": data.get("version", "0.0.0"),
                    "tab": data.get("tab", {"path": f"/{name}", "position": "end"}),
                    "entry": data.get("entry", "dist/index.js"),
                    "css": data.get("css"),
                    "has_api": bool(data.get("api")),
                    "source": source,
                    "_dir": str(child / "dashboard"),
                    "_api_file": data.get("api"),
                })
            except Exception as exc:
                _log.warning("Bad dashboard plugin manifest %s: %s", manifest_file, exc)
                continue
    return plugins


# Cache discovered plugins per-process (refresh on explicit re-scan).
_dashboard_plugins_cache: Optional[list] = None


def _get_dashboard_plugins(force_rescan: bool = False) -> list:
    global _dashboard_plugins_cache
    if _dashboard_plugins_cache is None or force_rescan:
        _dashboard_plugins_cache = _discover_dashboard_plugins()
    return _dashboard_plugins_cache


@app.get("/api/dashboard/plugins")
async def get_dashboard_plugins():
    """Return discovered dashboard plugins."""
    plugins = _get_dashboard_plugins()
    # Strip internal fields before sending to frontend.
    return [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in plugins
    ]


@app.get("/api/dashboard/plugins/rescan")
async def rescan_dashboard_plugins():
    """Force re-scan of dashboard plugins."""
    plugins = _get_dashboard_plugins(force_rescan=True)
    return {"ok": True, "count": len(plugins)}


@app.get("/dashboard-plugins/{plugin_name}/{file_path:path}")
async def serve_plugin_asset(plugin_name: str, file_path: str):
    """Serve static assets from a dashboard plugin directory.

    Only serves files from the plugin's ``dashboard/`` subdirectory.
    Path traversal is blocked by checking ``resolve().is_relative_to()``.
    """
    plugins = _get_dashboard_plugins()
    plugin = next((p for p in plugins if p["name"] == plugin_name), None)
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    base = Path(plugin["_dir"])
    target = (base / file_path).resolve()

    if not target.is_relative_to(base.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal blocked")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Guess content type
    suffix = target.suffix.lower()
    content_types = {
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".html": "text/html",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
    }
    media_type = content_types.get(suffix, "application/octet-stream")
    return FileResponse(target, media_type=media_type)


def _mount_plugin_api_routes():
    """Import and mount backend API routes from plugins that declare them.

    Each plugin's ``api`` field points to a Python file that must expose
    a ``router`` (FastAPI APIRouter).  Routes are mounted under
    ``/api/plugins/<name>/``.
    """
    for plugin in _get_dashboard_plugins():
        api_file_name = plugin.get("_api_file")
        if not api_file_name:
            continue
        api_path = Path(plugin["_dir"]) / api_file_name
        if not api_path.exists():
            _log.warning("Plugin %s declares api=%s but file not found", plugin["name"], api_file_name)
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"hermes_dashboard_plugin_{plugin['name']}", api_path,
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            router = getattr(mod, "router", None)
            if router is None:
                _log.warning("Plugin %s api file has no 'router' attribute", plugin["name"])
                continue
            app.include_router(router, prefix=f"/api/plugins/{plugin['name']}")
            _log.info("Mounted plugin API routes: /api/plugins/%s/", plugin["name"])
        except Exception as exc:
            _log.warning("Failed to load plugin %s API routes: %s", plugin["name"], exc)


# Mount plugin API routes before the SPA catch-all.
_mount_plugin_api_routes()

# 注意：mount_spa 包含一个 catch-all GET 路由 `/{full_path:path}`，会吞掉所有
# 未匹配到具体路径的 GET 请求（用于 SPA 客户端路由）。所以下面的 chat 路由
# 必须在 mount_spa 之前注册，否则 GET /api/chat/resume 会被当成 SPA 路由。


# ---------------------------------------------------------------------------
# Chat endpoint —— SSE 流式 + 强接续（断点续传）
#
# 设计要点（详细背景见 chat_api.py 的注释和 doc/工作记录-2026年0429.md）：
#   1) /api/chat/stream  POST：SSE 推 token，第一帧 meta 带 message_id；
#   2) /api/chat/resume  GET ：按 last_seq 或 Last-Event-ID 续传；
#   3) 同 session 一把 asyncio.Lock 互斥，避免 AIAgent 内部状态竞态；
#   4) stream_delta_callback 在 to_thread 工作线程里被调，必须用
#      loop.call_soon_threadsafe 把数据投回主 loop；
#   5) 老的同步 /api/chat 已移除（前端已改造为 SharedWorker + SSE）。
# ---------------------------------------------------------------------------

import itertools  # noqa: E402
import uuid  # noqa: E402
from typing import AsyncGenerator, Tuple  # noqa: E402

from fastapi.responses import StreamingResponse  # noqa: E402

from run_agent import AIAgent  # noqa: E402
from hermes_state import SessionDB  # noqa: E402

_chat_agent: "Optional[AIAgent]" = None
_chat_session_id: "Optional[str]" = None
_chat_session_db: "Optional[SessionDB]" = None

# 每个 session 一把异步锁
_chat_session_locks: Dict[str, asyncio.Lock] = {}

# Resume 轮询间隔
_CHAT_RESUME_POLL_INTERVAL = 0.2


class ChatStreamRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    # 前端可预生成 message_id；不传则后端 uuid4，方便失败时 resume
    message_id: Optional[str] = None


def _chat_get_db() -> SessionDB:
    global _chat_session_db
    if _chat_session_db is None:
        _chat_session_db = SessionDB()
    return _chat_session_db


def _chat_get_session_lock(session_id: str) -> asyncio.Lock:
    lock = _chat_session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_session_locks[session_id] = lock
    return lock


def _chat_get_agent(session_id: Optional[str] = None) -> Tuple[AIAgent, str]:
    global _chat_agent, _chat_session_id
    _chat_get_db()  # 确保 db 初始化
    config = load_config()
    system_prompt = (config.get("agent") or {}).get("system_prompt") or ""
    # model 可能是裸字符串或 dict，统一处理两种格式
    model_cfg = config.get("model") or {}
    if isinstance(model_cfg, dict):
        model_name = model_cfg.get("default", "deepseek-v4-pro")
    else:
        model_name = str(model_cfg) if model_cfg else "deepseek-v4-pro"
    if _chat_agent is None or session_id != _chat_session_id:
        _chat_agent = AIAgent(
            model=model_name,
            session_id=session_id,
            save_trajectories=True,
            session_db=_chat_get_db(),
            ephemeral_system_prompt=system_prompt or None,
        )
        _chat_session_id = _chat_agent.session_id
    else:
        # 每次请求都更新 system_prompt，避免缓存旧值导致配置修改不生效
        _chat_agent.ephemeral_system_prompt = system_prompt or None
    return _chat_agent, _chat_session_id


def _sse_pack(event: str, data: dict, seq: Optional[int] = None) -> str:
    """组装 SSE 帧。带 id 时浏览器会自动写入 Last-Event-ID。"""
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _chat_run_stream(
    user_message: str,
    session_id_in: Optional[str],
    message_id: str,
) -> AsyncGenerator[str, None]:
    """流式驱动 AIAgent，把 token 逐帧 yield 出去。"""
    db = _chat_get_db()
    agent, sid = _chat_get_agent(session_id_in)

    # 登记本次流（status=running）
    db.create_message_stream(message_id, sid, user_message)

    # 第一帧：把 message_id / session_id 告诉前端
    yield _sse_pack("meta", {"message_id": message_id, "session_id": sid})

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    seq_counter = itertools.count()

    # 同步回调 -> 异步队列：必须 call_soon_threadsafe，遵守事件循环绑定规则
    def _on_delta(text):
        if not text:
            return
        s = next(seq_counter)
        loop.call_soon_threadsafe(queue.put_nowait, ("token", s, text))

    prev_cb = getattr(agent, "stream_delta_callback", None)
    agent.stream_delta_callback = _on_delta

    async def _persist_chunk(s: int, text: str):
        try:
            await asyncio.to_thread(db.append_message_chunk, message_id, s, text, "token")
        except Exception as exc:  # noqa: BLE001
            _log.warning("persist chat chunk failed seq=%d: %s", s, exc)

    sess_lock = _chat_get_session_lock(sid)
    await sess_lock.acquire()  # 同 session 串行
    try:
        agent_task = asyncio.create_task(
            asyncio.to_thread(agent.run_conversation, user_message)
        )
        try:
            while True:
                getter = asyncio.create_task(queue.get())
                done, _pending = await asyncio.wait(
                    {getter, agent_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter in done:
                    _kind, s, text = getter.result()
                    yield _sse_pack("token", {"delta": text}, seq=s)
                    asyncio.create_task(_persist_chunk(s, text))
                else:
                    getter.cancel()

                if agent_task.done():
                    # 抽干残留 token
                    while not queue.empty():
                        _kind, s, text = queue.get_nowait()
                        yield _sse_pack("token", {"delta": text}, seq=s)
                        asyncio.create_task(_persist_chunk(s, text))

                    if agent_task.exception() is not None:
                        err = repr(agent_task.exception())
                        await asyncio.to_thread(
                            db.finish_message_stream, message_id, "error", None, err
                        )
                        yield _sse_pack("error", {"error": err})
                    else:
                        result = agent_task.result() or {}
                        final = result.get("final_response", "") or ""
                        await asyncio.to_thread(
                            db.finish_message_stream, message_id, "done", final, None
                        )
                        yield _sse_pack("done", {"final": final})
                    return
        except asyncio.CancelledError:
            await asyncio.to_thread(
                db.finish_message_stream, message_id, "aborted", None, "client disconnected"
            )
            raise
    finally:
        agent.stream_delta_callback = prev_cb
        sess_lock.release()


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatStreamRequest):
    """SSE 流式聊天入口。"""
    message_id = req.message_id or uuid.uuid4().hex
    return StreamingResponse(
        _chat_run_stream(req.message, req.session_id, message_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关 nginx 缓冲
            "X-Message-Id": message_id,
        },
    )


async def _chat_run_resume(
    message_id: str, last_seq: int
) -> AsyncGenerator[str, None]:
    db = _chat_get_db()
    info = await asyncio.to_thread(db.get_message_stream, message_id)
    if info is None:
        yield _sse_pack("error", {"error": f"message_id not found: {message_id}"})
        return

    yield _sse_pack(
        "meta",
        {"message_id": message_id, "session_id": info["session_id"], "resume": True},
    )

    cursor = last_seq
    chunks = await asyncio.to_thread(db.get_chunks_after, message_id, cursor)
    for seq, delta, _kind in chunks:
        yield _sse_pack("token", {"delta": delta}, seq=seq)
        cursor = seq

    while True:
        info = await asyncio.to_thread(db.get_message_stream, message_id)
        if info is None:
            yield _sse_pack("error", {"error": "stream vanished"})
            return
        chunks = await asyncio.to_thread(db.get_chunks_after, message_id, cursor)
        for seq, delta, _kind in chunks:
            yield _sse_pack("token", {"delta": delta}, seq=seq)
            cursor = seq

        status = info["status"]
        if status == "running":
            await asyncio.sleep(_CHAT_RESUME_POLL_INTERVAL)
            continue
        if status == "done":
            yield _sse_pack("done", {"final": info.get("final_text") or ""})
            return
        if status in ("error", "aborted"):
            yield _sse_pack(
                "error",
                {"error": info.get("error") or f"stream {status}"},
            )
            return
        yield _sse_pack("error", {"error": f"unknown status: {status}"})
        return


@app.get("/api/chat/resume")
async def chat_resume_endpoint(
    message_id: str,
    last_seq: int = -1,
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
):
    """SSE 断点续传。Last-Event-ID 优先于 last_seq。"""
    if last_event_id is not None:
        try:
            last_seq = int(last_event_id)
        except ValueError:
            pass
    return StreamingResponse(
        _chat_run_resume(message_id, last_seq),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Voice WebSocket endpoint —— VAD 驱动的实时语音对话
#
# 协议（JSON 文本帧）:
#   客户端 → 服务端:
#     {"type": "voice_start"}          进入语音模式，开始持续推送 PCM
#     {"type": "voice_stop"}           退出语音模式
#
#   服务端 → 客户端:
#     {"type": "status", "status": "listening|thinking|speaking|idle|error"}
#     {"type": "vad", "status": "speech_start|speech_end"}
#     {"type": "transcription", "text": "…"}  STT 转写结果
#     {"type": "token", "delta": "…"}        文本增量
#     {"type": "done", "final": "…"}         回复完成
#     {"type": "interrupted"}                被打断通知
#     {"type": "error", "error": "…"}        错误
#
# 二进制帧:
#   客户端 → 服务端: PCM 音频数据（16kHz, 16bit, mono）
#   服务端 → 客户端: TTS 音频数据（MP3 bytes）
#
# 状态机: LISTENING → THINKING → SPEAKING → LISTENING（循环）
#   - VAD 检测到人声结束 → 触发 STT + Agent + TTS
#   - SPEAKING 期间检测到新人声 → 打断，回到 LISTENING
#
# 设计原则:
#   - 音频不做任何持久存储，临时文件用完即删
#   - 复用现有 AIAgent 和 stream_delta_callback 机制
#   - VAD 引擎在服务端运行（silero-vad），前端只负责 PCM 采集
# ---------------------------------------------------------------------------

import asyncio as _asyncio
import json as _json
import os as _os
import queue as _queue
import tempfile as _tempfile
import threading as _threading
import uuid as _uuid

from fastapi import WebSocket as _WebSocket, WebSocketDisconnect as _WebSocketDisconnect

# silero-vad 要求的帧长：512 samples @ 16kHz = 32ms = 1024 bytes
_VAD_FRAME_SAMPLES = 512
_VAD_FRAME_BYTES = _VAD_FRAME_SAMPLES * 2  # 16bit = 2 bytes/sample

# 前端发送的帧长：320 samples @ 16kHz = 20ms = 640 bytes
_PCM_CHUNK_BYTES = 640

# 语音段最短/最长限制
_MIN_SPEECH_BYTES = 16000   # 0.5 秒
_MAX_SPEECH_BYTES = 480000  # 15 秒
_MIN_SPEECH_RMS = 350       # 低于该 RMS 认为是底噪/误触发

# TTS 每积累多少字触发一次
_TTS_CHUNK_SIZE = 80

_VOICE_SESSIONS: Dict[str, dict] = {}
"""session_id → {
    'vad_engine': VADEngine,
    'pcm_buffer': bytearray,     # 累积 PCM 直到凑够 VAD 帧长
    'speech_audio': bytearray,   # 当前语音段的 PCM 数据
    'agent_task': asyncio.Task | None,
    'interrupt_event': threading.Event | None,
    'state': 'idle' | 'listening' | 'thinking' | 'speaking',
}"""


@app.websocket("/api/voice/ws")
async def voice_websocket_endpoint(ws: _WebSocket):
    """WebSocket VAD 驱动的实时语音对话端点。

    客户端连接后发送 voice_start 进入语音模式，
    然后持续推送 PCM 音频帧。服务端用 silero-vad 检测人声边界，
    人声结束后自动触发 STT → Agent → TTS 全链路。
    """
    await ws.accept()
    session_id = _uuid.uuid4().hex
    _log.info("[voice] WebSocket 连接建立 session=%s", session_id)

    # ---- 辅助函数 ----

    async def _send_json(data: dict):
        try:
            await ws.send_text(_json.dumps(data, ensure_ascii=False))
        except Exception:
            pass

    async def _send_audio(audio_bytes: bytes):
        try:
            _log.info("[voice] send audio bytes=%d", len(audio_bytes))
            await ws.send_bytes(audio_bytes)
        except Exception as e:
            _log.warning("[voice] send audio failed: %s", e)

    async def _set_status(status: str):
        await _send_json({"type": "status", "status": status})

    def _run_stt(pcm_bytes: bytes) -> dict:
        """在 to_thread 中执行 STT。

        统一复用项目现有 STT 分发链路，让 web 语音跟随当前配置：
        当前环境下会走本地 faster-whisper，避免阿里云 404 和无效 OpenAI key 401。
        """
        from tools.aliyun_voice import pcm_to_wav
        from tools.transcription_tools import transcribe_audio

        wav_bytes = pcm_to_wav(pcm_bytes)
        tmp_path = None
        try:
            tmp = _tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(wav_bytes)
            tmp.close()
            tmp_path = tmp.name
            return transcribe_audio(tmp_path)
        finally:
            if tmp_path:
                try:
                    _os.unlink(tmp_path)
                except OSError:
                    pass

    def _run_tts(text: str) -> dict:
        """在 to_thread 中执行 TTS。

        优先走本地已安装的 edge-tts，避免当前阿里云 TTS 404。
        """
        tts_text = (text or "").strip()
        if not tts_text:
            return {"success": False, "error": "TTS 文字为空"}

        _log.info("[voice] TTS start chars=%d preview=%r", len(tts_text), tts_text[:40])

        # 读取 TTS 配置
        from hermes_cli.config import load_config as _load_config
        _cfg = _load_config()
        tts_config = _cfg.get("tts", {})
        provider = (tts_config.get("provider") or "edge").lower().strip()
        _log.info("[voice] TTS provider=%s", provider)

        try:
            if provider == "aliyun":
                return _tts_aliyun(tts_text, _cfg)
            elif provider == "openai":
                return _tts_openai(tts_text, tts_config)
            elif provider == "elevenlabs":
                return _tts_elevenlabs(tts_text, tts_config)
            elif provider == "neutts":
                return _tts_neutts(tts_text, tts_config)
            else:
                return _tts_edge(tts_text, tts_config)
        except Exception as e:
            _log.warning("[voice] TTS provider=%s failed: %s, 回退到 edge-tts", provider, e)
            try:
                return _tts_edge(tts_text, tts_config)
            except Exception as e2:
                _log.error("[voice] TTS edge 回退也失败: %s", e2)
                return {"success": False, "error": str(e2)}

    def _tts_edge(tts_text: str, tts_config: dict) -> dict:
        """Edge TTS —— 免费，无需 API Key，作为默认和回退 provider。"""
        import edge_tts as _edge_tts

        edge_config = tts_config.get("edge", {})
        voice = edge_config.get("voice") or "zh-CN-XiaoxiaoNeural"

        # 如果配置的 voice 语言与文本语言不匹配，自动切换
        has_cjk = any("一" <= ch <= "鿿" for ch in tts_text)
        voice_is_cjk = voice.startswith("zh-")
        if has_cjk and not voice_is_cjk:
            voice = "zh-CN-XiaoxiaoNeural"
        elif not has_cjk and voice_is_cjk:
            voice = "en-US-AriaNeural"

        tmp = _tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            async def _gen():
                communicate = _edge_tts.Communicate(tts_text, voice=voice)
                await communicate.save(tmp_path)
            _asyncio.run(_gen())
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            _log.info("[voice] TTS edge success voice=%s bytes=%d", voice, len(audio_data))
            return {"success": True, "audio_data": audio_data, "format": "mp3"}
        finally:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass

    def _tts_aliyun(tts_text: str, config: dict) -> dict:
        """阿里云 DashScope CosyVoice TTS。读 voice.aliyun.* 配置。"""
        from tools.aliyun_voice import aliyun_tts
        result = aliyun_tts(tts_text)
        if result.get("success"):
            _log.info("[voice] TTS aliyun success bytes=%d", len(result.get("audio_data", b"")))
        else:
            _log.warning("[voice] TTS aliyun failed: %s", result.get("error"))
        return result

    def _tts_openai(tts_text: str, tts_config: dict) -> dict:
        """OpenAI TTS。读 tts.openai.model / tts.openai.voice 配置，需要 OPENAI_API_KEY。"""
        import openai as _openai

        openai_config = tts_config.get("openai", {})
        model = openai_config.get("model") or "gpt-4o-mini-tts"
        voice = openai_config.get("voice") or "alloy"
        api_key = _os.getenv("OPENAI_API_KEY", "")
        base_url = _os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置")

        client = _openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.audio.speech.create(model=model, voice=voice, input=tts_text, response_format="mp3")
        audio_data = response.content
        _log.info("[voice] TTS openai success model=%s voice=%s bytes=%d", model, voice, len(audio_data))
        return {"success": True, "audio_data": audio_data, "format": "mp3"}

    def _tts_elevenlabs(tts_text: str, tts_config: dict) -> dict:
        """ElevenLabs TTS。读 tts.elevenlabs.* 配置，需要 ELEVENLABS_API_KEY。"""
        from elevenlabs.client import ElevenLabs as _ElevenLabs

        el_config = tts_config.get("elevenlabs", {})
        voice_id = el_config.get("voice_id") or "pNInz6obpgDQGcFmaJgB"
        model_id = el_config.get("model_id") or "eleven_multilingual_v2"
        api_key = _os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY 未设置")

        client = _ElevenLabs(api_key=api_key)
        audio_gen = client.text_to_speech.convert(text=tts_text, voice_id=voice_id, model_id=model_id, output_format="mp3_44100_128")
        chunks = []
        for chunk in audio_gen:
            if chunk:
                chunks.append(chunk)
        audio_data = b"".join(chunks)
        _log.info("[voice] TTS elevenlabs success voice=%s model=%s bytes=%d", voice_id, model_id, len(audio_data))
        return {"success": True, "audio_data": audio_data, "format": "mp3"}

    def _tts_neutts(tts_text: str, tts_config: dict) -> dict:
        """NeuTTS 本地合成。读 tts.neutts.* 配置，不需要网络。"""
        neutts_config = tts_config.get("neutts", {})
        model = neutts_config.get("model") or "neuphonic/neutts-air-q4-gguf"
        device = neutts_config.get("device") or "cpu"
        ref_audio = neutts_config.get("ref_audio") or ""
        ref_text = neutts_config.get("ref_text") or ""

        import neutts as _neutts

        tmp = _tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            synth = _neutts.NeuTTS(model=model, device=device, ref_audio=ref_audio or None, ref_text=ref_text or None)
            synth.synthesize(tts_text, tmp_path)
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            _log.info("[voice] TTS neutts success bytes=%d", len(audio_data))
            return {"success": True, "audio_data": audio_data, "format": "wav"}
        finally:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
            import edge_tts

            # 中英文本做一个简单的默认 voice 选择
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in tts_text)
            voice = "zh-CN-XiaoxiaoNeural" if has_cjk else "en-US-AriaNeural"

            tmp = _tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_path = tmp.name
            tmp.close()

            async def _generate():
                communicate = edge_tts.Communicate(tts_text, voice=voice)
                await communicate.save(tmp_path)

            _asyncio.run(_generate())
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
            _log.info("[voice] TTS edge success voice=%s bytes=%d", voice, len(audio_data))
            return {"success": True, "audio_data": audio_data, "format": "mp3"}
    def _run_agent_sync(
        user_text: str,
        sid: str,
        delta_queue: _queue.Queue,
        interrupt_event: _threading.Event,
    ) -> dict:
        """在 to_thread 中运行 AIAgent，通过 delta_queue 推送 token。"""
        from run_agent import AIAgent
        from hermes_state import SessionDB
        from hermes_cli.config import load_config

        config = load_config()
        db = SessionDB()
        system_prompt = (config.get("agent") or {}).get("system_prompt") or ""
        model_cfg = config.get("model") or {}
        if isinstance(model_cfg, dict):
            model_name = model_cfg.get("default", "deepseek-v4-pro")
        else:
            model_name = str(model_cfg) if model_cfg else "deepseek-v4-pro"
        agent = AIAgent(
            model=model_name,
            session_id=sid,
            save_trajectories=False,
            session_db=db,
            ephemeral_system_prompt=system_prompt or None,
        )

        def _on_delta(text: str):
            if text:
                delta_queue.put(("token", text))

        agent.stream_delta_callback = _on_delta

        result = {}
        error = None

        def _run():
            nonlocal result, error
            try:
                result = agent.run_conversation(user_text)
            except Exception as e:
                error = str(e)

        thread = _threading.Thread(target=_run, daemon=True)
        thread.start()

        while thread.is_alive():
            thread.join(timeout=0.1)
            if interrupt_event.is_set():
                try:
                    agent.interrupt("voice user interrupt")
                except Exception:
                    pass
                thread.join(timeout=5)
                delta_queue.put(("interrupted", ""))
                return {"final_response": "", "interrupted": True}

        if error:
            delta_queue.put(("error", error))
            return {"final_response": "", "error": error}

        return result

    async def _handle_speech_end(session_data: dict):
        """人声结束：触发 STT → Agent → TTS 全链路。"""
        speech_audio = bytes(session_data["speech_audio"])
        session_data["speech_audio"] = bytearray()

        # 检查语音段长度
        if len(speech_audio) < _MIN_SPEECH_BYTES:
            _log.debug("[voice] 语音段太短 (%d bytes)，丢弃", len(speech_audio))
            session_data["state"] = "listening"
            return

        if len(speech_audio) > _MAX_SPEECH_BYTES:
            _log.debug("[voice] 语音段过长 (%d bytes)，截断", len(speech_audio))
            speech_audio = speech_audio[:_MAX_SPEECH_BYTES]

        # 误触发过滤：低能量底噪不送 STT
        if speech_audio:
            import numpy as _np
            _samples = _np.frombuffer(speech_audio, dtype=_np.int16).astype(_np.float32)
            speech_rms = int(_np.sqrt(_np.mean(_samples * _samples))) if len(_samples) else 0
        else:
            speech_rms = 0
        if speech_rms < _MIN_SPEECH_RMS:
            _log.info("[voice] 忽略低能量语音段: rms=%d, bytes=%d", speech_rms, len(speech_audio))
            session_data["state"] = "listening"
            return

        # 第一步：STT 转写
        session_data["state"] = "thinking"
        await _set_status("thinking")
        stt_result = await _asyncio.to_thread(_run_stt, speech_audio)

        if not stt_result.get("success"):
            await _send_json({
                "type": "error",
                "error": f"语音识别失败: {stt_result.get('error', '未知错误')}"
            })
            session_data["state"] = "listening"
            await _set_status("listening")
            return

        transcript = (stt_result.get("transcript") or "").strip()

        # 过滤空转写 / 静音幻觉
        from tools.voice_mode import is_whisper_hallucination
        if not transcript or is_whisper_hallucination(transcript):
            _log.info("[voice] 忽略空转写/静音幻觉: %r", transcript)
            session_data["state"] = "listening"
            await _set_status("listening")
            return

        _log.info("[voice] STT 转写结果: %s", transcript)
        await _send_json({"type": "transcription", "text": transcript})

        # 第二步：运行 Agent
        delta_queue: _queue.Queue = _queue.Queue()
        interrupt_event = _threading.Event()

        agent_task = _asyncio.create_task(
            _asyncio.to_thread(
                _run_agent_sync,
                transcript,
                session_id,
                delta_queue,
                interrupt_event,
            )
        )
        session_data["agent_task"] = agent_task
        session_data["interrupt_event"] = interrupt_event

        # 第三步：消费 delta_queue，同时做 TTS
        full_text = ""
        tts_text_buffer = ""
        terminal_event = None

        async def _flush_tts():
            nonlocal tts_text_buffer
            if not tts_text_buffer.strip():
                return
            text_to_speak = tts_text_buffer.strip()
            tts_text_buffer = ""
            tts_result = await _asyncio.to_thread(_run_tts, text_to_speak)
            if tts_result.get("success"):
                await _send_audio(tts_result["audio_data"])

        session_data["state"] = "speaking"
        await _set_status("speaking")

        while True:
            # 检查是否在消费过程中被打断
            if interrupt_event.is_set():
                terminal_event = {"type": "interrupted"}
                break

            try:
                item = await _asyncio.get_running_loop().run_in_executor(
                    None, lambda: delta_queue.get(timeout=0.3)
                )
            except _queue.Empty:
                if agent_task.done():
                    # 抽干残留
                    while True:
                        try:
                            item = delta_queue.get_nowait()
                        except _queue.Empty:
                            break
                        if item[0] == "token":
                            full_text += item[1]
                            tts_text_buffer += item[1]
                            await _send_json({"type": "token", "delta": item[1]})
                            if len(tts_text_buffer) >= _TTS_CHUNK_SIZE:
                                await _flush_tts()
                        elif item[0] == "interrupted":
                            terminal_event = {"type": "interrupted"}
                        elif item[0] == "error":
                            terminal_event = {"type": "error", "error": item[1]}
                    break
                continue

            kind, value = item

            if kind == "token":
                full_text += value
                tts_text_buffer += value
                await _send_json({"type": "token", "delta": value})
                if len(tts_text_buffer) >= _TTS_CHUNK_SIZE:
                    await _flush_tts()

            elif kind == "interrupted":
                terminal_event = {"type": "interrupted"}
                break

            elif kind == "error":
                terminal_event = {"type": "error", "error": value}
                break

        if terminal_event is None:
            await _flush_tts()
            await _send_json({"type": "done", "final": full_text})
        elif terminal_event["type"] == "interrupted":
            await _flush_tts()
            await _send_json({"type": "interrupted"})
        else:
            await _flush_tts()
            await _send_json(terminal_event)

        # 清理
        if not agent_task.done():
            agent_task.cancel()
        session_data["agent_task"] = None
        session_data["interrupt_event"] = None

        # 回到监听状态
        session_data["state"] = "listening"
        await _set_status("listening")

    # ---- 主循环 ----

    # 初始状态：等待 voice_start
    await _set_status("idle")
    session_data = None

    try:
        while True:
            raw = await ws.receive()

            # 处理二进制帧（PCM 音频数据）
            if "bytes" in raw:
                if session_data is None:
                    continue  # 尚未进入语音模式，忽略

                pcm_chunk = raw["bytes"]
                # 每 50 帧记录一次，确认数据到达
                chunk_count = session_data.get("_chunk_count", 0) + 1
                session_data["_chunk_count"] = chunk_count
                if chunk_count <= 3 or chunk_count % 50 == 0:
                    _log.info("[voice] PCM chunk #%d: %d bytes, state=%s", chunk_count, len(pcm_chunk), session_data.get("state", "?"))
                state = session_data.get("state", "idle")

                # 累积 PCM 到缓冲区，凑够 VAD 帧长（512 samples = 1024 bytes）
                pcm_buffer = session_data["pcm_buffer"]
                pcm_buffer.extend(pcm_chunk)

                while len(pcm_buffer) >= _VAD_FRAME_BYTES:
                    # 取出一帧
                    vad_frame = bytes(pcm_buffer[:_VAD_FRAME_BYTES])
                    del pcm_buffer[:_VAD_FRAME_BYTES]

                    # 调用 VAD 引擎（捕获异常，避免 WebSocket 断开）
                    vad_engine = session_data["vad_engine"]
                    try:
                        vad_result = vad_engine.process_chunk(vad_frame)
                    except Exception as e:
                        _log.error("[voice] VAD process_chunk 异常: %s (frame=%d bytes)", e, len(vad_frame))
                        continue  # 跳过此帧，不中断主循环

                    if vad_result == "speech_start":
                        _log.info("[voice] VAD: speech_start")
                        await _send_json({"type": "vad", "status": "speech_start"})
                        # 如果当前在 SPEAKING 状态，打断
                        if state == "speaking":
                            _log.info("[voice] 用户打断 AI 回复")
                            interrupt_event = session_data.get("interrupt_event")
                            if interrupt_event:
                                interrupt_event.set()
                            # 取消 agent task
                            agent_task = session_data.get("agent_task")
                            if agent_task and not agent_task.done():
                                agent_task.cancel()
                            session_data["agent_task"] = None
                            session_data["interrupt_event"] = None
                            # 清空旧的语音段，开始新累积
                            session_data["speech_audio"] = bytearray()
                        state = "listening"
                        session_data["state"] = state
                        await _set_status("listening")
                        # 开始累积新语音段
                        session_data["speech_audio"].extend(vad_frame)

                    elif vad_result == "speech_continue":
                        # 持续人声，累积音频
                        session_data["speech_audio"].extend(vad_frame)

                    elif vad_result == "speech_end":
                        _log.info("[voice] VAD: speech_end, speech_audio=%d bytes", len(session_data["speech_audio"]))
                        await _send_json({"type": "vad", "status": "speech_end"})
                        # 触发处理
                        await _handle_speech_end(session_data)
                        state = session_data["state"]  # 可能已被 _handle_speech_end 更新

                continue

            # 处理文本帧（JSON 控制消息）
            if "text" not in raw:
                continue

            try:
                msg = _json.loads(raw["text"])
            except _json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            # ---- voice_start: 进入语音模式 ----
            if msg_type == "voice_start":
                if session_data is not None:
                    continue  # 已经在语音模式

                # 初始化 VAD 引擎
                try:
                    from tools.vad_engine import VADEngine
                    vad_engine = VADEngine(silence_threshold_ms=800)
                except Exception as e:
                    _log.error("[voice] VAD 引擎初始化失败: %s", e)
                    await _send_json({"type": "error", "error": f"VAD 引擎初始化失败: {e}"})
                    continue

                session_data = {
                    "vad_engine": vad_engine,
                    "pcm_buffer": bytearray(),
                    "speech_audio": bytearray(),
                    "agent_task": None,
                    "interrupt_event": None,
                    "state": "listening",
                }
                _VOICE_SESSIONS[session_id] = session_data
                await _set_status("listening")
                _log.info("[voice] 进入语音模式 session=%s", session_id)

            # ---- voice_stop: 退出语音模式 ----
            elif msg_type == "voice_stop":
                if session_data is None:
                    continue

                # 取消正在运行的 agent
                agent_task = session_data.get("agent_task")
                if agent_task and not agent_task.done():
                    agent_task.cancel()

                # 释放 VAD 引擎
                vad_engine = session_data.get("vad_engine")
                if vad_engine:
                    vad_engine.close()

                session_data = None
                _VOICE_SESSIONS.pop(session_id, None)
                await _set_status("idle")
                _log.info("[voice] 退出语音模式 session=%s", session_id)

    except _WebSocketDisconnect:
        _log.info("[voice] WebSocket 断开 session=%s", session_id)
    except Exception as e:
        _log.warning("[voice] WebSocket 异常 session=%s: %s", session_id, e)
    finally:
        # 清理资源
        if session_data:
            vad_engine = session_data.get("vad_engine")
            if vad_engine:
                vad_engine.close()
            agent_task = session_data.get("agent_task")
            if agent_task and not agent_task.done():
                agent_task.cancel()
        _VOICE_SESSIONS.pop(session_id, None)


# 所有具体 API 路由注册完毕，最后挂 SPA catch-all
mount_spa(app)


def start_server(
    host: str = "127.0.0.1",
    port: int = 9119,
    open_browser: bool = True,
    allow_public: bool = False,
):
    """Start the web UI server."""
    import uvicorn

    # Write session token so dev tooling can read it
    _TOKEN_PATH = PROJECT_ROOT / ".server.token"
    _TOKEN_PATH.write_text(_SESSION_TOKEN)
    _log.info("Session token written to %s", _TOKEN_PATH)

    _LOCALHOST = ("127.0.0.1", "localhost", "::1")
    if host not in _LOCALHOST and not allow_public:
        raise SystemExit(
            f"Refusing to bind to {host} — the dashboard exposes API keys "
            f"and config without robust authentication.\n"
            f"Use --insecure to override (NOT recommended on untrusted networks)."
        )
    if host not in _LOCALHOST:
        _log.warning(
            "Binding to %s with --insecure — the dashboard has no robust "
            "authentication. Only use on trusted networks.", host,
        )

    if open_browser:
        import threading
        import webbrowser

        def _open():
            import time as _t
            _t.sleep(1.0)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open, daemon=True).start()

    print(f"  Hermes Web UI → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
