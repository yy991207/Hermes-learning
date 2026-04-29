#!/usr/bin/env python3
"""
AI Agent Runner with Tool Calling

本模块提供了一个简洁、独立的代理，可以执行具有工具调用能力的AI模型。
它处理对话循环、工具执行和响应管理。

功能特性：
- 自动工具调用循环直至完成
- 可配置的模型参数
- 错误处理和恢复
- 消息历史管理
- 支持多个模型提供商

使用方法：
    from run_agent import AIAgent
    
    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

import asyncio
import base64
import concurrent.futures
import copy
import hashlib
import json
import logging
logger = logging.getLogger(__name__)
import os
import random
import re
import sys
import tempfile
import time
import threading
from types import SimpleNamespace
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home

# 首先从 ~/.hermes/.env 加载 .env，然后从项目根目录作为开发后备。
# 用户管理的环境文件应该在重启时覆盖过时的 shell 导出。
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
if _loaded_env_paths:
    for _env_path in _loaded_env_paths:
        logger.info("Loaded environment variables from %s", _env_path)
else:
    logger.info("No .env file found. Using system environment variables.")


# 导入我们的工具系统
from model_tools import (
    get_tool_definitions,
    get_toolset_for_tool,
    handle_function_call,
    check_toolset_requirements,
)
from tools.terminal_tool import cleanup_vm, get_active_env, is_persistent_env
from tools.tool_result_storage import maybe_persist_tool_result, enforce_turn_budget
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool import cleanup_browser


from hermes_constants import OPENROUTER_BASE_URL

# 代理内部组件提取到 agent/ 包中以实现模块化
from agent.memory_manager import build_memory_context_block, sanitize_context
from agent.retry_utils import jittered_backoff
from agent.error_classifier import classify_api_error, FailoverReason
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS,
    MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE,
    build_nous_subscription_prompt,
)
from agent.model_metadata import (
    fetch_model_metadata,
    estimate_tokens_rough, estimate_messages_tokens_rough, estimate_request_tokens_rough,
    get_next_probe_tier, parse_context_limit_from_error,
    parse_available_output_tokens_from_error,
    save_context_length, is_local_endpoint,
    query_ollama_num_ctx,
)
from agent.context_compressor import ContextCompressor
from agent.skill_evolution import SkillEvolutionManager
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.prompt_caching import apply_anthropic_cache_control
from agent.prompt_builder import build_skills_system_prompt, build_context_files_prompt, build_environment_hints, load_soul_md, TOOL_USE_ENFORCEMENT_GUIDANCE, TOOL_USE_ENFORCEMENT_MODELS, DEVELOPER_ROLE_MODELS, GOOGLE_MODEL_OPERATIONAL_GUIDANCE, OPENAI_MODEL_EXECUTION_GUIDANCE
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from agent.display import (
    KawaiiSpinner, build_tool_preview as _build_tool_preview,
    get_cute_tool_message as _get_cute_tool_message_impl,
    _detect_tool_failure,
    get_tool_emoji as _get_tool_emoji,
)
from agent.trajectory import (
    convert_scratchpad_to_think, has_incomplete_scratchpad,
    save_trajectory as _save_trajectory_to_file,
)
from utils import atomic_json_write, env_var_enabled



class _SafeWriter:
    """Transparent stdio wrapper that catches OSError/ValueError from broken pipes.

    When hermes-agent runs as a systemd service, Docker container, or headless
    daemon, the stdout/stderr pipe can become unavailable (idle timeout, buffer
    exhaustion, socket reset). Any print() call then raises
    ``OSError: [Errno 5] Input/output error``, which can crash agent setup or
    run_conversation() — especially via double-fault when an except handler
    also tries to print.

    Additionally, when subagents run in ThreadPoolExecutor threads, the shared
    stdout handle can close between thread teardown and cleanup, raising
    ``ValueError: I/O operation on closed file`` instead of OSError.

    This wrapper delegates all writes to the underlying stream and silently
    catches both OSError and ValueError. It is transparent when the wrapped
    stream is healthy.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


class IterationBudget:
    """Thread-safe iteration counter for an agent.

    Each agent (parent or subagent) gets its own ``IterationBudget``.
    The parent's budget is capped at ``max_iterations`` (default 90).
    Each subagent gets an independent budget capped at
    ``delegation.max_iterations`` (default 50) — this means total
    iterations across parent + subagents can exceed the parent's cap.
    Users control the per-subagent limit via ``delegation.max_iterations``
    in config.yaml.

    ``execute_code`` (programmatic tool calling) iterations are refunded via
    :meth:`refund` so they don't eat into the budget.
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration (e.g. for execute_code turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


# 绝不能并发运行的工具（交互式/面向用户）。
# 当这些工具中的任何一个出现在批次中时，我们会回退到顺序执行。
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# 没有共享可变会话状态的只读工具。
_PARALLEL_SAFE_TOOLS = frozenset({
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_extract",
    "web_search",
})

# 当文件工具针对独立路径时，可以并发运行。
_PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})

# 并行工具执行的最大并发工作线程数。
_MAX_TOOL_WORKERS = 8

# 表示终端命令可能修改/删除文件的模式。
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,
)
# 覆盖文件的输出重定向（> 但不是 >>）
_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


def _is_destructive_command(cmd: str) -> bool:
    """启发式判断：这个终端命令看起来像是修改/删除文件吗？"""
    if not cmd:
        return False
    if _DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if _REDIRECT_OVERWRITE.search(cmd):
        return True
    return False


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """当工具调用批次可以安全并发运行时返回 True。"""
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False
            reserved_paths.append(scoped_path)
            continue

        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False

    return True


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Path | None:
    """返回路径范围工具的标准化文件目标。"""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    # 避免使用 resolve()；文件可能还不存在。
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """当两个路径可能引用同一子树时返回 True。"""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        # 空路径不应该到达这里（上游已防护），但为了安全起见。
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]



_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')




def _sanitize_surrogates(text: str) -> str:
    """用 U+FFFD（替换字符）替换单独的代理代码点。

    代理在 UTF-8 中是无效的，会导致 OpenAI SDK 内部的 ``json.dumps()`` 崩溃。
    当文本不包含代理时，这是一个快速的无操作。
    """
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


def _sanitize_structure_surrogates(payload: Any) -> bool:
    """就地替换嵌套字典/列表负载中的代理代码点。

    ``_sanitize_structure_non_ascii`` 的镜像，但用于代理恢复。
    用于清理嵌套结构字段（例如 ``reasoning_details`` — 一个包含
    ``summary``/``text`` 字符串的字典数组），这些字段是扁平化按字段检查
    无法触及的。如果替换了任何代理，则返回 True。
    """
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[key] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[idx] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found


def _sanitize_messages_surrogates(messages: list) -> bool:
    """清理消息列表中所有字符串内容的代理字符。

    就地遍历消息字典。如果找到并替换了任何代理则返回 True，
    否则返回 False。涵盖内容/文本、名称、工具调用元数据/参数，
    以及任何额外的字符串或嵌套结构字段（``reasoning``、
    ``reasoning_content``、``reasoning_details`` 等），这样重试
    不会在非内容字段上失败。字节级推理模型（xiaomi/mimo、kimi、glm）
    可以在推理输出中发出单独的代理，这些代理会在下一轮传递到
    ``api_messages["reasoning_content"]`` 并导致 OpenAI SDK 内部的 json.dumps 崩溃。
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
        name = msg.get("name")
        if isinstance(name, str) and _SURROGATE_RE.search(name):
            msg["name"] = _SURROGATE_RE.sub('\ufffd', name)
            found = True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and _SURROGATE_RE.search(tc_id):
                    tc["id"] = _SURROGATE_RE.sub('\ufffd', tc_id)
                    found = True
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn_name = fn.get("name")
                    if isinstance(fn_name, str) and _SURROGATE_RE.search(fn_name):
                        fn["name"] = _SURROGATE_RE.sub('\ufffd', fn_name)
                        found = True
                    fn_args = fn.get("arguments")
                    if isinstance(fn_args, str) and _SURROGATE_RE.search(fn_args):
                        fn["arguments"] = _SURROGATE_RE.sub('\ufffd', fn_args)
                        found = True
        # 遍历任何额外的字符串/嵌套字段（reasoning、
        # reasoning_content、reasoning_details 等）——来自字节级推理模型
        # （xiaomi/mimo、kimi、glm）的代理可能潜伏在这些字段中，
        # 并且未被上面的按字段检查覆盖。
        # 匹配 _sanitize_messages_non_ascii 的覆盖范围（PR #10537）。
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                if _SURROGATE_RE.search(value):
                    msg[key] = _SURROGATE_RE.sub('\ufffd', value)
                    found = True
            elif isinstance(value, (dict, list)):
                if _sanitize_structure_surrogates(value):
                    found = True
    return found


def _strip_non_ascii(text: str) -> str:
    """删除非 ASCII 字符，替换为最接近的 ASCII 等价物或直接删除。

    当系统编码为 ASCII 且无法处理任何非 ASCII 字符时
    用作最后的手段（例如 Chromebooks 上的 LANG=C）。
    """
    return text.encode('ascii', errors='ignore').decode('ascii')


def _sanitize_messages_non_ascii(messages: list) -> bool:
    """清理消息列表中所有字符串内容的非 ASCII 字符。

    这是仅 ASCII 编码系统（LANG=C、Chromebooks、最小容器）的
    最后恢复手段。如果找到并清理了任何非 ASCII 内容，则返回 True。
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # 清理内容（字符串）
        content = msg.get("content")
        if isinstance(content, str):
            sanitized = _strip_non_ascii(content)
            if sanitized != content:
                msg["content"] = sanitized
                found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        sanitized = _strip_non_ascii(text)
                        if sanitized != text:
                            part["text"] = sanitized
                            found = True
        # 清理名称字段（在工具结果中可能包含非 ASCII）
        name = msg.get("name")
        if isinstance(name, str):
            sanitized = _strip_non_ascii(name)
            if sanitized != name:
                msg["name"] = sanitized
                found = True
        # 清理 tool_calls
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        fn_args = fn.get("arguments")
                        if isinstance(fn_args, str):
                            sanitized = _strip_non_ascii(fn_args)
                            if sanitized != fn_args:
                                fn["arguments"] = sanitized
                                found = True
        # 清理任何额外的顶级字符串字段（例如 reasoning_content）
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                sanitized = _strip_non_ascii(value)
                if sanitized != value:
                    msg[key] = sanitized
                    found = True
    return found


def _sanitize_tools_non_ascii(tools: list) -> bool:
    """就地清理工具负载中的非 ASCII 字符。"""
    return _sanitize_structure_non_ascii(tools)


def _sanitize_structure_non_ascii(payload: Any) -> bool:
    """就地清理嵌套字典/列表负载中的非 ASCII 字符。"""
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[key] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[idx] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found





# =========================================================================
# 大型工具结果处理器 — 将超大输出保存到临时文件
# =========================================================================


# =========================================================================
# Qwen Portal 头部 — 模拟 QwenCode CLI 以兼容 portal.qwen.ai。
# 提取为模块级辅助函数，以便 __init__ 和 _apply_client_headers_for_base_url 可以共享它。
# =========================================================================
_QWEN_CODE_VERSION = "0.14.1"


def _qwen_portal_headers() -> dict:
    """返回 Qwen Portal API 所需的默认 HTTP 头部。"""
    import platform as _plat

    _ua = f"QwenCode/{_QWEN_CODE_VERSION} ({_plat.system().lower()}; {_plat.machine()})"
    return {
        "User-Agent": _ua,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": _ua,
        "X-DashScope-AuthType": "qwen-oauth",
    }


class AIAgent:
    """
    具有工具调用能力的 AI 代理。

    此类管理支持函数调用的 AI 模型的对话流程、工具执行和响应处理。
    """

    @property
    def _iters_since_skill(self) -> int:
        manager = getattr(self, "skill_evolution", None)
        return manager.iters_since_skill if manager else 0

    @_iters_since_skill.setter
    def _iters_since_skill(self, value: int) -> None:
        manager = getattr(self, "skill_evolution", None)
        if manager is None:
            self.skill_evolution = SkillEvolutionManager()
            manager = self.skill_evolution
        try:
            manager.iters_since_skill = int(value)
        except Exception:
            manager.iters_since_skill = 0

    @property
    def _skill_nudge_interval(self) -> int:
        manager = getattr(self, "skill_evolution", None)
        if manager is None:
            return SkillEvolutionManager.DEFAULT_CREATION_NUDGE_INTERVAL
        return manager.creation_nudge_interval

    @_skill_nudge_interval.setter
    def _skill_nudge_interval(self, value: int) -> None:
        manager = getattr(self, "skill_evolution", None)
        if manager is None:
            self.skill_evolution = SkillEvolutionManager(value)
        else:
            manager.creation_nudge_interval = SkillEvolutionManager._coerce_interval(value)

    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        acp_command: str = None,
        acp_args: list[str] | None = None,
        command: str = None,
        args: list[str] | None = None,
        model: str = "",
        max_iterations: int = 90,  # 默认工具调用迭代次数（与子代理共享）
        tool_delay: float = 1.0,
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
        provider_require_parameters: bool = False,
        provider_data_collection: str = None,
        session_id: str = None,
        tool_progress_callback: callable = None,
        tool_start_callback: callable = None,
        tool_complete_callback: callable = None,
        thinking_callback: callable = None,
        reasoning_callback: callable = None,
        clarify_callback: callable = None,
        step_callback: callable = None,
        stream_delta_callback: callable = None,
        interim_assistant_callback: callable = None,
        tool_gen_callback: callable = None,
        status_callback: callable = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        service_tier: str = None,
        request_overrides: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        user_id: str = None,
        gateway_session_key: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        session_db=None,
        parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None,
        fallback_model: Dict[str, Any] = None,
        credential_pool=None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 50,
        pass_session_id: bool = False,
        persist_session: bool = True,
    ):
        """
        初始化 AI 代理。

        参数：
            base_url (str): 模型 API 的基础 URL（可选）
            api_key (str): 身份验证的 API 密钥（可选，如果未提供则使用环境变量）
            provider (str): 提供商标识符（可选；用于遥测/路由提示）
            api_mode (str): API 模式覆盖："chat_completions" 或 "codex_responses"
            model (str): 要使用的模型名称（默认："anthropic/claude-opus-4.6"）
            max_iterations (int): 工具调用迭代的最大次数（默认：90）
            tool_delay (float): 工具调用之间的延迟（秒）（默认：1.0）
            enabled_toolsets (List[str]): 仅启用这些工具集中的工具（可选）
            disabled_toolsets (List[str]): 禁用这些工具集中的工具（可选）
            save_trajectories (bool): 是否将对话轨迹保存到 JSONL 文件（默认：False）
            verbose_logging (bool): 启用详细日志记录用于调试（默认：False）
            quiet_mode (bool): 抑制进度输出以获得干净的 CLI 体验（默认：False）
            ephemeral_system_prompt (str): 代理执行期间使用的系统提示，但不保存到轨迹（可选）
            log_prefix_chars (int): 在日志预览中显示的工具调用/响应字符数（默认：100）
            log_prefix (str): 添加到所有日志消息的前缀，用于并行处理中的识别（默认：""）
            providers_allowed (List[str]): 允许的 OpenRouter 提供商（可选）
            providers_ignored (List[str]): 忽略的 OpenRouter 提供商（可选）
            providers_order (List[str]): 按顺序尝试的 OpenRouter 提供商（可选）
            provider_sort (str): 按价格/吞吐量/延迟对提供商排序（可选）
            session_id (str): 用于日志记录的预生成会话 ID（可选，如果未提供则自动生成）
            tool_progress_callback (callable): 进度通知的回调函数(tool_name, args_preview)
            clarify_callback (callable): 交互式用户问题的回调函数(question, choices) -> str。
                由平台层（CLI 或网关）提供。如果为 None，clarify 工具返回错误。
            max_tokens (int): 模型响应的最大令牌数（可选，如果未设置则使用模型默认值）
            reasoning_config (Dict): OpenRouter 推理配置覆盖（例如 {"effort": "none"} 以禁用思考）。
                如果为 None，OpenRouter 默认为 {"enabled": True, "effort": "medium"}。设置以禁用/自定义推理。
            prefill_messages (List[Dict]): 作为预填充上下文添加到对话历史的消息。
                用于注入少样本示例或启动模型的响应风格。
                示例：[{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
                注意：Anthropic Sonnet 4.6+ 和 Opus 4.6+ 拒绝以助手角色消息结尾的对话
                （400 错误）。对于这些模型，使用结构化输出或
                output_config.format 而不是尾随助手预填充。
            platform (str): 用户所在的接口平台（例如 "cli"、"telegram"、"discord"、"whatsapp"）。
                用于向系统提示注入平台特定的格式提示。
            skip_context_files (bool): 如果为 True，跳过自动注入 SOUL.md、AGENTS.md 和 .cursorrules
                到系统提示中。用于批处理和数据生成，以避免用用户特定的角色或项目指令污染轨迹。
        """
        _install_safe_stdio()

        self.model = model
        self.max_iterations = max_iterations
        # 共享迭代预算 — 父代理创建，子代理继承。
        # 由父代理 + 所有子代理的每个 LLM 回合消耗。
        self.iteration_budget = iteration_budget or IterationBudget(max_iterations)
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform  # "cli"、"telegram"、"discord"、"whatsapp" 等。
        self._user_id = user_id  # 平台用户标识符（网关会话）
        self._gateway_session_key = gateway_session_key  # 稳定的每聊天密钥（例如 agent:main:telegram:dm:123）
        # 可插拔打印函数 — CLI 用 _cprint 替换它，以便原始 ANSI 状态行
        # 通过 prompt_toolkit 的渲染器路由，而不是直接进入 stdout，
        # 在那里 patch_stdout 的 StdoutProxy 会弄乱转义序列。None = 使用 builtins.print。
        self._print_fn = None
        self.background_review_callback = None  # 网关传递的可选同步回调
        self.skip_context_files = skip_context_files
        self.pass_session_id = pass_session_id
        self.persist_session = persist_session
        self._credential_pool = credential_pool
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        # 存储有效的基础 URL 用于功能检测（提示缓存、推理等）
        self.base_url = base_url or ""
        provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
        self.provider = provider_name or ""
        self.acp_command = acp_command or command
        self.acp_args = list(acp_args or args or [])
        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse"}:
            self.api_mode = api_mode
        elif self.provider == "openai-codex":
            self.api_mode = "codex_responses"
        elif self.provider == "xai":
            self.api_mode = "codex_responses"
        elif (provider_name is None) and "chatgpt.com/backend-api/codex" in self._base_url_lower:
            self.api_mode = "codex_responses"
            self.provider = "openai-codex"
        elif (provider_name is None) and "api.x.ai" in self._base_url_lower:
            self.api_mode = "codex_responses"
            self.provider = "xai"
        elif self.provider == "anthropic" or (provider_name is None and "api.anthropic.com" in self._base_url_lower):
            self.api_mode = "anthropic_messages"
            self.provider = "anthropic"
        elif self._base_url_lower.rstrip("/").endswith("/anthropic"):
            # 第三方 Anthropic 兼容端点（例如 MiniMax、DashScope）
            # 使用以 /anthropic 结尾的 URL 约定。自动检测这些，
            # 以便使用 Anthropic Messages API 适配器而不是聊天完成。
            self.api_mode = "anthropic_messages"
        elif self.provider == "bedrock" or "bedrock-runtime" in self._base_url_lower:
            # AWS Bedrock — 从提供商名称或基础 URL 自动检测。
            self.api_mode = "bedrock_converse"
        else:
            self.api_mode = "chat_completions"

        try:
            from hermes_cli.model_normalize import (
                _AGGREGATOR_PROVIDERS,
                normalize_model_for_provider,
            )

            if self.provider not in _AGGREGATOR_PROVIDERS:
                self.model = normalize_model_for_provider(self.model, self.provider)
        except Exception:
            pass

        # GPT-5.x 模型通常需要 Responses API 路径，但一些
        # 提供商有例外（例如 Copilot 的 gpt-5-mini 仍然
        # 使用聊天完成）。对于直接 OpenAI URL（api.openai.com）也自动升级，
        # 因为所有较新的工具调用模型更喜欢那里的 Responses。
        # ACP 运行时被排除：CopilotACPClient 处理自己的路由，
        # 并且不实现 Responses API 表面。
        # 当明确提供 api_mode 时，尊重它 — 用户
        # 知道他们的端点支持什么（#10473）。
        if (
            api_mode is None
            and self.api_mode == "chat_completions"
            and self.provider != "copilot-acp"
            and not str(self.base_url or "").lower().startswith("acp://copilot")
            and not str(self.base_url or "").lower().startswith("acp+tcp://")
            and (
                self._is_direct_openai_url()
                or self._provider_model_requires_responses_api(
                    self.model,
                    provider=self.provider,
                )
            )
        ):
            self.api_mode = "codex_responses"

        # 在后台线程中预热 OpenRouter 模型元数据缓存。
        # fetch_model_metadata() 缓存 1 小时；这避免了在估算定价时
        # 第一次 API 响应上的阻塞 HTTP 请求。
        if self.provider == "openrouter" or self._is_openrouter_url():
            threading.Thread(
                target=lambda: fetch_model_metadata(),
                daemon=True,
            ).start()

        self.tool_progress_callback = tool_progress_callback
        self.tool_start_callback = tool_start_callback
        self.tool_complete_callback = tool_complete_callback
        self.suppress_status_output = False
        self.thinking_callback = thinking_callback
        self.reasoning_callback = reasoning_callback
        self.clarify_callback = clarify_callback
        self.step_callback = step_callback
        self.stream_delta_callback = stream_delta_callback
        self.interim_assistant_callback = interim_assistant_callback
        self.status_callback = status_callback
        self.tool_gen_callback = tool_gen_callback

        
        # 工具执行状态 — 即使注册了流消费者，也允许在工具执行期间使用 _vprint
        # （那时没有令牌流）
        self._executing_tools = False

        # 中断机制，用于跳出工具循环
        self._interrupt_requested = False
        self._interrupt_message = None  # 触发中断的可选消息
        self._execution_thread_id: int | None = None  # 在 run_conversation() 开始时设置
        self._interrupt_thread_signal_pending = False
        self._client_lock = threading.RLock()

        # /steer 机制 — 将用户注释注入到下一个工具结果中
        # 而不中断代理。与 interrupt() 不同，steer() 不会
        # 设置 _interrupt_requested；它等待当前工具批次
        # 自然完成，然后 drain 钩子将文本附加到
        # 最后一个工具结果的内容中，以便模型在下次迭代中看到它。
        # 消息角色交替得到保留（我们修改现有的工具消息，
        # 而不是插入新的用户回合）。
        self._pending_steer: Optional[str] = None
        self._pending_steer_lock = threading.Lock()

        # 并发工具工作线程跟踪。`_execute_tool_calls_concurrent`
        # 在自己的 ThreadPoolExecutor 工作器上运行每个工具 — 那些工作器
        # 线程的 tid 与 `_execution_thread_id` 不同，因此
        # `_set_interrupt(True, _execution_thread_id)` 本身不会导致
        # 工作器内部的 `is_interrupted()` 返回 True。在这里跟踪
        # 工作器，以便 `interrupt()` / `clear_interrupt()` 可以明确地
        # 扩散到它们的 tid。
        self._tool_worker_threads: set[int] = set()
        self._tool_worker_threads_lock = threading.Lock()
        
        # 子代理委派状态
        self._delegate_depth = 0        # 0 = 顶级代理，子代理递增
        self._active_children = []      # 运行中的子 AIAgent（用于中断传播）
        self._active_children_lock = threading.Lock()
        
        # 存储 OpenRouter 提供商偏好
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        self.provider_require_parameters = provider_require_parameters
        self.provider_data_collection = provider_data_collection

        # 存储工具集过滤选项
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        
        # 模型响应配置
        self.max_tokens = max_tokens  # None = 使用模型默认值
        self.reasoning_config = reasoning_config  # None = 使用默认值（OpenRouter 为 medium）
        self.service_tier = service_tier
        self.request_overrides = dict(request_overrides or {})
        self.prefill_messages = prefill_messages or []  # 预填充的对话回合
        self._force_ascii_payload = False
        
        # Anthropic 提示缓存：通过 OpenRouter 为 Claude 模型自动启用。
        # 通过缓存对话前缀在多轮对话中减少约 75% 的输入成本。
        # 使用 system_and_3 策略（4 个断点）。
        is_openrouter = self._is_openrouter_url()
        is_claude = "claude" in self.model.lower()
        is_native_anthropic = self.api_mode == "anthropic_messages" and self.provider == "anthropic"
        self._use_prompt_caching = (is_openrouter and is_claude) or is_native_anthropic
        self._cache_ttl = "5m"  # 默认 5 分钟 TTL（1.25 倍写入成本）
        
        # 迭代预算：仅当 LLM 实际耗尽迭代预算时
        # （api_call_count >= max_iterations）才会通知 LLM。此时
        # 我们注入一条消息，允许最后一次 API 调用，如果模型
        # 没有产生文本响应，强制发送用户消息要求
        # 它总结。没有中间压力警告 — 它们导致模型
        # 在复杂任务上过早"放弃"（#7915）。
        self._budget_exhausted_injected = False
        self._budget_grace_call = False

        # 活动跟踪 — 在每次 API 调用、工具执行和
        # 流块时更新。由网关超时处理器用于报告
        # 代理被杀死时正在做什么，以及通过"仍在工作"
        # 通知显示进度。
        self._last_activity_ts: float = time.time()
        self._last_activity_desc: str = "initializing"
        self._current_tool: str | None = None
        self._api_call_count: int = 0

        # 速率限制跟踪 — 在每次 API 调用后从 x-ratelimit-* 响应头更新。
        # 由 /usage 斜杠命令访问。
        self._rate_limit_state: Optional["RateLimitState"] = None

        # 集中式日志记录 — agent.log (INFO+) 和 errors.log (WARNING+)
        # 都位于 ~/.hermes/logs/ 下。幂等，因此网关模式
        # （每条消息创建一个新的 AIAgent）不会重复处理器。
        from hermes_logging import setup_logging, setup_verbose_logging
        setup_logging(hermes_home=_hermes_home)

        if self.verbose_logging:
            setup_verbose_logging()
            logger.info("Verbose logging enabled (third-party library logs suppressed)")
        else:
            if self.quiet_mode:
                # 在安静模式（CLI 默认），抑制控制台上的所有工具/基础架构日志
                # 噪音。TUI 有自己的丰富状态显示；
                # logger INFO/WARNING 消息只会弄乱它。
                # 文件处理器（agent.log、errors.log）仍然捕获所有内容。
                for quiet_logger in [
                    'tools',               # all tools.* (terminal, browser, web, file, etc.)
                    'run_agent',            # agent runner internals
                    'trajectory_compressor',
                    'cron',                 # scheduler (only relevant in daemon mode)
                    'hermes_cli',           # CLI helpers
                ]:
                    logging.getLogger(quiet_logger).setLevel(logging.ERROR)
        
        # 内部流回调（在流式 TTS 期间设置）。
        # 在这里初始化，以便 _vprint 可以在 run_conversation 之前引用它。
        self._stream_callback = None
        # 延迟段落中断标志 — 在工具迭代后设置，以便
        # 单个 "\n\n" 被添加到下一个真实文本增量之前。
        self._stream_needs_break = False
        # 在当前模型响应期间已经通过实时令牌回调传递的可见助手文本。
        # 用于避免当提供者稍后将其作为完成的临时助手消息返回时重新发送相同的注释。
        self._current_streamed_assistant_text = ""

        # 可选的当前回合用户消息覆盖，当面向 API 的
        # 用户消息故意与持久化的记录不同时使用
        # （例如 CLI 语音模式仅为实时调用添加临时前缀）。
        self._persist_user_message_idx = None
        self._persist_user_message_override = None

        # 缓存每个图像负载/URL 的 anthropic 图像到文本回退，以便
        # 单个工具循环不会在同一图像历史记录上重复运行辅助视觉。
        self._anthropic_image_fallback_cache: Dict[str, str] = {}

        # 通过集中式提供商路由器初始化 LLM 客户端。
        # 路由器处理身份验证解析、基础 URL、头部，以及
        # 所有已知提供商的 Codex/Anthropic 包装。
        # raw_codex=True 因为主代理需要直接的 responses.stream()
        # 访问以进行 Codex Responses API 流式传输。
        self._anthropic_client = None
        self._is_anthropic_oauth = False

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
            # Bedrock + Claude → 使用 AnthropicBedrock SDK 以获得完整的功能对等
            # （提示缓存、思考预算、自适应思考）。
            _is_bedrock_anthropic = self.provider == "bedrock"
            if _is_bedrock_anthropic:
                from agent.anthropic_adapter import build_anthropic_bedrock_client
                import re as _re
                _region_match = _re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
                _br_region = _region_match.group(1) if _region_match else "us-east-1"
                self._bedrock_region = _br_region
                self._anthropic_client = build_anthropic_bedrock_client(_br_region)
                self._anthropic_api_key = "aws-sdk"
                self._anthropic_base_url = base_url
                self._is_anthropic_oauth = False
                self.api_key = "aws-sdk"
                self.client = None
                self._client_kwargs = {}
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model} (AWS Bedrock + AnthropicBedrock SDK, {_br_region})")
            else:
                # 仅当提供商实际上是 Anthropic 时才回退到 ANTHROPIC_TOKEN。
                # 其他 anthropic_messages 提供商（MiniMax、Alibaba 等）必须使用自己的 API 密钥。
                # 回退会将 Anthropic 凭据发送到第三方端点（修复 #1739、#minimax-401）。
                _is_native_anthropic = self.provider == "anthropic"
                effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = base_url
                from agent.anthropic_adapter import _is_oauth_token as _is_oat
                self._is_anthropic_oauth = _is_oat(effective_key)
                self._anthropic_client = build_anthropic_client(effective_key, base_url)
                # No OpenAI client needed for Anthropic mode
                self.client = None
                self._client_kwargs = {}
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model} (Anthropic native)")
                    if effective_key and len(effective_key) > 12:
                        print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
        elif self.api_mode == "bedrock_converse":
            # AWS Bedrock — 直接使用 boto3，不需要 OpenAI 客户端。
            # 区域从 base_url 提取或默认为 us-east-1。
            import re as _re
            _region_match = _re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
            self._bedrock_region = _region_match.group(1) if _region_match else "us-east-1"
            # 防护栏配置 — 在初始化时从 config.yaml 读取。
            self._bedrock_guardrail_config = None
            try:
                from hermes_cli.config import load_config as _load_br_cfg
                _gr = _load_br_cfg().get("bedrock", {}).get("guardrail", {})
                if _gr.get("guardrail_identifier") and _gr.get("guardrail_version"):
                    self._bedrock_guardrail_config = {
                        "guardrailIdentifier": _gr["guardrail_identifier"],
                        "guardrailVersion": _gr["guardrail_version"],
                    }
                    if _gr.get("stream_processing_mode"):
                        self._bedrock_guardrail_config["streamProcessingMode"] = _gr["stream_processing_mode"]
                    if _gr.get("trace"):
                        self._bedrock_guardrail_config["trace"] = _gr["trace"]
            except Exception:
                pass
            self.client = None
            self._client_kwargs = {}
            if not self.quiet_mode:
                _gr_label = " + Guardrails" if self._bedrock_guardrail_config else ""
                print(f"🤖 AI Agent initialized with model: {self.model} (AWS Bedrock, {self._bedrock_region}{_gr_label})")
        else:
            if api_key and base_url:
                # 来自 CLI/网关的明确凭据 — 直接构造。
                # 运行时提供商解析器已经为我们处理了身份验证。
                client_kwargs = {"api_key": api_key, "base_url": base_url}
                if self.provider == "copilot-acp":
                    client_kwargs["command"] = self.acp_command
                    client_kwargs["args"] = self.acp_args
                effective_base = base_url
                if "openrouter" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                        "X-OpenRouter-Title": "Hermes Agent",
                        "X-OpenRouter-Categories": "productivity,cli-agent",
                    }
                elif "api.githubcopilot.com" in effective_base.lower():
                    from hermes_cli.models import copilot_default_headers

                    client_kwargs["default_headers"] = copilot_default_headers()
                elif "api.kimi.com" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "User-Agent": "KimiCLI/1.30.0",
                    }
                elif "portal.qwen.ai" in effective_base.lower():
                    client_kwargs["default_headers"] = _qwen_portal_headers()
            else:
                # 没有明确凭据 — 使用集中式提供商路由器
                from agent.auxiliary_client import resolve_provider_client
                _routed_client, _ = resolve_provider_client(
                    self.provider or "auto", model=self.model, raw_codex=True)
                if _routed_client is not None:
                    client_kwargs = {
                        "api_key": _routed_client.api_key,
                        "base_url": str(_routed_client.base_url),
                    }
                    # Preserve any default_headers the router set
                    if hasattr(_routed_client, '_default_headers') and _routed_client._default_headers:
                        client_kwargs["default_headers"] = dict(_routed_client._default_headers)
                else:
                    # 当用户明确选择了非 OpenRouter 提供商
                    # 但未找到凭据时，快速失败并显示清晰消息，
                    # 而不是静默路由通过 OpenRouter。
                    _explicit = (self.provider or "").strip().lower()
                    if _explicit and _explicit not in ("auto", "openrouter", "custom"):
                        # 从提供商配置中查找实际的环境变量名称
                        # — 一些提供商使用非标准名称
                        # （例如 alibaba → DASHSCOPE_API_KEY，而不是 ALIBABA_API_KEY）。
                        _env_hint = f"{_explicit.upper()}_API_KEY"
                        try:
                            from hermes_cli.auth import PROVIDER_REGISTRY
                            _pcfg = PROVIDER_REGISTRY.get(_explicit)
                            if _pcfg and _pcfg.api_key_env_vars:
                                _env_hint = _pcfg.api_key_env_vars[0]
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"Provider '{_explicit}' is set in config.yaml but no API key "
                            f"was found. Set the {_env_hint} environment "
                            f"variable, or switch to a different provider with `hermes model`."
                        )
                    # 没有配置提供商 — 用清晰消息拒绝。
                    raise RuntimeError(
                        "No LLM provider configured. Run `hermes model` to "
                        "select a provider, or run `hermes setup` for first-time "
                        "configuration."
                    )
            
            self._client_kwargs = client_kwargs  # stored for rebuilding after interrupt

            # 为 OpenRouter 上的 Claude 启用细粒度工具流式传输。
            # 没有这个，Anthropic 会缓冲整个工具调用并在思考时
            # 沉默数分钟 — OpenRouter 的上游代理
            # 在沉默期间超时。beta 头部使 Anthropic
            # 逐令牌流式传输工具调用参数，保持连接活跃。
            _effective_base = str(client_kwargs.get("base_url", "")).lower()
            if "openrouter" in _effective_base and "claude" in (self.model or "").lower():
                headers = client_kwargs.get("default_headers") or {}
                existing_beta = headers.get("x-anthropic-beta", "")
                _FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14"
                if _FINE_GRAINED not in existing_beta:
                    if existing_beta:
                        headers["x-anthropic-beta"] = f"{existing_beta},{_FINE_GRAINED}"
                    else:
                        headers["x-anthropic-beta"] = _FINE_GRAINED
                    client_kwargs["default_headers"] = headers

            self.api_key = client_kwargs.get("api_key", "")
            self.base_url = client_kwargs.get("base_url", self.base_url)
            try:
                self.client = self._create_openai_client(client_kwargs, reason="agent_init", shared=True)
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model}")
                    if base_url:
                        print(f"🔗 Using custom base URL: {base_url}")
                    # Always show API key info (masked) for debugging auth issues
                    key_used = client_kwargs.get("api_key", "none")
                    if key_used and key_used != "dummy-key" and len(key_used) > 12:
                        print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                    else:
                        print(f"⚠️  Warning: API key appears invalid or missing (got: '{key_used[:20] if key_used else 'none'}...')")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        # 提供商回退链 — 当主提供商耗尽（速率限制、过载、连接
        # 失败）时尝试的备份提供商的有序列表。
        # 支持传统的单一字典 ``fallback_model`` 和
        # 新的列表 ``fallback_providers`` 格式。
        if isinstance(fallback_model, list):
            self._fallback_chain = [
                f for f in fallback_model
                if isinstance(f, dict) and f.get("provider") and f.get("model")
            ]
        elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
            self._fallback_chain = [fallback_model]
        else:
            self._fallback_chain = []
        self._fallback_index = 0
        self._fallback_activated = False
        # 为向后兼容保留的传统属性（测试、外部调用者）
        self._fallback_model = self._fallback_chain[0] if self._fallback_chain else None
        if self._fallback_chain and not self.quiet_mode:
            if len(self._fallback_chain) == 1:
                fb = self._fallback_chain[0]
                print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
            else:
                print(f"🔄 Fallback chain ({len(self._fallback_chain)} providers): " +
                      " → ".join(f"{f['model']} ({f['provider']})" for f in self._fallback_chain))

        # 获取可用工具并进行过滤
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=self.quiet_mode,
        )
        
        # 显示工具配置并存储有效工具名称以进行验证
        self.valid_tool_names = set()
        if self.tools:
            self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}
            tool_names = sorted(self.valid_tool_names)
            if not self.quiet_mode:
                print(f"🛠️  Loaded {len(self.tools)} tools: {', '.join(tool_names)}")
                
                # 如果应用了过滤，显示过滤信息
                if enabled_toolsets:
                    print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
                if disabled_toolsets:
                    print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        elif not self.quiet_mode:
            print("🛠️  No tools loaded (all tools filtered out or unavailable)")
        
        # 检查工具要求
        if self.tools and not self.quiet_mode:
            requirements = check_toolset_requirements()
            missing_reqs = [name for name, available in requirements.items() if not available]
            if missing_reqs:
                print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")
        
        # 显示轨迹保存状态
        if self.save_trajectories and not self.quiet_mode:
            print("📝 Trajectory saving enabled")
        
        # 显示临时系统提示状态
        if self.ephemeral_system_prompt and not self.quiet_mode:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")
        
        # 显示提示缓存状态
        if self._use_prompt_caching and not self.quiet_mode:
            source = "native Anthropic" if is_native_anthropic else "Claude via OpenRouter"
            print(f"💾 Prompt caching: ENABLED ({source}, {self._cache_ttl} TTL)")
        
        # 会话日志设置 - 自动保存对话轨迹以进行调试
        self.session_start = datetime.now()
        if session_id:
            # 使用提供的会话 ID（例如，来自 CLI）
            self.session_id = session_id
        else:
            # 生成新的会话 ID
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"
        
        # 会话日志与网关会话一起进入 ~/.hermes/sessions/
        hermes_home = get_hermes_home()
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
        
        # 跟踪会话日志记录的对话消息
        self._session_messages: List[Dict[str, Any]] = []
        
        # 缓存的系统提示 -- 每个会话构建一次，仅在压缩时重建
        self._cached_system_prompt: Optional[str] = None
        
        # 文件系统检查点管理器（透明 — 不是工具）
        from tools.checkpoint_manager import CheckpointManager
        self._checkpoint_mgr = CheckpointManager(
            enabled=checkpoints_enabled,
            max_snapshots=checkpoint_max_snapshots,
        )
        
        # SQLite 会话存储（可选 -- 由 CLI 或网关提供）
        self._session_db = session_db
        self._parent_session_id = parent_session_id
        self._last_flushed_db_idx = 0  # 跟踪 DB 写入游标以防止重复写入
        if self._session_db:
            try:
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    model_config={
                        "max_iterations": self.max_iterations,
                        "reasoning_config": reasoning_config,
                        "max_tokens": max_tokens,
                    },
                    user_id=None,
                    parent_session_id=self._parent_session_id,
                )
            except Exception as e:
                # 瞬时 SQLite 锁争用（例如 CLI 和网关并发写入）
                # 绝不能永久禁用此代理的 session_search。
                # 保持 _session_db 活跃 — 后续消息刷新和
                # session_search 调用在锁清除后仍将工作。
                # 会话行可能在此运行的索引中丢失，但这是可恢复的（刷新 upsert 行）。
                logger.warning(
                    "Session DB create_session failed (session_search still available): %s", e
                )
        
        # 用于任务规划的内存待办事项列表（每个代理/会话一个）
        from tools.todo_tool import TodoStore
        self._todo_store = TodoStore()
        
        # 为内存、技能和压缩部分加载一次配置
        try:
            from hermes_cli.config import load_config as _load_agent_config
            _agent_cfg = _load_agent_config()
        except Exception:
            _agent_cfg = {}

        # 持久内存（MEMORY.md + USER.md）-- 从磁盘加载
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_nudge_interval = 10
        self._memory_flush_min_turns = 6
        self._turns_since_memory = 0
        self.skill_evolution = SkillEvolutionManager.from_config(_agent_cfg)
        if not skip_memory:
            try:
                mem_config = _agent_cfg.get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
                self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
                self._memory_flush_min_turns = int(mem_config.get("flush_min_turns", 6))
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore
                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),
                        user_char_limit=mem_config.get("user_char_limit", 1375),
                    )
                    self._memory_store.load_from_disk()
            except Exception:
                pass  # Memory is optional -- don't break agent init
        


        # 内存提供商插件（外部 -- 一次一个，与内置一起）
        # 从配置读取 memory.provider 以选择要激活的插件。
        self._memory_manager = None
        if not skip_memory:
            try:
                _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

                if _mem_provider_name:
                    from agent.memory_manager import MemoryManager as _MemoryManager
                    from plugins.memory import load_memory_provider as _load_mem
                    self._memory_manager = _MemoryManager()
                    _mp = _load_mem(_mem_provider_name)
                    if _mp and _mp.is_available():
                        self._memory_manager.add_provider(_mp)
                    if self._memory_manager.providers:
                        from hermes_constants import get_hermes_home as _ghh
                        _init_kwargs = {
                            "session_id": self.session_id,
                            "platform": platform or "cli",
                            "hermes_home": str(_ghh()),
                            "agent_context": "primary",
                        }
                        # 内存提供商作用域的线程会话标题
                        # （例如 honcho 使用它来推导聊天范围的会话密钥）
                        if self._session_db:
                            try:
                                _st = self._session_db.get_session_title(self.session_id)
                                if _st:
                                    _init_kwargs["session_title"] = _st
                            except Exception:
                                pass
                        # 每用户内存作用域的线程网关用户身份
                        if self._user_id:
                            _init_kwargs["user_id"] = self._user_id
                        # 稳定每聊天 Honcho 会话隔离的线程网关会话密钥
                        if self._gateway_session_key:
                            _init_kwargs["gateway_session_key"] = self._gateway_session_key
                        # 每配置文件提供商作用域的配置文件身份
                        try:
                            from hermes_cli.profiles import get_active_profile_name
                            _profile = get_active_profile_name()
                            _init_kwargs["agent_identity"] = _profile
                            _init_kwargs["agent_workspace"] = "hermes"
                        except Exception:
                            pass
                        self._memory_manager.initialize_all(**_init_kwargs)
                        logger.info("Memory provider '%s' activated", _mem_provider_name)
                    else:
                        logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                        self._memory_manager = None
            except Exception as _mpe:
                logger.warning("Memory provider plugin init failed: %s", _mpe)
                self._memory_manager = None

        # 将内存提供商工具模式注入到工具表面。
        # 跳过名称已存在的工具（插件可能通过 ctx.register_tool()
        # 注册相同的工具，这些工具通过 get_tool_definitions() 进入 self.tools）。
        # 重复的函数名在强制执行唯一名称的提供商上导致 400 错误
        # （例如通过 Nous Portal 的 Xiaomi MiMo）。
        if self._memory_manager and self.tools is not None:
            _existing_tool_names = {
                t.get("function", {}).get("name")
                for t in self.tools
                if isinstance(t, dict)
            }
            for _schema in self._memory_manager.get_all_tool_schemas():
                _tname = _schema.get("name", "")
                if _tname and _tname in _existing_tool_names:
                    continue  # already registered via plugin path
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                if _tname:
                    self.valid_tool_names.add(_tname)
                    _existing_tool_names.add(_tname)

        # 技能配置：技能创建提醒的推动间隔
        # 工具使用强制配置："auto"（默认 -- 匹配硬编码模型列表）、
        # true（总是）、false（从不）或子字符串列表。
        _agent_section = _agent_cfg.get("agent", {})
        if not isinstance(_agent_section, dict):
            _agent_section = {}
        self._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")

        # 初始化上下文压缩器以进行自动上下文管理
        # 在接近模型的上下文限制时压缩对话
        # 通过 config.yaml 配置（压缩部分）
        _compression_cfg = _agent_cfg.get("compression", {})
        if not isinstance(_compression_cfg, dict):
            _compression_cfg = {}
        compression_threshold = float(_compression_cfg.get("threshold", 0.50))
        compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in ("true", "1", "yes")
        compression_target_ratio = float(_compression_cfg.get("target_ratio", 0.20))
        compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))

        # 从模型配置读取显式的 context_length 覆盖
        _model_cfg = _agent_cfg.get("model", {})
        if isinstance(_model_cfg, dict):
            _config_context_length = _model_cfg.get("context_length")
        else:
            _config_context_length = None
        if _config_context_length is not None:
            try:
                _config_context_length = int(_config_context_length)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid model.context_length in config.yaml: %r — "
                    "must be a plain integer (e.g. 256000, not '256K'). "
                    "Falling back to auto-detection.",
                    _config_context_length,
                )
                import sys
                print(
                    f"\n⚠ Invalid model.context_length in config.yaml: {_config_context_length!r}\n"
                    f"  Must be a plain integer (e.g. 256000, not '256K').\n"
                    f"  Falling back to auto-detected context window.\n",
                    file=sys.stderr,
                )
                _config_context_length = None

        # 存储以在 switch_model 中重用（因此配置覆盖在模型切换间保持）
        self._config_context_length = _config_context_length

        # 检查 custom_providers 的每个模型的 context_length
        if _config_context_length is None:
            try:
                from hermes_cli.config import get_compatible_custom_providers
                _custom_providers = get_compatible_custom_providers(_agent_cfg)
            except Exception:
                _custom_providers = _agent_cfg.get("custom_providers")
                if not isinstance(_custom_providers, list):
                    _custom_providers = []
            for _cp_entry in _custom_providers:
                if not isinstance(_cp_entry, dict):
                    continue
                _cp_url = (_cp_entry.get("base_url") or "").rstrip("/")
                if _cp_url and _cp_url == self.base_url.rstrip("/"):
                    _cp_models = _cp_entry.get("models", {})
                    if isinstance(_cp_models, dict):
                        _cp_model_cfg = _cp_models.get(self.model, {})
                        if isinstance(_cp_model_cfg, dict):
                            _cp_ctx = _cp_model_cfg.get("context_length")
                            if _cp_ctx is not None:
                                try:
                                    _config_context_length = int(_cp_ctx)
                                except (TypeError, ValueError):
                                    logger.warning(
                                        "Invalid context_length for model %r in "
                                        "custom_providers: %r — must be a plain "
                                        "integer (e.g. 256000, not '256K'). "
                                        "Falling back to auto-detection.",
                                        self.model, _cp_ctx,
                                    )
                                    import sys
                                    print(
                                        f"\n⚠ Invalid context_length for model {self.model!r} in custom_providers: {_cp_ctx!r}\n"
                                        f"  Must be a plain integer (e.g. 256000, not '256K').\n"
                                        f"  Falling back to auto-detected context window.\n",
                                        file=sys.stderr,
                                    )
                    break
        
        # 选择上下文引擎：配置驱动（如内存提供商）。
        # 1. 检查 config.yaml context.engine 设置
        # 2. 检查 plugins/context_engine/<name>/ 目录（仓库附带）
        # 3. 检查通用插件系统（用户安装的插件）
        # 4. 回退到内置 ContextCompressor
        _selected_engine = None
        _engine_name = "compressor"  # default
        try:
            _ctx_cfg = _agent_cfg.get("context", {}) if isinstance(_agent_cfg, dict) else {}
            _engine_name = _ctx_cfg.get("engine", "compressor") or "compressor"
        except Exception:
            pass

        if _engine_name != "compressor":
            # 尝试从 plugins/context_engine/<name>/ 加载
            try:
                from plugins.context_engine import load_context_engine
                _selected_engine = load_context_engine(_engine_name)
            except Exception as _ce_load_err:
                logger.debug("Context engine load from plugins/context_engine/: %s", _ce_load_err)

            # 尝试通用插件系统作为回退
            if _selected_engine is None:
                try:
                    from hermes_cli.plugins import get_plugin_context_engine
                    _candidate = get_plugin_context_engine()
                    if _candidate and _candidate.name == _engine_name:
                        _selected_engine = _candidate
                except Exception:
                    pass

            if _selected_engine is None:
                logger.warning(
                    "Context engine '%s' not found — falling back to built-in compressor",
                    _engine_name,
                )
        # 否则：配置说 "compressor" — 使用内置，不要自动激活插件

        if _selected_engine is not None:
            self.context_compressor = _selected_engine
            # 解析插件引擎的 context_length — 镜像 switch_model() 路径
            from agent.model_metadata import get_model_context_length
            _plugin_ctx_len = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                config_context_length=_config_context_length,
                provider=self.provider,
            )
            self.context_compressor.update_model(
                model=self.model,
                context_length=_plugin_ctx_len,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                provider=self.provider,
            )
            if not self.quiet_mode:
                logger.info("Using context engine: %s", _selected_engine.name)
        else:
            self.context_compressor = ContextCompressor(
                model=self.model,
                threshold_percent=compression_threshold,
                protect_first_n=3,
                protect_last_n=compression_protect_last,
                summary_target_ratio=compression_target_ratio,
                summary_model_override=None,
                quiet_mode=self.quiet_mode,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                config_context_length=_config_context_length,
                provider=self.provider,
                api_mode=self.api_mode,
            )
        self.compression_enabled = compression_enabled

        # 拒绝上下文窗口低于可靠工具调用工作流
        # 所需最小值的模型（64K 令牌）。
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        _ctx = getattr(self.context_compressor, "context_length", 0)
        if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Model {self.model} has a context window of {_ctx:,} tokens, "
                f"which is below the minimum {MINIMUM_CONTEXT_LENGTH:,} required "
                f"by Hermes Agent.  Choose a model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context, or set "
                f"model.context_length in config.yaml to override."
            )

        # 注入上下文引擎工具模式（例如 lcm_grep、lcm_describe、lcm_expand）
        self._context_engine_tool_names: set = set()
        if hasattr(self, "context_compressor") and self.context_compressor and self.tools is not None:
            for _schema in self.context_compressor.get_tool_schemas():
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                _tname = _schema.get("name", "")
                if _tname:
                    self.valid_tool_names.add(_tname)
                    self._context_engine_tool_names.add(_tname)

        # 通知上下文引擎会话开始
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_start(
                    self.session_id,
                    hermes_home=str(get_hermes_home()),
                    platform=self.platform or "cli",
                    model=self.model,
                    context_length=getattr(self.context_compressor, "context_length", 0),
                )
            except Exception as _ce_err:
                logger.debug("Context engine on_session_start: %s", _ce_err)

        self._subdirectory_hints = SubdirectoryHintTracker(
            working_dir=os.getenv("TERMINAL_CWD") or None,
        )
        self._user_turn_count = 0

        # 会话的累积令牌使用量
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # ── Ollama num_ctx 注入 ──
        # Ollama 默认为 2048 上下文，无论模型的能力如何。
        # 当针对 Ollama 服务器运行时，检测模型的最大上下文
        # 并在每个聊天请求上传递 num_ctx，以便使用完整的窗口。
        # 用户覆盖：在 config.yaml 中设置 model.ollama_num_ctx 以限制 VRAM 使用。
        self._ollama_num_ctx: int | None = None
        _ollama_num_ctx_override = None
        if isinstance(_model_cfg, dict):
            _ollama_num_ctx_override = _model_cfg.get("ollama_num_ctx")
        if _ollama_num_ctx_override is not None:
            try:
                self._ollama_num_ctx = int(_ollama_num_ctx_override)
            except (TypeError, ValueError):
                logger.debug("Invalid ollama_num_ctx config value: %r", _ollama_num_ctx_override)
        if self._ollama_num_ctx is None and self.base_url and is_local_endpoint(self.base_url):
            try:
                _detected = query_ollama_num_ctx(self.model, self.base_url)
                if _detected and _detected > 0:
                    self._ollama_num_ctx = _detected
            except Exception as exc:
                logger.debug("Ollama num_ctx detection failed: %s", exc)
        if self._ollama_num_ctx and not self.quiet_mode:
            logger.info(
                "Ollama num_ctx: will request %d tokens (model max from /api/show)",
                self._ollama_num_ctx,
            )

        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {self.context_compressor.threshold_tokens:,})")
            else:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (auto-compression disabled)")

        # 立即检查，以便 CLI 用户在启动时看到警告。
        # 网关 status_callback 尚未连接，因此任何警告都存储在
        # _compression_warning 中并在第一次 run_conversation() 中重播。
        self._compression_warning = None
        self._check_compression_model_feasibility()

        # 为每回合恢复快照主要运行时。当回退
        # 在一个回合中激活时，下一回合恢复这些值，
        # 以便首选模型每次都有新的尝试。
        # 使用单个字典，因此新状态字段易于添加，而无需 N 个单独属性。
        _cc = self.context_compressor
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            # _try_activate_fallback() 覆盖的上下文引擎状态。
            # 对 model/base_url/api_key/provider 使用 getattr，因为插件
            # 引擎可能没有这些（它们是 ContextCompressor 特定的）。
            "compressor_model": getattr(_cc, "model", self.model),
            "compressor_base_url": getattr(_cc, "base_url", self.base_url),
            "compressor_api_key": getattr(_cc, "api_key", ""),
            "compressor_provider": getattr(_cc, "provider", self.provider),
            "compressor_context_length": _cc.context_length,
            "compressor_threshold_tokens": _cc.threshold_tokens,
        }
        if self.api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

    def reset_session_state(self):
        """将所有会话范围的令牌计数器重置为 0 以开始新会话。
        
        此方法封装了所有会话级别指标的重置逻辑，
        包括：
        - 令牌使用计数器（输入、输出、总计、提示、完成）
        - 缓存读/写令牌
        - API 调用次数
        - 推理令牌
        - 估算成本跟踪
        - 上下文压缩器内部计数器
        
        该方法使用 ``hasattr`` 检查安全地处理可选属性
        （例如上下文压缩器）。
        
        这将计数器重置逻辑保持在一个地方的 DRY 和可维护性，
        而不是分散在多个方法中。
        """
        # 令牌使用计数器
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # 回合计数器（在 reset_session_state 首次编写后添加 — #2635）
        self._user_turn_count = 0

        # 上下文引擎重置（适用于内置压缩器和插件）
        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.on_session_reset()
    
    def switch_model(self, new_model, new_provider, api_key='', base_url='', api_mode=''):
        """就地切换实时代理的模型/提供商。

        在 ``model_switch.switch_model()`` 解析凭据并
        验证模型后，由 /model 命令处理器（CLI 和网关）调用。
        此方法执行实际的运行时交换：重建客户端、
        更新缓存标志和刷新上下文压缩器。

        该实现镜像了 ``_try_activate_fallback()`` 的
        客户端交换逻辑，但也更新 ``_primary_runtime``，
        以便更改在回合间持续存在（不同于回合范围的回退）。
        """
        import logging
        import re as _re
        from hermes_cli.providers import determine_api_mode

        # ── 如果未提供，确定 api_mode ──
        if not api_mode:
            api_mode = determine_api_mode(new_provider, base_url)

        # 深度防御：确保 OpenCode base_url 不会将尾随的
        # /v1 带入 anthropic_messages 客户端，这将导致 SDK
        # 命中 /v1/v1/messages。`model_switch.switch_model()` 已经剥离了
        # 这个，但我们在这里防护，以便任何直接调用者（未来的代码路径、
        # 测试）不能重新引入双 /v1 404 错误。
        if (
            api_mode == "anthropic_messages"
            and new_provider in ("opencode-zen", "opencode-go")
            and isinstance(base_url, str)
            and base_url
        ):
            base_url = _re.sub(r"/v1/?$", "", base_url)

        old_model = self.model
        old_provider = self.provider

        # ── 交换核心运行时字段 ──
        self.model = new_model
        self.provider = new_provider
        self.base_url = base_url or self.base_url
        self.api_mode = api_mode
        if api_key:
            self.api_key = api_key

        # ── 构建新客户端 ──
        if api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            # 仅当提供商实际上是 Anthropic 时才回退到 ANTHROPIC_TOKEN。
            # 其他 anthropic_messages 提供商（MiniMax、Alibaba 等）必须使用自己的
            # API 密钥 — 回退会将 Anthropic 凭据发送到第三方端点。
            _is_native_anthropic = new_provider == "anthropic"
            effective_key = (api_key or self.api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or self.api_key or "")
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url or getattr(self, "_anthropic_base_url", None)
            self._anthropic_client = build_anthropic_client(
                effective_key, self._anthropic_base_url,
            )
            self._is_anthropic_oauth = _is_oauth_token(effective_key)
            self.client = None
            self._client_kwargs = {}
        else:
            effective_key = api_key or self.api_key
            effective_base = base_url or self.base_url
            self._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            self.client = self._create_openai_client(
                dict(self._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        # ── 重新评估提示缓存 ──
        is_native_anthropic = api_mode == "anthropic_messages" and new_provider == "anthropic"
        self._use_prompt_caching = (
            ("openrouter" in (self.base_url or "").lower() and "claude" in new_model.lower())
            or is_native_anthropic
        )

        # ── 更新上下文压缩器 ──
        if hasattr(self, "context_compressor") and self.context_compressor:
            from agent.model_metadata import get_model_context_length
            new_context_length = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                provider=self.provider,
                config_context_length=getattr(self, "_config_context_length", None),
            )
            self.context_compressor.update_model(
                model=self.model,
                context_length=new_context_length,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                provider=self.provider,
                api_mode=self.api_mode,
            )

        # ── 使缓存的系统提示失效，以便下一回合重建 ──
        self._cached_system_prompt = None

        # ── 更新 _primary_runtime 以便更改在回合间持续存在 ──
        _cc = self.context_compressor if hasattr(self, "context_compressor") and self.context_compressor else None
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            "compressor_model": getattr(_cc, "model", self.model) if _cc else self.model,
            "compressor_base_url": getattr(_cc, "base_url", self.base_url) if _cc else self.base_url,
            "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
            "compressor_provider": getattr(_cc, "provider", self.provider) if _cc else self.provider,
            "compressor_context_length": _cc.context_length if _cc else 0,
            "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
        }
        if api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

        # ── 重置回退状态 ──
        self._fallback_activated = False
        self._fallback_index = 0

        logging.info(
            "Model switched in-place: %s (%s) -> %s (%s)",
            old_model, old_provider, new_model, new_provider,
        )

    def _safe_print(self, *args, **kwargs):
        """静默处理断管/关闭 stdout 的打印。

        在无头环境（systemd、Docker、nohup）中，stdout 可能在
        会话中途变得不可用。原始的 ``print()`` 会引发 ``OSError``，
        这可能导致 cron 作业崩溃并丢失已完成的工作。

        内部通过 ``self._print_fn``（默认：内置 ``print``）路由，
        以便调用者（如 CLI）可以注入正确处理 ANSI 转义序列的
        渲染器（例如 prompt_toolkit 的 ``print_formatted_text(ANSI(...))``），
        而无需触及此方法。
        """
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except (OSError, ValueError):
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """详细打印 — 在主动流式传输令牌时被抑制。

        对于应该始终显示的错误/警告消息，即使
        在流式播放（TTS 或显示）期间，也传递 ``force=True``。

        在工具执行期间（``_executing_tools`` 为 True），即使注册了
        流消费者，也允许打印，因为此时没有
        令牌正在流式传输。

        在主响应已交付且剩余工具调用是响应后清理
        （``_mute_post_response``）时，所有非强制输出都被抑制。

        ``suppress_status_output`` 是更严格的 CLI 自动化模式，
        用于可解析的单查询流程，如 ``hermes chat -q``。
        在该模式下，所有通过 ``_vprint`` 路由的状态/诊断打印
        都被抑制，以保持 stdout 的机器可读性。
        """
        if getattr(self, "suppress_status_output", False):
            return
        if not force and getattr(self, "_mute_post_response", False):
            return
        if not force and self._has_stream_consumers() and not self._executing_tools:
            return
        self._safe_print(*args, **kwargs)

    def _should_start_quiet_spinner(self) -> bool:
        """当安静模式旋转器输出有安全接收器时返回 True。

        在无头/stdio 协议环境中，没有自定义 ``_print_fn`` 的
        原始旋转器会回退到 ``sys.stdout`` 并可能损坏协议
        流，如 ACP JSON-RPC。仅当以下情况之一时才允许安静旋转器：
        - 输出通过 ``_print_fn`` 明确重新路由；或
        - stdout 是真实的 TTY。
        """
        if self._print_fn is not None:
            return True
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def _should_emit_quiet_tool_messages(self) -> bool:
        """当安静模式工具摘要应直接打印时返回 True。

        安静模式由交互式 CLI 和嵌入式/库调用者使用。
        当没有回调拥有渲染时，CLI 可能仍希望有紧凑的进度提示。
        另一方面，嵌入式/库调用者期望安静模式真正静默。
        """
        return (
            self.quiet_mode
            and not self.tool_progress_callback
            and getattr(self, "platform", "") == "cli"
        )

    def _emit_status(self, message: str) -> None:
        """向 CLI 和网关通道发出生命周期状态消息。

        CLI 用户通过 ``_vprint(force=True)`` 看到消息，因此无论
        详细/安静模式如何，它始终可见。网关消费者
        通过 ``status_callback("lifecycle", ...)`` 接收它。

        此辅助函数从不引发 — 异常被吞噬，因此它不会
        中断重试/回退逻辑。
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("lifecycle", message)
            except Exception:
                logger.debug("status_callback error in _emit_status", exc_info=True)

    def _current_main_runtime(self) -> Dict[str, str]:
        """返回会话范围辅助路由的实时主运行时。"""
        return {
            "model": getattr(self, "model", "") or "",
            "provider": getattr(self, "provider", "") or "",
            "base_url": getattr(self, "base_url", "") or "",
            "api_key": getattr(self, "api_key", "") or "",
            "api_mode": getattr(self, "api_mode", "") or "",
        }

    def _check_compression_model_feasibility(self) -> None:
        """如果辅助压缩模型的上下文窗口小于主模型的
        压缩阈值，则在会话开始时发出警告。

        当辅助模型无法容纳需要总结的内容时，
        压缩要么完全失败（LLM 调用错误），要么产生
        严重截断的摘要。

        在 ``__init__`` 期间调用，以便 CLI 用户立即看到警告
        （通过 ``_vprint``）。网关在构造后设置 ``status_callback``，
        因此 ``_replay_compression_warning()`` 在第一次
        ``run_conversation()`` 调用时通过回调重新发送
        存储的警告。
        """
        if not self.compression_enabled:
            return
        try:
            from agent.auxiliary_client import get_text_auxiliary_client
            from agent.model_metadata import get_model_context_length

            client, aux_model = get_text_auxiliary_client(
                "compression",
                main_runtime=self._current_main_runtime(),
            )
            if client is None or not aux_model:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "No auxiliary LLM provider for compression — "
                    "summaries will be unavailable."
                )
                return

            aux_base_url = str(getattr(client, "base_url", ""))
            aux_api_key = str(getattr(client, "api_key", ""))

            # 读取压缩模型的用户配置的 context_length。
            # 自定义端点通常不支持 /models API 查询，因此
            # get_model_context_length() 会回退到 128K 默认值，
            # 忽略显式配置值。将其作为最高优先级提示传递，
            # 以便始终尊重配置的值。
            _aux_cfg = (self.config or {}).get("auxiliary", {}).get("compression", {})
            _aux_context_config = _aux_cfg.get("context_length") if isinstance(_aux_cfg, dict) else None
            if _aux_context_config is not None:
                try:
                    _aux_context_config = int(_aux_context_config)
                except (TypeError, ValueError):
                    _aux_context_config = None

            aux_context = get_model_context_length(
                aux_model,
                base_url=aux_base_url,
                api_key=aux_api_key,
                config_context_length=_aux_context_config,
            )

            threshold = self.context_compressor.threshold_tokens
            if aux_context < threshold:
                # 建议一个适合辅助模型的阈值，
                # 向下取整到干净的百分比。
                safe_pct = int((aux_context / self.context_compressor.context_length) * 100)
                msg = (
                    f"⚠ Compression model ({aux_model}) context "
                    f"is {aux_context:,} tokens, but the main model's "
                    f"compression threshold is {threshold:,} tokens. "
                    f"Context compression will not be possible — the "
                    f"content to summarise will exceed the auxiliary "
                    f"model's context window.\n"
                    f"  Fix options (config.yaml):\n"
                    f"  1. Use a larger compression model:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{threshold:,}+-context>\n"
                    f"  2. Lower the compression threshold to fit "
                    f"the current model:\n"
                    f"       compression:\n"
                    f"         threshold: 0.{safe_pct:02d}"
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "Auxiliary compression model %s has %d token context, "
                    "below the main model's compression threshold of %d "
                    "tokens — compression summaries will fail or be "
                    "severely truncated.",
                    aux_model,
                    aux_context,
                    threshold,
                )
        except Exception as exc:
            logger.debug(
                "Compression feasibility check failed (non-fatal): %s", exc
            )

    def _replay_compression_warning(self) -> None:
        """通过 ``status_callback`` 重新发送压缩警告。

        在 ``__init__`` 期间，网关的 ``status_callback`` 尚未连接，
        因此 ``_emit_status`` 只能到达 ``_vprint``（CLI）。
        此方法在第一次 ``run_conversation()`` 开始时调用一次 —
        到那时网关已设置回调，因此每个平台
        （Telegram、Discord、Slack 等）都会收到警告。
        """
        msg = getattr(self, "_compression_warning", None)
        if msg and self.status_callback:
            try:
                self.status_callback("lifecycle", msg)
            except Exception:
                pass

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """当基础 URL 目标是 OpenAI 的原生 API 时返回 True。"""
        url = (base_url or self._base_url_lower).lower()
        return "api.openai.com" in url and "openrouter" not in url

    def _is_openrouter_url(self) -> bool:
        """当基础 URL 目标是 OpenRouter 时返回 True。"""
        return "openrouter" in self._base_url_lower

    @staticmethod
    def _model_requires_responses_api(model: str) -> bool:
        """对于需要 Responses API 路径的模型返回 True。

        GPT-5.x 模型在 /v1/chat/completions 上被 OpenAI
        和 OpenRouter 拒绝（错误：``unsupported_api_for_model``）。
        检测这些模型，以便无论哪个提供商提供模型，
        都设置正确的 api_mode。
        """
        m = model.lower()
        # 去除供应商前缀（例如 "openai/gpt-5.4" → "gpt-5.4"）
        if "/" in m:
            m = m.rsplit("/", 1)[-1]
        return m.startswith("gpt-5")

    @staticmethod
    def _provider_model_requires_responses_api(
        model: str,
        *,
        provider: Optional[str] = None,
    ) -> bool:
        """当此提供商/模型对应使用 Responses API 时返回 True。"""
        normalized_provider = (provider or "").strip().lower()
        if normalized_provider == "copilot":
            try:
                from hermes_cli.models import _should_use_copilot_responses_api
                return _should_use_copilot_responses_api(model)
            except Exception:
                # 如果由于任何原因无法使用 Copilot 特定的
                # 逻辑，则回退到通用的 GPT-5 规则。
                pass
        return AIAgent._model_requires_responses_api(model)

    def _max_tokens_param(self, value: int) -> dict:
        """返回当前提供商的正确 max tokens 参数。
        
        OpenAI 的较新模型（gpt-4o、o 系列、gpt-5+）需要
        'max_completion_tokens'。OpenRouter、本地模型和较旧的
        OpenAI 模型使用 'max_tokens'。
        """
        if self._is_direct_openai_url():
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    def _has_content_after_think_block(self, content: str) -> bool:
        """
        检查内容在任何推理/思考块之后是否有实际文本。

        这检测模型仅输出推理但没有实际响应的情况，
        这表明应该重试的不完整生成。
        必须与 _strip_think_blocks() 标签变体保持同步。

        参数：
            content: 要检查的助手消息内容

        返回：
            如果思考块后有有意义的内容则为 True，否则为 False
        """
        if not content:
            return False

        # 删除所有推理标签变体（必须匹配 _strip_think_blocks）
        cleaned = self._strip_think_blocks(content)

        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())
    
    def _strip_think_blocks(self, content: str) -> str:
        """Remove reasoning/thinking blocks from content, returning only visible text.

        Handles four cases:
          1. Closed tag pairs (``<think>…</think>``) — the common path when
             the provider emits complete reasoning blocks.
          2. Unterminated open tag at a block boundary (start of text or
             after a newline) — e.g. MiniMax M2.7 / NIM endpoints where the
             closing tag is dropped.  Everything from the open tag to end
             of string is stripped.  The block-boundary check mirrors
             ``gateway/stream_consumer.py``'s filter so models that mention
             ``<think>`` in prose aren't over-stripped.
          3. Stray orphan open/close tags that slip through.
          4. Tag variants: ``<think>``, ``<thinking>``, ``<reasoning>``,
             ``<REASONING_SCRATCHPAD>``, ``<thought>`` (Gemma 4), all
             case-insensitive.
        """
        if not content:
            return ""
        # 1. 闭合标签对 — 对所有变体不区分大小写，以便
        #    混合大小写标签（<THINK>、<Thinking>）不会漏到
        #    未终止标签传递中并带走尾随内容。
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # 2. 未终止的推理块 — 块边界的开放标签
        #    （文本开头或换行后）没有匹配的闭合。
        #    从标签剥离到字符串结尾。修复 #8878 / #9568
        #    （MiniMax M2.7 将原始推理泄漏到助手内容中）。
        content = re.sub(
            r'(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*$',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # 3. 漏过的孤立开放/闭合标签。
        content = re.sub(
            r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*',
            '',
            content,
            flags=re.IGNORECASE,
        )
        return content

    @staticmethod
    def _has_natural_response_ending(content: str) -> bool:
        """启发式：可见助手文本看起来是否故意完成？"""
        if not content:
            return False
        stripped = content.rstrip()
        if not stripped:
            return False
        if stripped.endswith("```"):
            return True
        return stripped[-1] in '.!?:)"\']}。！？：）】」』》'

    def _is_ollama_glm_backend(self) -> bool:
        """检测受 Ollama/GLM 停止误报影响的窄后端系列。"""
        model_lower = (self.model or "").lower()
        provider_lower = (self.provider or "").lower()
        if "glm" not in model_lower and provider_lower != "zai":
            return False
        if "ollama" in self._base_url_lower or ":11434" in self._base_url_lower:
            return True
        return bool(self.base_url and is_local_endpoint(self.base_url))

    def _should_treat_stop_as_truncated(
        self,
        finish_reason: str,
        assistant_message,
        messages: Optional[list] = None,
    ) -> bool:
        """检测 Ollama 托管的 GLM 模型的保守 stop->length 误报。"""
        if finish_reason != "stop" or self.api_mode != "chat_completions":
            return False
        if not self._is_ollama_glm_backend():
            return False
        if not any(
            isinstance(msg, dict) and msg.get("role") == "tool"
            for msg in (messages or [])
        ):
            return False
        if assistant_message is None or getattr(assistant_message, "tool_calls", None):
            return False

        content = getattr(assistant_message, "content", None)
        if not isinstance(content, str):
            return False

        visible_text = self._strip_think_blocks(content).strip()
        if not visible_text:
            return False
        if len(visible_text) < 20 or not re.search(r"\s", visible_text):
            return False

        return not self._has_natural_response_ending(visible_text)

    def _looks_like_codex_intermediate_ack(
        self,
        user_message: str,
        assistant_content: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """检测应该继续而不是结束回合的规划/确认消息。"""
        if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
            return False

        assistant_text = self._strip_think_blocks(assistant_content or "").strip().lower()
        if not assistant_text:
            return False
        if len(assistant_text) > 1200:
            return False

        has_future_ack = bool(
            re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
        )
        if not has_future_ack:
            return False

        action_markers = (
            "look into",
            "look at",
            "inspect",
            "scan",
            "check",
            "analyz",
            "review",
            "explore",
            "read",
            "open",
            "run",
            "test",
            "fix",
            "debug",
            "search",
            "find",
            "walkthrough",
            "report back",
            "summarize",
        )
        workspace_markers = (
            "directory",
            "current directory",
            "current dir",
            "cwd",
            "repo",
            "repository",
            "codebase",
            "project",
            "folder",
            "filesystem",
            "file tree",
            "files",
            "path",
        )

        user_text = (user_message or "").strip().lower()
        user_targets_workspace = (
            any(marker in user_text for marker in workspace_markers)
            or "~/" in user_text
            or "/" in user_text
        )
        assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
        assistant_targets_workspace = any(
            marker in assistant_text for marker in workspace_markers
        )
        return (user_targets_workspace or assistant_targets_workspace) and assistant_mentions_action
    
    
    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """
        从助手消息中提取推理/思考内容。
        
        OpenRouter 和各种提供商可以以多种格式返回推理：
        1. message.reasoning - 直接推理字段（DeepSeek、Qwen 等）
        2. message.reasoning_content - 替代字段（Moonshot AI、Novita 等）
        3. message.reasoning_details - {type, summary, ...} 对象数组（OpenRouter 统一）
        
        参数：
            assistant_message: 来自 API 响应的助手消息对象
            
        返回：
            合并的推理文本，如果未找到推理则为 None
        """
        reasoning_parts = []
        
        # 检查直接推理字段
        if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)
        
        # 检查 reasoning_content 字段（某些提供商使用的替代名称）
        if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
            # 如果与 reasoning 相同则不要重复
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)
        
        # 检查 reasoning_details 数组（OpenRouter 统一格式）
        # 格式：[{"type": "reasoning.summary", "summary": "...", ...}, ...]
        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    # 从推理细节对象中提取摘要
                    summary = (
                        detail.get('summary')
                        or detail.get('thinking')
                        or detail.get('content')
                        or detail.get('text')
                    )
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        # 一些提供商将推理直接嵌入助手内容中，
        # 而不是返回结构化推理字段。仅当未找到结构化推理时
        # 才回退到内联提取。
        content = getattr(assistant_message, "content", None)
        if not reasoning_parts and isinstance(content, str) and content:
            inline_patterns = (
                r"<think>(.*?)</think>",
                r"<thinking>(.*?)</thinking>",
                r"<thought>(.*?)</thought>",
                r"<reasoning>(.*?)</reasoning>",
                r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
            )
            for pattern in inline_patterns:
                flags = re.DOTALL | re.IGNORECASE
                for block in re.findall(pattern, content, flags=flags):
                    cleaned = block.strip()
                    if cleaned and cleaned not in reasoning_parts:
                        reasoning_parts.append(cleaned)
        
        # 合并所有推理部分
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        
        return None

    def _cleanup_task_resources(self, task_id: str) -> None:
        """清理给定任务的 VM 和浏览器资源。

        当活动终端环境标记为持久化（``persistent_filesystem=True``）时，
        跳过 ``cleanup_vm``，以便长期存在的沙箱容器在回合间保持存活。
        ``terminal_tool._cleanup_inactive_envs`` 中的空闲收割器
        仍然在超过 ``terminal.lifetime_seconds`` 时拆除它们。
        非持久化后端像以前一样每回合拆除，以防止资源泄漏
        （此钩子对 Morph 后端的原始意图，见提交 fbd3a2fd）。
        """
        try:
            if is_persistent_env(task_id):
                if self.verbose_logging:
                    logging.debug(
                        f"Skipping per-turn cleanup_vm for persistent env {task_id}; "
                        f"idle reaper will handle it."
                    )
            else:
                cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

    # ------------------------------------------------------------------
    # 后台内存/技能审查
    # ------------------------------------------------------------------

    _MEMORY_REVIEW_PROMPT = (
        "回顾上面的对话，如果合适的话考虑保存到内存。\n\n"
        "重点关注：\n"
        "1. 用户是否透露了关于自己的信息——他们的角色、愿望、\n"
        "偏好或值得记住的个人细节？\n"
        "2. 用户是否表达了关于你应该如何行为、他们的工作\n"
        "风格或他们希望你运行方式的期望？\n\n"
        "如果有突出的内容，使用内存工具保存它。\n"
        "如果没有值得保存的内容，只需说'Nothing to save.'然后停止。"
    )

    _SKILL_REVIEW_PROMPT = (
        "回顾上面的对话，如果合适的话考虑保存或更新一个技能。\n\n"
        "重点关注：是否使用了非平凡的方法来完成需要试错的\n"
        "任务，或者由于过程中的经验发现而改变了方向，或者\n"
        "用户期望或期望不同的方法或结果？\n\n"
        "如果相关技能已经存在，请用你学到的内容更新它。\n"
        "否则，如果方法是可重用的，则创建一个新技能。\n"
        "如果没有值得保存的内容，只需说'Nothing to save.'然后停止。"
    )

    _COMBINED_REVIEW_PROMPT = (
        "回顾上面的对话并考虑两件事：\n\n"
        "**内存**：用户是否透露了关于自己的信息——他们的角色、\n"
        "愿望、偏好或个人细节？用户是否表达了关于你应该如何行为、\n"
        "他们的工作风格或他们希望你运行方式的期望？\n"
        "如果是这样，使用内存工具保存。\n\n"
        "**技能**：是否使用了非平凡的方法来完成需要试错的\n"
        "任务，或者由于过程中的经验发现而改变了方向，或者\n"
        "用户期望或期望不同的方法或结果？如果相关技能\n"
        "已经存在，请更新它。否则，如果方法是可重用的，则创建一个新的。\n\n"
        "只有在有真正值得保存的内容时才行动。\n"
        "如果没有突出的内容，只需说'Nothing to save.'然后停止。"
    )

    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        self.skill_evolution.spawn_background_review(
            self,
            messages_snapshot,
            review_memory=review_memory,
            review_skills=review_skills,
        )
        return
        """生成后台线程以审查对话以进行内存/技能保存。

        创建具有与主会话相同模型、工具和上下文的完整 AIAgent 分支。
        审查提示作为分支对话中的下一个用户回合附加。
        直接写入共享的内存/技能存储。
        从不修改主对话历史或产生用户可见的输出。
        """
        import threading

        # 根据触发的触发器选择正确的提示
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            prompt = self._MEMORY_REVIEW_PROMPT
        else:
            prompt = self._SKILL_REVIEW_PROMPT

        def _run_review():
            import contextlib, os as _os
            review_agent = None
            try:
                with open(_os.devnull, "w") as _devnull, \
                     contextlib.redirect_stdout(_devnull), \
                     contextlib.redirect_stderr(_devnull):
                    review_agent = AIAgent(
                        model=self.model,
                        max_iterations=8,
                        quiet_mode=True,
                        platform=self.platform,
                        provider=self.provider,
                    )
                    review_agent._memory_store = self._memory_store
                    review_agent._memory_enabled = self._memory_enabled
                    review_agent._user_profile_enabled = self._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0

                    review_agent.run_conversation(
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                # 扫描审查代理的消息以查找成功的工具操作
                # 并向用户显示紧凑的摘要。
                actions = []
                for msg in getattr(review_agent, "_session_messages", []):
                    if not isinstance(msg, dict) or msg.get("role") != "tool":
                        continue
                    try:
                        data = json.loads(msg.get("content", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not data.get("success"):
                        continue
                    message = data.get("message", "")
                    target = data.get("target", "")
                    if "created" in message.lower():
                        actions.append(message)
                    elif "updated" in message.lower():
                        actions.append(message)
                    elif "added" in message.lower() or (target and "add" in message.lower()):
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "Entry added" in message:
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "removed" in message.lower() or "replaced" in message.lower():
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")

                if actions:
                    summary = " · ".join(dict.fromkeys(actions))
                    self._safe_print(f"  💾 {summary}")
                    _bg_cb = self.background_review_callback
                    if _bg_cb:
                        try:
                            _bg_cb(f"💾 {summary}")
                        except Exception:
                            pass

            except Exception as e:
                logger.debug("Background memory/skill review failed: %s", e)
            finally:
                # 关闭所有资源（httpx 客户端、子进程等），以便
                # GC 不会尝试在死掉的 asyncio 事件循环上清理它们
                # （这会产生"事件循环已关闭"错误）。
                if review_agent is not None:
                    try:
                        review_agent.close()
                    except Exception:
                        pass

        t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        t.start()

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """在持久化/返回之前重写当前回合的用户消息。

        某些调用路径需要仅 API 的用户消息变体，而不让
        那些合成文本泄漏到持久化的记录或恢复的会话
        历史中。当为活动回合配置覆盖时，就地修改
        内存中的消息列表，以便持久化和返回的历史保持干净。
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """在任何退出路径上将会话状态保存到 JSON 日志和 SQLite。

        确保对话永远不会丢失，即使在错误或提前返回时。
        当 ``persist_session=False``（临时辅助流程）时跳过。
        """
        if not self.persist_session:
            return
        self._apply_persist_user_message_override(messages)
        self._session_messages = messages
        self._save_session_log(messages)
        self._flush_messages_to_session_db(messages, conversation_history)

    def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """将任何未刷新的消息持久化到 SQLite 会话存储。

        使用 _last_flushed_db_idx 跟踪哪些消息已经
        写入，因此重复调用（来自多个退出路径）只写入
        真正的新消息 — 防止重复写入错误（#860）。
        """
        if not self._session_db:
            return
        self._apply_persist_user_message_override(messages)
        try:
            # 如果 create_session() 在启动时失败（例如瞬时锁），
            # 会话行可能尚不存在。ensure_session() 使用 INSERT OR
            # IGNORE，因此当行已经存在时它是无操作。
            self._session_db.ensure_session(
                self.session_id,
                source=self.platform or "cli",
                model=self.model,
            )
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self._session_db.append_message(
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning") if role == "assistant" else None,
                    reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                    codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        获取直到（但不包括）最后一个助手回合的消息。
        
        当我们需要"回滚"到对话中最后一个成功点时使用此方法，
        通常是在最终助手消息不完整或格式错误时。
        
        参数：
            messages: 完整消息列表
            
        返回：
            直到最后一个完整助手回合的消息（以用户/工具消息结束）
        """
        if not messages:
            return []
        
        # 查找最后一个助手消息的索引
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            # 未找到助手消息，返回所有消息
            return messages.copy()
        
        # 返回直到（不包括）最后一个助手消息的所有内容
        return messages[:last_assistant_idx]
    
    def _format_tools_for_system_message(self) -> str:
        """
        Format tool definitions for the system message in the trajectory format.
        
        返回：
            str: 工具定义的 JSON 字符串表示
        """
        if not self.tools:
            return "[]"
        
        # 将工具定义转换为轨迹中期望的格式
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None  # 匹配示例中的格式
            }
            formatted_tools.append(formatted_tool)
        
        return json.dumps(formatted_tools, ensure_ascii=False)
    
    def _convert_to_trajectory_format(self, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
        """
        将内部消息格式转换为轨迹格式以进行保存。
        
        参数：
            messages (List[Dict]): 内部消息历史
            user_query (str): 原始用户查询
            completed (bool): 对话是否成功完成
            
        返回：
            List[Dict]: 轨迹格式的消息
        """
        trajectory = []
        
        # 添加带有工具定义的系统消息
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )
        
        trajectory.append({
            "from": "system",
            "value": system_msg
        })
        
        # 将实际的用户提示（来自数据集）添加为第一个人类消息
        trajectory.append({
            "from": "human",
            "value": user_query
        })
        
        # 跳过第一条消息（用户查询），因为我们已经在上面添加了它。
        # 预填充消息仅在 API 调用时注入（不在消息列表中），
        # 因此这里不需要偏移调整。
        i = 1
        
        while i < len(messages):
            msg = messages[i]
            
            if msg["role"] == "assistant":
                # 检查此消息是否有工具调用
                if "tool_calls" in msg and msg["tool_calls"]:
                    # 格式化带有工具调用的助手消息
                    # 为轨迹存储在推理周围添加 <think> 标签
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    if msg.get("content") and msg["content"].strip():
                        # 将任何 <REASONING_SCRATCHPAD> 标签转换为 <think> 标签
                        # （当禁用原生思考且模型通过 XML 推理时使用）
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"
                    
                    # 添加包装在 XML 标签中的工具调用
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict): continue
                        # 解析参数 — 应该总是成功，因为我们在对话期间验证
                        # 但保留 try-except 作为安全网
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            # 这不应该发生，因为我们在对话期间验证和重试，
                            # 但如果发生了，记录警告并使用空字典
                            logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                            arguments = {}
                        
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                    
                    # 确保每个 gpt 回合都有一个 <think> 块（如果没有推理则为空）
                    # 以便格式对训练数据保持一致
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.rstrip()
                    })
                    
                    # 收集所有后续工具响应
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # 用 XML 标签格式化工具响应
                        tool_response = "<tool_response>\n"
                        
                        # 如果看起来像 JSON，尝试将工具内容解析为 JSON
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # 如果不是有效 JSON 则保持为字符串
                        
                        tool_index = len(tool_responses)
                        tool_name = (
                            msg["tool_calls"][tool_index]["function"]["name"]
                            if tool_index < len(msg["tool_calls"])
                            else "unknown"
                        )
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": tool_name,
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1
                    
                    # 将所有工具响应添加为单条消息
                    if tool_responses:
                        trajectory.append({
                            "from": "tool",
                            "value": "\n".join(tool_responses)
                        })
                        i = j - 1  # 跳过我们刚刚处理的工具消息
                
                else:
                    # 没有工具调用的常规助手消息
                    # 为轨迹存储在推理周围添加 <think> 标签
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    # 将任何 <REASONING_SCRATCHPAD> 标签转换为 <think> 标签
                    # （当禁用原生思考且模型通过 XML 推理时使用）
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)
                    
                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.strip()
                    })
            
            elif msg["role"] == "user":
                trajectory.append({
                    "from": "human",
                    "value": msg["content"]
                })
            
            i += 1
        
        return trajectory
    
    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """
        将对话轨迹保存到 JSONL 文件。
        
        参数：
            messages (List[Dict]): 完整消息历史
            user_query (str): 原始用户查询
            completed (bool): 对话是否成功完成
        """
        if not self.save_trajectories:
            return
        
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)
    
    @staticmethod
    def _summarize_api_error(error: Exception) -> str:
        """从 API 错误中提取人类可读的单行摘要。

        通过提取 <title> 标签而不是转储原始 HTML 来处理
        Cloudflare HTML 错误页面（502、503 等）。
        对于其他所有情况，回退到截断的 str(error)。
        """
        import re as _re
        raw = str(error)

        # Cloudflare / 代理 HTML 页面：获取 <title> 以获得干净的摘要
        if "<!DOCTYPE" in raw or "<html" in raw:
            m = _re.search(r"<title[^>]*>([^<]+)</title>", raw, _re.IGNORECASE)
            title = m.group(1).strip() if m else "HTML error page (title not found)"
            # 如果存在，也获取 Cloudflare Ray ID
            ray = _re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
            ray_id = ray.group(1).strip() if ray else None
            status_code = getattr(error, "status_code", None)
            parts = []
            if status_code:
                parts.append(f"HTTP {status_code}")
            parts.append(title)
            if ray_id:
                parts.append(f"Ray {ray_id}")
            return " — ".join(parts)

        # 来自 OpenAI/Anthropic SDK 的 JSON 主体错误
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
            if msg:
                status_code = getattr(error, "status_code", None)
                prefix = f"HTTP {status_code}: " if status_code else ""
                return f"{prefix}{msg[:300]}"

        # 回退：截断原始字符串但给予超过 200 个字符的空间
        status_code = getattr(error, "status_code", None)
        prefix = f"HTTP {status_code}: " if status_code else ""
        return f"{prefix}{raw[:500]}"

    def _mask_api_key_for_logs(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

    def _clean_error_message(self, error_msg: str) -> str:
        """
        清理错误消息以供用户显示，删除 HTML 内容并截断。
        
        参数：
            error_msg: 来自 API 或异常的原始错误消息
            
        返回：
            干净的、用户友好的错误消息
        """
        if not error_msg:
            return "Unknown error"
            
        # 删除 HTML 内容（CloudFlare 和网关错误页面常见）
        if error_msg.strip().startswith('<!DOCTYPE html') or '<html' in error_msg:
            return "Service temporarily unavailable (HTML error page returned)"
            
        # 删除换行符和过多的空白
        cleaned = ' '.join(error_msg.split())
        
        # 如果太长则截断
        if len(cleaned) > 150:
            cleaned = cleaned[:150] + "..."
            
        return cleaned

    @staticmethod
    def _extract_api_error_context(error: Exception) -> Dict[str, Any]:
        """从提供商错误中提取结构化的速率限制详细信息。"""
        context: Dict[str, Any] = {}

        body = getattr(error, "body", None)
        payload = None
        if isinstance(body, dict):
            payload = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(payload, dict):
            reason = payload.get("code") or payload.get("error")
            if isinstance(reason, str) and reason.strip():
                context["reason"] = reason.strip()
            message = payload.get("message") or payload.get("error_description")
            if isinstance(message, str) and message.strip():
                context["message"] = message.strip()
            for key in ("resets_at", "reset_at"):
                value = payload.get(key)
                if value not in (None, ""):
                    context["reset_at"] = value
                    break
            retry_after = payload.get("retry_after")
            if retry_after not in (None, "") and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass
            ratelimit_reset = headers.get("x-ratelimit-reset")
            if ratelimit_reset and "reset_at" not in context:
                context["reset_at"] = ratelimit_reset

        if "message" not in context:
            raw_message = str(error).strip()
            if raw_message:
                context["message"] = raw_message[:500]

        if "reset_at" not in context:
            message = context.get("message") or ""
            if isinstance(message, str):
                delay_match = re.search(r"quotaResetDelay[:\s\"]+(\\d+(?:\\.\\d+)?)(ms|s)", message, re.IGNORECASE)
                if delay_match:
                    value = float(delay_match.group(1))
                    seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                    context["reset_at"] = time.time() + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

        return context

    def _usage_summary_for_api_request_hook(self, response: Any) -> Optional[Dict[str, Any]]:
        """用于 ``post_api_request`` 插件的令牌桶（没有原始 ``response`` 对象）。"""
        if response is None:
            return None
        raw_usage = getattr(response, "usage", None)
        if not raw_usage:
            return None
        from dataclasses import asdict

        cu = normalize_usage(raw_usage, provider=self.provider, api_mode=self.api_mode)
        summary = asdict(cu)
        summary.pop("raw_usage", None)
        summary["prompt_tokens"] = cu.prompt_tokens
        summary["total_tokens"] = cu.total_tokens
        return summary

    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """
        为活动推理 API 转储调试友好的 HTTP 请求记录。

        从 api_kwargs 捕获请求主体（排除仅传输键
        如 timeout）。用于调试提供商端 4xx 失败，
        其中重试没有用。
        """
        try:
            body = copy.deepcopy(api_kwargs)
            body.pop("timeout", None)
            body = {k: v for k, v in body.items() if v is not None}

            api_key = None
            try:
                api_key = getattr(self.client, "api_key", None)
            except Exception as e:
                logger.debug("Could not extract API key for debug dump: %s", e)

            dump_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "reason": reason,
                "request": {
                    "method": "POST",
                    "url": f"{self.base_url.rstrip('/')}{'/responses' if self.api_mode == 'codex_responses' else '/chat/completions'}",
                    "headers": {
                        "Authorization": f"Bearer {self._mask_api_key_for_logs(api_key)}",
                        "Content-Type": "application/json",
                    },
                    "body": body,
                },
            }

            if error is not None:
                error_info: Dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for attr_name in ("status_code", "request_id", "code", "param", "type"):
                    attr_value = getattr(error, attr_name, None)
                    if attr_value is not None:
                        error_info[attr_name] = attr_value

                body_attr = getattr(error, "body", None)
                if body_attr is not None:
                    error_info["body"] = body_attr

                response_obj = getattr(error, "response", None)
                if response_obj is not None:
                    try:
                        error_info["response_status"] = getattr(response_obj, "status_code", None)
                        error_info["response_text"] = response_obj.text
                    except Exception as e:
                        logger.debug("Could not extract error response details: %s", e)

                dump_payload["error"] = error_info

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dump_file = self.logs_dir / f"request_dump_{self.session_id}_{timestamp}.json"
            dump_file.write_text(
                json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            self._vprint(f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}")

            if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
                print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(f"Failed to dump API request debug payload: {dump_error}")
            return None

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """将 REASONING_SCRATCHPAD 转换为 think 标签并清理空白。"""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """
        将完整的原始会话保存到 JSON 文件。

        完全按照代理看到的方式存储每条消息：用户消息、
        助手消息（带有推理、finish_reason、tool_calls）、
        工具响应（带有 tool_call_id、tool_name）和注入的系统
        消息（压缩摘要、todo 快照等）。

        REASONING_SCRATCHPAD 标签转换为 <think> 块以保持一致性。
        在每个回合后覆盖，因此它始终反映最新状态。
        """
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # 清理会话日志的助手内容
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                cleaned.append(msg)

            # 保护：永远不要用较少的消息覆盖较大的会话日志。
            # 这防止当 --resume 加载一个其消息未完全写入 SQLite 的会话时的数据丢失
            # — 恢复的代理以部分历史开始，否则会破坏完整的 JSON 日志。
            if self.session_log_file.exists():
                try:
                    existing = json.loads(self.session_log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # 损坏的现有文件 — 允许覆盖

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": self._cached_system_prompt or "",
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                self.session_log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")
    
    def interrupt(self, message: str = None) -> None:
        """
        请求代理中断其当前的工具调用循环。
        
        从另一个线程（例如输入处理程序、消息接收器）调用此方法，
        以优雅地停止代理并处理新消息。
        
        还向长时间运行的工具执行（例如终端命令）发出信号
        以提前终止，以便代理可以立即响应。
        
        参数：
            message: 触发中断的可选新消息。
                     如果提供，代理将将其包含在其响应上下文中。
        
        示例（CLI）：
            # 在单独的输入线程中：
            if user_typed_something:
                agent.interrupt(user_input)
        
        示例（消息传递）：
            # 当活动会话到达新消息时：
            if session_has_running_agent:
                running_agent.interrupt(new_message.text)
        """
        self._interrupt_requested = True
        self._interrupt_message = message
        # 向所有工具发出信号以立即中止任何进行中的操作。
        # 将中断范围限制在此代理的执行线程，以便
        # 在同一进程（网关）中运行的其他代理不受影响。
        if self._execution_thread_id is not None:
            _set_interrupt(True, self._execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            # 中断在 run_conversation() 完成之前到达，
            # 将代理绑定到其执行线程。延迟工具级别的
            # 中断信号，直到启动完成，而不是错误地
            # 针对调用者线程。
            self._interrupt_thread_signal_pending = True
        # 扩展到并发工具工作线程。这些工作线程在自己的
        # tid（ThreadPoolExecutor 工作线程）上运行工具，因此
        # 工具内的 `is_interrupted()` 只有在它们的特定 tid 在
        # `_interrupted_threads` 集合中时才会看到中断。
        # 没有此传播，已经运行的并发工具（例如挂在网络 I/O 上的
        # 终端命令）永远不会注意到中断，必须运行到自己的
        # 超时。请参阅 `_run_tool` 了解匹配的进入/退出簿记。
        # `getattr` 回退覆盖通过 object.__new__ 构建 AIAgent 并跳过 __init__ 的测试存根。
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(True, _wtid)
                except Exception:
                    pass
        # 将中断传播到任何运行的子代理（子代理委托）
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
    
    def clear_interrupt(self) -> None:
        """清除任何待处理的中断请求和每线程工具中断信号。"""
        self._interrupt_requested = False
        self._interrupt_message = None
        self._interrupt_thread_signal_pending = False
        if self._execution_thread_id is not None:
            _set_interrupt(False, self._execution_thread_id)
        # 同时清除任何并发工具工作线程位。跟踪的
        # 工作线程通常在退出时清除自己的位，但这里的显式
        # 清除保证没有过时的中断可以在回合边界后存活
        # 并在后续的、不相关的工具调用上触发，该调用恰好
        # 被调度到相同的回收工作线程 tid。
        # `getattr` 回退覆盖通过 object.__new__ 构建 AIAgent 并跳过 __init__ 的测试存根。
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(False, _wtid)
                except Exception:
                    pass
        # 硬中断取代任何待处理的 /steer — 该引导是为
        # 代理的下一个工具调用迭代准备的，现在不会
        # 再发生。删除它而不是在后续回合中通过延迟注入
        # 惊讶用户。
        _steer_lock = getattr(self, "_pending_steer_lock", None)
        if _steer_lock is not None:
            with _steer_lock:
                self._pending_steer = None

    def steer(self, text: str) -> bool:
        """
        在不中断的情况下将用户消息注入到下一个工具结果中。

        与 interrupt() 不同，这不会停止当前的工具调用。
        文本被存储，代理循环在当前工具批次完成后将其
        附加到最后一个工具结果的内容中。模型
        在其下一次迭代中将引导视为工具输出的一部分。

        线程安全：可从网关/CLI/TUI 线程调用。在排放点之前的
        多次调用用换行符连接。

        参数：
            text: 要注入的用户文本。空字符串被忽略。

        返回：
            如果引导被接受则为 True，如果文本为空则为 False。
        """
        if not text or not text.strip():
            return False
        cleaned = text.strip()
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            # 通过 object.__new__ 构建 AIAgent 的测试存根跳过 __init__。
            # 回退到直接属性设置；在这些存根中不期望有并发调用者。
            existing = getattr(self, "_pending_steer", None)
            self._pending_steer = (existing + "\n" + cleaned) if existing else cleaned
            return True
        with _lock:
            if self._pending_steer:
                self._pending_steer = self._pending_steer + "\n" + cleaned
            else:
                self._pending_steer = cleaned
        return True

    def _drain_pending_steer(self) -> Optional[str]:
        """返回待处理的引导文本（如果有）并清除插槽。

        在附加工具结果后从代理执行线程调用是安全的。
        当没有待处理的引导时返回 None。
        """
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            text = getattr(self, "_pending_steer", None)
            self._pending_steer = None
            return text
        with _lock:
            text = self._pending_steer
            self._pending_steer = None
        return text

    def _apply_pending_steer_to_tool_results(self, messages: list, num_tool_msgs: int) -> None:
        """将任何待处理的 /steer 文本附加到此回合中最后一个工具结果。

        在工具调用批次结束时、下一次 API 调用之前调用。
        引导被附加到最后一个 ``role:"tool"`` 消息的内容中，
        带有清晰的标记，以便模型理解它来自用户
        而不是来自工具本身。角色交替被保留 —
        没有插入新内容，我们只修改现有内容。

        参数：
            messages: 运行中的消息列表。
            num_tool_msgs: 此批次中附加的工具结果数量；
                用于安全定位尾部切片。
        """
        if num_tool_msgs <= 0 or not messages:
            return
        steer_text = self._drain_pending_steer()
        if not steer_text:
            return
        # 在最近的尾部中查找最后一个工具角色消息。
        # 跳过非工具消息可以防止未来代码在边界处
        # 附加其他内容。
        target_idx = None
        for j in range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1):
            msg = messages[j]
            if isinstance(msg, dict) and msg.get("role") == "tool":
                target_idx = j
                break
        if target_idx is None:
            # 此批次中没有工具结果（例如全部被中断跳过）；
            # 将引导放回，以便调用者的回退路径可以
            # 将其作为正常的下一回合用户消息传递。
            _lock = getattr(self, "_pending_steer_lock", None)
            if _lock is not None:
                with _lock:
                    if self._pending_steer:
                        self._pending_steer = self._pending_steer + "\n" + steer_text
                    else:
                        self._pending_steer = steer_text
            else:
                existing = getattr(self, "_pending_steer", None)
                self._pending_steer = (existing + "\n" + steer_text) if existing else steer_text
            return
        marker = f"\n\n[USER STEER (injected mid-run, not tool output): {steer_text}]"
        existing_content = messages[target_idx].get("content", "")
        if not isinstance(existing_content, str):
            # Anthropic 多模态内容块 — 保留它们并在末尾
            # 附加文本块。
            try:
                blocks = list(existing_content) if existing_content else []
                blocks.append({"type": "text", "text": marker.lstrip()})
                messages[target_idx]["content"] = blocks
            except Exception:
                # 如果内容形状意外，则回退到字符串替换。
                messages[target_idx]["content"] = f"{existing_content}{marker}"
        else:
            messages[target_idx]["content"] = existing_content + marker
        logger.info(
            "Delivered /steer to agent after tool batch (%d chars): %s",
            len(steer_text),
            steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
        )

    def _touch_activity(self, desc: str) -> None:
        """更新最后活动时间戳和描述（线程安全）。"""
        self._last_activity_ts = time.time()
        self._last_activity_desc = desc

    def _capture_rate_limits(self, http_response: Any) -> None:
        """从 HTTP 响应中解析 x-ratelimit-* 头部并缓存状态。

        在每次流式 API 调用后调用。httpx Response 对象
        在 OpenAI SDK Stream 上通过 ``stream.response`` 可用。
        """
        if http_response is None:
            return
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        try:
            from agent.rate_limit_tracker import parse_rate_limit_headers
            state = parse_rate_limit_headers(headers, provider=self.provider)
            if state is not None:
                self._rate_limit_state = state
        except Exception:
            pass  # 永远不要让头部解析破坏代理循环

    def get_rate_limit_state(self):
        """返回最后捕获的 RateLimitState，或 None。"""
        return self._rate_limit_state

    def get_activity_summary(self) -> dict:
        """返回代理当前活动的快照以进行诊断。

        由网关超时处理程序调用，以报告代理在被杀死时
        正在做什么，以及由定期的"仍在工作"通知调用。
        """
        elapsed = time.time() - self._last_activity_ts
        return {
            "last_activity_ts": self._last_activity_ts,
            "last_activity_desc": self._last_activity_desc,
            "seconds_since_activity": round(elapsed, 1),
            "current_tool": self._current_tool,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "budget_used": self.iteration_budget.used,
            "budget_max": self.iteration_budget.max_total,
        }

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """关闭内存提供商和上下文引擎 — 在实际会话边界调用。

        这会在内存管理器上调用 on_session_end() 然后 shutdown_all()，
        并在上下文引擎上调用 on_session_end()。
        不是每回合调用 — 仅在 CLI 退出、/reset、网关
        会话过期等时调用。
        """
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
            try:
                self._memory_manager.shutdown_all()
            except Exception:
                pass
        # 通知上下文引擎会话结束（刷新 DAG、关闭数据库等）
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_end(
                    self.session_id or "",
                    messages or [],
                )
            except Exception:
                pass
    
    def commit_memory_session(self, messages: list = None) -> None:
        """触发会话结束提取而不拆除提供商。
        当 session_id 轮换时调用（例如 /new、上下文压缩）；
        提供商保持其状态并在旧的 session_id 下继续运行 —
        它们现在只是刷新待处理的提取。"""
        if not self._memory_manager:
            return
        try:
            self._memory_manager.on_session_end(messages or [])
        except Exception:
            pass

    def release_clients(self) -> None:
        """释放 LLM 客户端资源而不拆除会话工具状态。

        当网关出于内存管理原因（LRU 上限或空闲 TTL）
        从 _agent_cache 驱逐此代理时使用 — 会话可能
        随时通过重新构建的 AIAgent 恢复，该 AIAgent 重用
        相同的 task_id / session_id，因此我们绝不能杀死：
          - task_id 的 process_registry 条目（用户的后台 shell）
          - task_id 的终端沙箱（cwd、env、shell 状态）
          - task_id 的浏览器守护进程（打开的标签页、cookies）
          - 内存提供商（有自己的生命周期；保持运行）

        我们确实关闭：
          - OpenAI/httpx 客户端池（大量持有的内存 + 套接字；
            重新构建的代理无论如何都会获得新客户端）
          - 活动的子子代理（每回合的人工制品；可以安全丢弃）

        可以安全多次调用。不同于 close() — 这是
        实际会话边界（/new、/reset、会话过期）的
        硬拆除。
        """
        # 关闭活动子代理（每回合；无跨回合持久性）。
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.release_clients()
                except Exception:
                    # 对子代理回退到完全关闭；它们是每回合的。
                    try:
                        child.close()
                    except Exception:
                        pass
        except Exception:
            pass

        # 关闭 OpenAI/httpx 客户端以立即释放套接字。
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="cache_evict", shared=True)
                self.client = None
        except Exception:
            pass

    def close(self) -> None:
        """释放此代理实例持有的所有资源。

        清理否则会成为孤儿的子进程资源：
        - ProcessRegistry 中跟踪的后台进程
        - 终端沙箱环境
        - 浏览器守护进程会话
        - 活动子代理（子代理委托）
        - OpenAI/httpx 客户端连接

        可以安全多次调用（幂等）。每个清理步骤
        都是独立保护的，因此一个步骤中的失败不会阻止其余步骤。
        """
        task_id = getattr(self, "session_id", None) or ""

        # 1. 杀死此任务的后台进程
        try:
            from tools.process_registry import process_registry
            process_registry.kill_all(task_id=task_id)
        except Exception:
            pass

        # 2. 清理终端沙箱环境
        try:
            from tools.terminal_tool import cleanup_vm
            cleanup_vm(task_id)
        except Exception:
            pass

        # 3. 清理浏览器守护进程会话
        try:
            from tools.browser_tool import cleanup_browser
            cleanup_browser(task_id)
        except Exception:
            pass

        # 4. 关闭活动子代理
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.close()
                except Exception:
                    pass
        except Exception:
            pass

        # 5. 关闭 OpenAI/httpx 客户端
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="agent_close", shared=True)
                self.client = None
        except Exception:
            pass

    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """
        从对话历史中恢复 todo 状态。
        
        网关为每条消息创建一个新的 AIAgent，因此内存中的
        TodoStore 是空的。我们扫描历史以查找最近的 todo
        工具响应并重放它以重建状态。
        """
        # 向后遍历历史以查找最近的 todo 工具响应
        last_todo_response = None
        for msg in reversed(history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # 快速检查：todo 响应包含 "todos" 键
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        
        if last_todo_response:
            # 将项目重放到存储中（替换模式）
            self._todo_store.write(last_todo_response, merge=False)
            if not self.quiet_mode:
                self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)
    
    @property
    def is_interrupted(self) -> bool:
        """检查是否已请求中断。"""
        return self._interrupt_requested










    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        从所有层组装完整的系统提示。
        
        每会话调用一次（缓存在 self._cached_system_prompt 上）并且仅在
        上下文压缩事件后重建。这确保系统提示
        在会话中的所有回合中保持稳定，最大化前缀缓存命中。
        """
        # 层（按顺序）：
        #   1. 代理身份 — SOUL.md（如果可用），否则为 DEFAULT_AGENT_IDENTITY
        #   2. 用户/网关系统提示（如果提供）
        #   3. 持久内存（冻结快照）
        #   4. 技能指导（如果加载了技能工具）
        #   5. 上下文文件（AGENTS.md、.cursorrules — 当用作身份时此处排除 SOUL.md）
        #   6. 当前日期和时间（在构建时冻结）
        #   7. 平台特定的格式提示

        # 尝试将 SOUL.md 作为主要身份（除非跳过上下文文件）
        _soul_loaded = False
        if not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            # 回退到硬编码身份
            prompt_parts = [DEFAULT_AGENT_IDENTITY]

        # 工具感知的行为指导：仅在加载工具时注入
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
        if nous_subscription_prompt:
            prompt_parts.append(nous_subscription_prompt)
        # 工具使用强制：告诉模型实际调用工具而不是
        # 描述预期操作。由 config.yaml 控制
        # agent.tool_use_enforcement:
        #   "auto"（默认）— 匹配 TOOL_USE_ENFORCEMENT_MODELS
        #   true  — 始终注入（所有模型）
        #   false — 永不注入
        #   list  — 要匹配的自定义模型名称子字符串
        if self.valid_tool_names:
            _enforce = self._tool_use_enforcement
            _inject = False
            if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in ("true", "always", "yes", "on")):
                _inject = True
            elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in ("false", "never", "no", "off")):
                _inject = False
            elif isinstance(_enforce, list):
                model_lower = (self.model or "").lower()
                _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
            else:
                # "auto" or any unrecognised value — use hardcoded defaults
                model_lower = (self.model or "").lower()
                _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
            if _inject:
                prompt_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                _model_lower = (self.model or "").lower()
                # Google 模型操作指导（简洁性、绝对
                # 路径、并行工具调用、编辑前验证等）
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    prompt_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                # OpenAI GPT/Codex 执行纪律（工具持久性、
                # 先决条件检查、验证、反幻觉）。
                if "gpt" in _model_lower or "codex" in _model_lower:
                    prompt_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        # 以便它可以向用户引用它们而不是重新发明答案。

        # 注意：ephemeral_system_prompt 不包含在这里。它仅在
        # API 调用时注入，以便它保持在缓存/存储的系统提示之外。
        if system_message is not None:
            prompt_parts.append(system_message)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # USER.md 在启用时始终包含。
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        # 外部内存提供商系统提示块（添加到内置）
        if self._memory_manager:
            try:
                _ext_mem_block = self._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    prompt_parts.append(_ext_mem_block)
            except Exception:
                pass

        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {
                toolset
                for toolset in (
                    get_toolset_for_tool(tool_name) for tool_name in self.valid_tool_names
                )
                if toolset
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        if not self.skip_context_files:
            # 在设置时使用 TERMINAL_CWD 进行上下文文件发现（网关
            # 模式）。网关进程从 hermes-agent 安装目录运行，
            # 因此 os.getcwd() 会获取仓库的 AGENTS.md 和
            # 其他开发文件 — 无益地将令牌使用量增加约 10k。
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        if self.model:
            timestamp_line += f"\nModel: {self.model}"
        if self.provider:
            timestamp_line += f"\nProvider: {self.provider}"
        prompt_parts.append(timestamp_line)

        # Alibaba Coding Plan API 始终返回 "glm-4.7" 作为模型名称，无论
        # 请求的模型是什么。在系统提示中注入显式模型身份，
        # 以便代理可以正确报告它是什么模型（API 错误的变通方法）。
        if self.provider == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            prompt_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        # 环境提示（WSL、Termux 等）— 告诉代理有关
        # 执行环境的信息，以便它可以转换路径和调整行为。
        _env_hints = build_environment_hints()
        if _env_hints:
            prompt_parts.append(_env_hints)

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(p.strip() for p in prompt_parts if p.strip())

    # =========================================================================
    # 调用前/后护栏（灵感来自 PR #1321 — @alireza78a）
    # =========================================================================

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """从 tool_call 条目（字典或对象）中提取调用 ID。"""
        if isinstance(tc, dict):
            return tc.get("id", "") or ""
        return getattr(tc, "id", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

    @staticmethod
    def _sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """在每次 LLM 调用之前修复孤立的 tool_call / tool_result 对。

        无条件运行 — 不取决于上下文压缩器是否存在 —
        因此会话加载或手动消息操作的孤立项总是被捕获。
        """
        # --- 角色允许列表：删除 API 不接受的角色消息 ---
        filtered = []
        for msg in messages:
            role = msg.get("role")
            if role not in AIAgent._VALID_API_ROLES:
                logger.debug(
                    "Pre-call sanitizer: dropping message with invalid role %r",
                    role,
                )
                continue
            filtered.append(msg)
        messages = filtered

        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = AIAgent._get_tool_call_id_static(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. 删除没有匹配助手调用的工具结果
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            logger.debug(
                "Pre-call sanitizer: removed %d orphaned tool result(s)",
                len(orphaned_results),
            )

        # 2. 为结果被删除的调用注入存根结果
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = AIAgent._get_tool_call_id_static(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result unavailable — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            logger.debug(
                "Pre-call sanitizer: added %d stub tool result(s)",
                len(missing_results),
            )
        return messages

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """将过多的 delegate_task 调用截断到 max_concurrent_children。

        delegate_tool 限制单个调用内的任务列表，但
        模型可以在一个回合中发出多个独立的 delegate_task tool_calls。
        这会截断多余的部分，保留所有非委托调用。

        如果不需要截断，则返回原始列表。
        """
        from tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= max_children:
            return tool_calls
        kept_delegates = 0
        truncated = []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates < max_children:
                    truncated.append(tc)
                    kept_delegates += 1
            else:
                truncated.append(tc)
        logger.warning(
            "Truncated %d excess delegate_task call(s) to enforce "
            "max_concurrent_children=%d limit",
            delegate_count - max_children, max_children,
        )
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """删除单个回合内的重复（tool_name, arguments）对。

        只保留每个唯一对的第一次出现。
        如果未找到重复项，则返回原始列表。
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls

    def _repair_tool_call(self, tool_name: str) -> str | None:
        """在中止之前尝试修复不匹配的工具名称。

        1. 尝试小写
        2. 尝试标准化（小写 + 连字符/空格 -> 下划线）
        3. 尝试模糊匹配（difflib，cutoff=0.7）

        如果在 valid_tool_names 中找到，则返回修复的名称，否则返回 None。
        """
        from difflib import get_close_matches

        # 1. 小写
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered

        # 2. 标准化
        normalized = lowered.replace("-", "_").replace(" ", "_")
        if normalized in self.valid_tool_names:
            return normalized

        # 3. 模糊匹配
        matches = get_close_matches(lowered, self.valid_tool_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None

    def _invalidate_system_prompt(self):
        """
        使缓存的系统提示失效，强制在下一回合重建。
        
        在上下文压缩事件后调用。同时从磁盘重新加载内存，
        以便重建的提示捕获此会话的任何写入。
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()

    def _responses_tools(self, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
        """将 chat-completions 工具架构转换为 Responses 函数工具架构。"""
        source_tools = tools if tools is not None else self.tools
        if not source_tools:
            return None

        converted: List[Dict[str, Any]] = []
        for item in source_tools:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append({
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "strict": False,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted or None

    @staticmethod
    def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
        """从工具调用内容生成确定性 call_id。

        当 API 不提供 call_id 时用作回退。
        确定性 ID 防止缓存失效 — 随机 UUID 会
        使每次 API 调用的前缀唯一，破坏 OpenAI 的提示缓存。
        """
        import hashlib
        seed = f"{fn_name}:{arguments}:{index}"
        digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"call_{digest}"

    @staticmethod
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """将存储的工具 id 分割为 (call_id, response_item_id)。"""
        if not isinstance(raw_id, str):
            return None, None
        value = raw_id.strip()
        if not value:
            return None, None
        if "|" in value:
            call_id, response_item_id = value.split("|", 1)
            call_id = call_id.strip() or None
            response_item_id = response_item_id.strip() or None
            return call_id, response_item_id
        if value.startswith("fc_"):
            return None, value
        return value, None

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """构建有效的 Responses `function_call.id`（必须以 `fc_` 开头）。"""
        if isinstance(response_item_id, str):
            candidate = response_item_id.strip()
            if candidate.startswith("fc_"):
                return candidate

        source = (call_id or "").strip()
        if source.startswith("fc_"):
            return source
        if source.startswith("call_") and len(source) > len("call_"):
            return f"fc_{source[len('call_'):]}"

        sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
        if sanitized.startswith("fc_"):
            return sanitized
        if sanitized.startswith("call_") and len(sanitized) > len("call_"):
            return f"fc_{sanitized[len('call_'):]}"
        if sanitized:
            return f"fc_{sanitized[:48]}"

        seed = source or str(response_item_id or "") or uuid.uuid4().hex
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
        return f"fc_{digest}"

    def _chat_messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将内部聊天样式消息转换为 Responses 输入项。"""
        items: List[Dict[str, Any]] = []
        seen_item_ids: set = set()

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                continue

            if role in {"user", "assistant"}:
                content = msg.get("content", "")
                content_text = str(content) if content is not None else ""

                if role == "assistant":
                    # 重播来自先前回合的加密推理项，
                    # 以便 API 可以保持连贯的推理链。
                    codex_reasoning = msg.get("codex_reasoning_items")
                    has_codex_reasoning = False
                    if isinstance(codex_reasoning, list):
                        for ri in codex_reasoning:
                            if isinstance(ri, dict) and ri.get("encrypted_content"):
                                item_id = ri.get("id")
                                if item_id and item_id in seen_item_ids:
                                    continue
                                # 剥离 "id" 字段 — 使用 store=False 时，
                                # Responses API 无法通过 ID 查找项并
                                # 返回 404。encrypted_content blob 是
                                # 自包含的，用于推理链连续性。
                                replay_item = {k: v for k, v in ri.items() if k != "id"}
                                items.append(replay_item)
                                if item_id:
                                    seen_item_ids.add(item_id)
                                has_codex_reasoning = True

                    if content_text.strip():
                        items.append({"role": "assistant", "content": content_text})
                    elif has_codex_reasoning:
                        # Responses API 在每个推理项之后需要一个后续项
                        # （否则：missing_following_item 错误）。
                        # 当助手仅产生推理而没有可见内容时，
                        # 发出空助手消息作为所需的后续项。
                        items.append({"role": "assistant", "content": ""})

                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get("function", {})
                            fn_name = fn.get("name")
                            if not isinstance(fn_name, str) or not fn_name.strip():
                                continue

                            embedded_call_id, embedded_response_item_id = self._split_responses_tool_id(
                                tc.get("id")
                            )
                            call_id = tc.get("call_id")
                            if not isinstance(call_id, str) or not call_id.strip():
                                call_id = embedded_call_id
                            if not isinstance(call_id, str) or not call_id.strip():
                                if (
                                    isinstance(embedded_response_item_id, str)
                                    and embedded_response_item_id.startswith("fc_")
                                    and len(embedded_response_item_id) > len("fc_")
                                ):
                                    call_id = f"call_{embedded_response_item_id[len('fc_'):]}"
                                else:
                                    _raw_args = str(fn.get("arguments", "{}"))
                                    call_id = self._deterministic_call_id(fn_name, _raw_args, len(items))
                            call_id = call_id.strip()

                            arguments = fn.get("arguments", "{}")
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments, ensure_ascii=False)
                            elif not isinstance(arguments, str):
                                arguments = str(arguments)
                            arguments = arguments.strip() or "{}"

                            items.append({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": fn_name,
                                "arguments": arguments,
                            })
                    continue

                items.append({"role": role, "content": content_text})
                continue

            if role == "tool":
                raw_tool_call_id = msg.get("tool_call_id")
                call_id, _ = self._split_responses_tool_id(raw_tool_call_id)
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                        call_id = raw_tool_call_id.strip()
                if not isinstance(call_id, str) or not call_id.strip():
                    continue
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(msg.get("content", "") or ""),
                })

        return items

    def _preflight_codex_input_items(self, raw_items: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            raise ValueError("Codex Responses input must be a list of input items.")

        normalized: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Codex Responses input[{idx}] must be an object.")

            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")

                arguments = item.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                arguments = arguments.strip() or "{}"

                normalized.append(
                    {
                        "type": "function_call",
                        "call_id": call_id.strip(),
                        "name": name.strip(),
                        "arguments": arguments,
                    }
                )
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
                output = item.get("output", "")
                if output is None:
                    output = ""
                if not isinstance(output, str):
                    output = str(output)

                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id.strip(),
                        "output": output,
                    }
                )
                continue

            if item_type == "reasoning":
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    item_id = item.get("id")
                    if isinstance(item_id, str) and item_id:
                        if item_id in seen_ids:
                            continue
                        seen_ids.add(item_id)
                    reasoning_item = {"type": "reasoning", "encrypted_content": encrypted}
                    # 不要在传出项中包含 "id" — 使用
                    # store=False（我们的默认值）时，API 尝试
                    # 服务器端解析 id 并返回 404。id 仍然在
                    # 上面用于通过 seen_ids 进行本地去重。
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        reasoning_item["summary"] = summary
                    else:
                        reasoning_item["summary"] = []
                    normalized.append(reasoning_item)
                continue

            role = item.get("role")
            if role in {"user", "assistant"}:
                content = item.get("content", "")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)

                normalized.append({"role": role, "content": content})
                continue

            raise ValueError(
                f"Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r})."
            )

        return normalized

    def _preflight_codex_api_kwargs(
        self,
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(api_kwargs, dict):
            raise ValueError("Codex Responses request must be a dict.")

        required = {"model", "instructions", "input"}
        missing = [key for key in required if key not in api_kwargs]
        if missing:
            raise ValueError(f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}.")

        model = api_kwargs.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Codex Responses request 'model' must be a non-empty string.")
        model = model.strip()

        instructions = api_kwargs.get("instructions")
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            instructions = str(instructions)
        instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY

        normalized_input = self._preflight_codex_input_items(api_kwargs.get("input"))

        tools = api_kwargs.get("tools")
        normalized_tools = None
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError("Codex Responses request 'tools' must be a list when provided.")
            normalized_tools = []
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] must be an object.")
                if tool.get("type") != "function":
                    raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")

                name = tool.get("name")
                parameters = tool.get("parameters")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses tools[{idx}] is missing a valid name.")
                if not isinstance(parameters, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] is missing valid parameters.")

                description = tool.get("description", "")
                if description is None:
                    description = ""
                if not isinstance(description, str):
                    description = str(description)

                strict = tool.get("strict", False)
                if not isinstance(strict, bool):
                    strict = bool(strict)

                normalized_tools.append(
                    {
                        "type": "function",
                        "name": name.strip(),
                        "description": description,
                        "strict": strict,
                        "parameters": parameters,
                    }
                )

        store = api_kwargs.get("store", False)
        if store is not False:
            raise ValueError("Codex Responses contract requires 'store' to be false.")

        allowed_keys = {
            "model", "instructions", "input", "tools", "store",
            "reasoning", "include", "max_output_tokens", "temperature",
            "tool_choice", "parallel_tool_calls", "prompt_cache_key", "service_tier",
            "extra_headers",
        }
        normalized: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": normalized_input,
            "store": False,
        }
        if normalized_tools is not None:
            normalized["tools"] = normalized_tools

        # 传递推理配置
        reasoning = api_kwargs.get("reasoning")
        if isinstance(reasoning, dict):
            normalized["reasoning"] = reasoning
        include = api_kwargs.get("include")
        if isinstance(include, list):
            normalized["include"] = include
        service_tier = api_kwargs.get("service_tier")
        if isinstance(service_tier, str) and service_tier.strip():
            normalized["service_tier"] = service_tier.strip()

        # 传递 max_output_tokens 和 temperature
        max_output_tokens = api_kwargs.get("max_output_tokens")
        if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
            normalized["max_output_tokens"] = int(max_output_tokens)
        temperature = api_kwargs.get("temperature")
        if isinstance(temperature, (int, float)):
            normalized["temperature"] = float(temperature)

        # 传递 tool_choice、parallel_tool_calls、prompt_cache_key
        for passthrough_key in ("tool_choice", "parallel_tool_calls", "prompt_cache_key"):
            val = api_kwargs.get(passthrough_key)
            if val is not None:
                normalized[passthrough_key] = val

        extra_headers = api_kwargs.get("extra_headers")
        if extra_headers is not None:
            if not isinstance(extra_headers, dict):
                raise ValueError("Codex Responses request 'extra_headers' must be an object.")
            normalized_headers: Dict[str, str] = {}
            for key, value in extra_headers.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Codex Responses request 'extra_headers' keys must be non-empty strings.")
                if value is None:
                    continue
                normalized_headers[key.strip()] = str(value)
            if normalized_headers:
                normalized["extra_headers"] = normalized_headers

        if allow_stream:
            stream = api_kwargs.get("stream")
            if stream is not None and stream is not True:
                raise ValueError("Codex Responses 'stream' must be true when set.")
            if stream is True:
                normalized["stream"] = True
            allowed_keys.add("stream")
        elif "stream" in api_kwargs:
            raise ValueError("Codex Responses stream flag is only allowed in fallback streaming requests.")

        unexpected = sorted(key for key in api_kwargs if key not in allowed_keys)
        if unexpected:
            raise ValueError(
                f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}."
            )

        return normalized

    def _extract_responses_message_text(self, item: Any) -> str:
        """从 Responses 消息输出项中提取助手文本。"""
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            return ""

        chunks: List[str] = []
        for part in content:
            ptype = getattr(part, "type", None)
            if ptype not in {"output_text", "text"}:
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks).strip()

    def _extract_responses_reasoning_text(self, item: Any) -> str:
        """从 Responses 推理项中提取紧凑的推理文本。"""
        summary = getattr(item, "summary", None)
        if isinstance(summary, list):
            chunks: List[str] = []
            for part in summary:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            return text.strip()
        return ""

    def _normalize_codex_response(self, response: Any) -> tuple[Any, str]:
        """将 Responses API 对象标准化为类似 assistant_message 的对象。"""
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            # 当答案完全通过流事件传递时，Codex 后端可以返回空输出。
            # 在引发之前检查 output_text 作为最后的回退。
            out_text = getattr(response, "output_text", None)
            if isinstance(out_text, str) and out_text.strip():
                logger.debug(
                    "Codex response has empty output but output_text is present (%d chars); "
                    "synthesizing output item.", len(out_text.strip()),
                )
                output = [SimpleNamespace(
                    type="message", role="assistant", status="completed",
                    content=[SimpleNamespace(type="output_text", text=out_text.strip())],
                )]
                response.output = output
            else:
                raise RuntimeError("Responses API returned no output items")

        response_status = getattr(response, "status", None)
        if isinstance(response_status, str):
            response_status = response_status.strip().lower()
        else:
            response_status = None

        if response_status in {"failed", "cancelled"}:
            error_obj = getattr(response, "error", None)
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = str(error_obj) if error_obj else f"Responses API returned status '{response_status}'"
            raise RuntimeError(error_msg)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_items_raw: List[Dict[str, Any]] = []
        tool_calls: List[Any] = []
        has_incomplete_items = response_status in {"queued", "in_progress", "incomplete"}
        saw_commentary_phase = False
        saw_final_answer_phase = False

        for item in output:
            item_type = getattr(item, "type", None)
            item_status = getattr(item, "status", None)
            if isinstance(item_status, str):
                item_status = item_status.strip().lower()
            else:
                item_status = None

            if item_status in {"queued", "in_progress", "incomplete"}:
                has_incomplete_items = True

            if item_type == "message":
                item_phase = getattr(item, "phase", None)
                if isinstance(item_phase, str):
                    normalized_phase = item_phase.strip().lower()
                    if normalized_phase in {"commentary", "analysis"}:
                        saw_commentary_phase = True
                    elif normalized_phase in {"final_answer", "final"}:
                        saw_final_answer_phase = True
                message_text = self._extract_responses_message_text(item)
                if message_text:
                    content_parts.append(message_text)
            elif item_type == "reasoning":
                reasoning_text = self._extract_responses_reasoning_text(item)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                # 捕获完整的推理项以实现多回合连续性。
                # encrypted_content 是 API 在后续回合中需要返回的
                # 不透明 blob，以保持连贯的推理链。
                encrypted = getattr(item, "encrypted_content", None)
                if isinstance(encrypted, str) and encrypted:
                    raw_item = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = getattr(item, "id", None)
                    if isinstance(item_id, str) and item_id:
                        raw_item["id"] = item_id
                    # 捕获摘要 — API 在重播推理项时需要
                    summary = getattr(item, "summary", None)
                    if isinstance(summary, list):
                        raw_summary = []
                        for part in summary:
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                raw_summary.append({"type": "summary_text", "text": text})
                        raw_item["summary"] = raw_summary
                    reasoning_items_raw.append(raw_item)
            elif item_type == "function_call":
                if item_status in {"queued", "in_progress", "incomplete"}:
                    continue
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))
            elif item_type == "custom_tool_call":
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "input", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))

        final_text = "\n".join([p for p in content_parts if p]).strip()
        if not final_text and hasattr(response, "output_text"):
            out_text = getattr(response, "output_text", "")
            if isinstance(out_text, str):
                final_text = out_text.strip()

        assistant_message = SimpleNamespace(
            content=final_text,
            tool_calls=tool_calls,
            reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=reasoning_items_raw or None,
        )

        if tool_calls:
            finish_reason = "tool_calls"
        elif has_incomplete_items or (saw_commentary_phase and not saw_final_answer_phase):
            finish_reason = "incomplete"
        elif reasoning_items_raw and not final_text:
            # 响应仅包含推理（加密思考状态），没有
            # 可见内容或工具调用。模型仍在思考中，
            # 需要另一个回合来产生实际答案。
            # 将其标记为 "stop" 会将其发送到空内容重试循环，
            # 该循环消耗 3 次重试然后失败 — 将其视为不完整，
            # 以便 Codex 延续路径正确处理它。
            finish_reason = "incomplete"
        else:
            finish_reason = "stop"
        return assistant_message, finish_reason

    def _thread_identity(self) -> str:
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"

    def _client_log_context(self) -> str:
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """检查 OpenAI 客户端是否关闭。

        处理 is_closed 的属性和方法形式：
        - httpx.Client.is_closed 是布尔属性
        - openai.OpenAI.is_closed 是返回布尔值的方法

        先前的错误：getattr(client, "is_closed", False) 返回了绑定方法，
        该方法始终为真，导致每次调用时不必要地重新创建客户端。
        """
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            # 处理方法（openai SDK）与属性（httpx）
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
        # 将 client_kwargs 视为只读。调用者传入 self._client_kwargs（或其浅表
        # 副本）；任何就地变异会泄漏回存储的字典并在
        # 后续请求中重用。#10933 通过注入一个 httpx.Client
        # 传输来解决这个问题，该传输在第一次请求后被拆除，因此
        # 下一个请求包装了一个关闭的传输并在每次重试时
        # 引发"Cannot send a request, as the client has been closed"。
        # 撤销解决了该特定路径；此副本锁定了合同，
        # 因此未来的传输/keepalive 工作不能重新引入
        # 同类错误。
        client_kwargs = dict(client_kwargs)
        _validate_proxy_env_urls()
        _validate_base_url(client_kwargs.get("base_url"))
        if self.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(**client_kwargs)
            logger.info(
                "Copilot ACP client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        if self.provider == "google-gemini-cli" or str(client_kwargs.get("base_url", "")).startswith("cloudcode-pa://"):
            from agent.gemini_cloudcode_adapter import GeminiCloudCodeClient

            # 剥离 Gemini 客户端不接受的 OpenAI 特定 kwargs
            safe_kwargs = {
                k: v for k, v in client_kwargs.items()
                if k in {"api_key", "base_url", "default_headers", "project_id", "timeout"}
            }
            client = GeminiCloudCodeClient(**safe_kwargs)
            logger.info(
                "Gemini Cloud Code Assist client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        # 注入 TCP keepalives 以便内核检测死连接的提供商连接，
        # 而不是让它们在 CLOSE-WAIT 中静默挂起（#10324）。
        # 没有这个，在流中途断开的对等方会将套接字留在一种状态，
        # 其中 epoll_wait 永远不会触发，``httpx`` 读取超时可能不会触发，
        # 并且代理会挂起直到手动杀死。空闲 30 秒后探测，
        # 每 10 秒重试一次，3 次后放弃 → 在约 60 秒内检测到死对等方。
        #
        # 针对 #10933 的安全性：上面的 ``client_kwargs = dict(client_kwargs)``
        # 意味着此注入仅落在本地每次调用副本中，
        # 永远不会回到 ``self._client_kwargs``。因此每次 ``_create_openai_client``
        # 调用都会获得其自己的新鲜 ``httpx.Client``，其
        # 生命周期与传递给它的 OpenAI 客户端绑定。当
        # OpenAI 客户端关闭（重建、拆除、凭证轮换）时，
        # 配对的 ``httpx.Client`` 随之关闭，下一次调用
        # 构建一个新的 — 不能重用陈旧的关闭传输。
        # ``tests/run_agent/test_create_openai_client_reuse.py`` 和
        # ``tests/run_agent/test_sequential_chats_live.py`` 中的测试固定了此不变量。
        if "http_client" not in client_kwargs:
            try:
                import httpx as _httpx
                import socket as _socket
                _sock_opts = [(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)]
                if hasattr(_socket, "TCP_KEEPIDLE"):
                    # Linux
                    _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPIDLE, 30))
                    _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL, 10))
                    _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT, 3))
                elif hasattr(_socket, "TCP_KEEPALIVE"):
                    # macOS (uses TCP_KEEPALIVE instead of TCP_KEEPIDLE)
                    _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPALIVE, 30))
                client_kwargs["http_client"] = _httpx.Client(
                    transport=_httpx.HTTPTransport(socket_options=_sock_opts),
                )
            except Exception:
                pass  # 如果套接字选项失败，则回退到默认传输
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """强制关闭底层 TCP 套接字以防止 CLOSE-WAIT 累积。

        当提供商在流中途断开连接时，httpx 的 ``client.close()``
        执行优雅关闭，将套接字留在 CLOSE-WAIT 中，直到
        OS 超时（通常几分钟）。此方法遍历 httpx 传输
        池并发出 ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()`` 以
        强制立即 TCP RST，释放文件描述符。

        返回强制关闭的套接字数量。
        """
        import socket as _socket

        closed = 0
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return 0
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return 0
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return 0
            # httpx 使用 httpcore 连接池；连接位于
            # _connections（列表）或 _pool（列表）中，取决于版本。
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            for conn in list(connections):
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                closed += 1
        except Exception as exc:
            logger.debug("Force-close TCP sockets sweep error: %s", exc)
        return closed

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None:
            return
        # 首先强制关闭 TCP 套接字以防止 CLOSE-WAIT 累积，
        # 然后执行优雅的 SDK 级别关闭。
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
                reason,
                shared,
                force_closed,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._create_openai_client(self._client_kwargs, reason=reason, shared=True)
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning(
            "Detected closed shared OpenAI client; recreating before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self._openai_client_lock():
            return self.client

    def _cleanup_dead_connections(self) -> bool:
        """检测并清理主客户端上的死 TCP 连接。

        检查 httpx 连接池中处于不健康状态的套接字
        （CLOSE-WAIT、错误）。如果发现任何，则强制关闭所有套接字
        并从头开始重建主客户端。

        如果发现并清理了死连接，则返回 True。
        """
        client = getattr(self, "client", None)
        if client is None:
            return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return False
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return False
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return False
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            dead_count = 0
            for conn in list(connections):
                # 检查空闲但已关闭套接字的连接
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                # 使用非阻塞 recv peek 探测套接字健康状况
                import socket as _socket
                try:
                    sock.setblocking(False)
                    data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                    if data == b"":
                        dead_count += 1
                except BlockingIOError:
                    pass  # 没有可用数据 — 套接字健康
                except OSError:
                    dead_count += 1
                finally:
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
            if dead_count > 0:
                logger.warning(
                    "Found %d dead connection(s) in client pool — rebuilding client",
                    dead_count,
                )
                self._replace_primary_openai_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False

    def _create_request_openai_client(self, *, reason: str) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        self._close_openai_client(client, reason=reason, shared=False)

    def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
        """执行一个流式 Responses API 请求并返回最终响应。"""
        import httpx as _httpx

        active_client = client or self._ensure_primary_openai_client(reason="codex_stream_direct")
        max_stream_retries = 1
        has_tool_calls = False
        first_delta_fired = False
        # 累积流式文本，以便在 get_final_response()
        # 返回空输出时我们可以恢复（例如 chatgpt.com backend-api 发送
        # response.incomplete 而不是 response.completed）。
        self._codex_streamed_text_parts: list = []
        for attempt in range(max_stream_retries + 1):
            collected_output_items: list = []
            try:
                with active_client.responses.stream(**api_kwargs) as stream:
                    for event in stream:
                        self._touch_activity("receiving stream response")
                        if self._interrupt_requested:
                            break
                        event_type = getattr(event, "type", "")
                        # 在文本内容增量上触发回调（在工具调用期间抑制）
                        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
                            delta_text = getattr(event, "delta", "")
                            if delta_text:
                                self._codex_streamed_text_parts.append(delta_text)
                            if delta_text and not has_tool_calls:
                                if not first_delta_fired:
                                    first_delta_fired = True
                                    if on_first_delta:
                                        try:
                                            on_first_delta()
                                        except Exception:
                                            pass
                                self._fire_stream_delta(delta_text)
                        # 跟踪工具调用以抑制文本流式传输
                        elif "function_call" in event_type:
                            has_tool_calls = True
                        # 触发推理回调
                        elif "reasoning" in event_type and "delta" in event_type:
                            reasoning_text = getattr(event, "delta", "")
                            if reasoning_text:
                                self._fire_reasoning_delta(reasoning_text)
                        # 收集已完成的输出项 — 某些后端
                        # （chatgpt.com/backend-api/codex）通过 response.output_item.done
                        # 流式传输有效项，但 SDK 的
                        # get_final_response() 返回空输出列表。
                        elif event_type == "response.output_item.done":
                            done_item = getattr(event, "item", None)
                            if done_item is not None:
                                collected_output_items.append(done_item)
                        # 记录未完成的终端事件以进行诊断
                        elif event_type in ("response.incomplete", "response.failed"):
                            resp_obj = getattr(event, "response", None)
                            status = getattr(resp_obj, "status", None) if resp_obj else None
                            incomplete_details = getattr(resp_obj, "incomplete_details", None) if resp_obj else None
                            logger.warning(
                                "Codex Responses stream received terminal event %s "
                                "(status=%s, incomplete_details=%s, streamed_chars=%d). %s",
                                event_type, status, incomplete_details,
                                sum(len(p) for p in self._codex_streamed_text_parts),
                                self._client_log_context(),
                            )
                    final_response = stream.get_final_response()
                    # 补丁：ChatGPT Codex 后端流式传输有效输出项
                    # 但 get_final_response() 可以返回空输出列表。
                    # 从收集的项回填或从增量合成。
                    _out = getattr(final_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            final_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex stream: backfilled %d output items from stream events",
                                len(collected_output_items),
                            )
                        elif self._codex_streamed_text_parts and not has_tool_calls:
                            assembled = "".join(self._codex_streamed_text_parts)
                            final_response.output = [SimpleNamespace(
                                type="message",
                                role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex stream: synthesized output from %d text deltas (%d chars)",
                                len(self._codex_streamed_text_parts), len(assembled),
                            )
                    return final_response
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed (attempt %s/%s); retrying. %s error=%s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                        exc,
                    )
                    continue
                logger.debug(
                    "Codex Responses stream transport failed; falling back to create(stream=True). %s error=%s",
                    self._client_log_context(),
                    exc,
                )
                return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
            except RuntimeError as exc:
                err_text = str(exc)
                missing_completed = "response.completed" in err_text
                if missing_completed and attempt < max_stream_retries:
                    logger.debug(
                        "Responses stream closed before completion (attempt %s/%s); retrying. %s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                    )
                    continue
                if missing_completed:
                    logger.debug(
                        "Responses stream did not emit response.completed; falling back to create(stream=True). %s",
                        self._client_log_context(),
                    )
                    return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
                raise

    def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
        """Codex 风格 Responses 后端上流完成边缘情况的回退路径。"""
        active_client = client or self._ensure_primary_openai_client(reason="codex_create_stream_fallback")
        fallback_kwargs = dict(api_kwargs)
        fallback_kwargs["stream"] = True
        fallback_kwargs = self._preflight_codex_api_kwargs(fallback_kwargs, allow_stream=True)
        stream_or_response = active_client.responses.create(**fallback_kwargs)

        # 仍然返回具体响应的模拟或提供商的兼容性填充。
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            return stream_or_response

        terminal_response = None
        collected_output_items: list = []
        collected_text_deltas: list = []
        try:
            for event in stream_or_response:
                self._touch_activity("receiving stream response")
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")

                # 收集输出项和文本增量以进行回填
                if event_type == "response.output_item.done":
                    done_item = getattr(event, "item", None)
                    if done_item is None and isinstance(event, dict):
                        done_item = event.get("item")
                    if done_item is not None:
                        collected_output_items.append(done_item)
                elif event_type in ("response.output_text.delta",):
                    delta = getattr(event, "delta", "")
                    if not delta and isinstance(event, dict):
                        delta = event.get("delta", "")
                    if delta:
                        collected_text_deltas.append(delta)

                if event_type not in {"response.completed", "response.incomplete", "response.failed"}:
                    continue

                terminal_response = getattr(event, "response", None)
                if terminal_response is None and isinstance(event, dict):
                    terminal_response = event.get("response")
                if terminal_response is not None:
                    # 从收集的流事件回填空输出
                    _out = getattr(terminal_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            terminal_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex fallback stream: backfilled %d output items",
                                len(collected_output_items),
                            )
                        elif collected_text_deltas:
                            assembled = "".join(collected_text_deltas)
                            terminal_response.output = [SimpleNamespace(
                                type="message", role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex fallback stream: synthesized from %d deltas (%d chars)",
                                len(collected_text_deltas), len(assembled),
                            )
                    return terminal_response
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if terminal_response is not None:
            return terminal_response
        raise RuntimeError("Responses create(stream=True) fallback did not emit a terminal response.")

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider != "openai-codex":
            return False

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="codex_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "chat_completions" or self.provider != "nous":
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous 请求不应继承仅 OpenRouter 的归属头部。
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
            return False
        # 仅为本机 Anthropic 提供商刷新凭证。
        # 其他 anthropic_messages 提供商（MiniMax、Alibaba 等）使用自己的密钥。
        if self.provider != "anthropic":
            return False

        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(new_token, getattr(self, "_anthropic_base_url", None))
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        # 更新 OAuth 标志 — 令牌类型可能已更改（API 密钥 ↔ OAuth）
        from agent.anthropic_adapter import _is_oauth_token
        self._is_anthropic_oauth = _is_oauth_token(new_token)
        return True

    def _apply_client_headers_for_base_url(self, base_url: str) -> None:
        from agent.auxiliary_client import _OR_HEADERS

        normalized = (base_url or "").lower()
        if "openrouter" in normalized:
            self._client_kwargs["default_headers"] = dict(_OR_HEADERS)
        elif "api.githubcopilot.com" in normalized:
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif "api.kimi.com" in normalized:
            self._client_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.30.0"}
        elif "portal.qwen.ai" in normalized:
            self._client_kwargs["default_headers"] = _qwen_portal_headers()
        else:
            self._client_kwargs.pop("default_headers", None)

    def _swap_credential(self, entry) -> None:
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key
            self._anthropic_base_url = runtime_base
            self._anthropic_client = build_anthropic_client(runtime_key, runtime_base)
            self._is_anthropic_oauth = _is_oauth_token(runtime_key)
            self.api_key = runtime_key
            self.base_url = runtime_base
            return

        self.api_key = runtime_key
        self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(self.base_url)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _recover_with_credential_pool(
        self,
        *,
        status_code: Optional[int],
        has_retried_429: bool,
        classified_reason: Optional[FailoverReason] = None,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, bool]:
        """通过池轮换尝试凭证恢复。

        返回 (recovered, has_retried_429)。
        速率限制：首次出现重试相同凭证（设置标志 True）。
                    连续第二次失败轮换到下一个凭证。
        计费耗尽：立即轮换。
        认证失败：在轮换前尝试令牌刷新。

        `classified_reason` 让恢复路径尊重结构化错误
        分类器，而不是仅依赖原始 HTTP 代码。这对于
        在不同状态代码下暴露计费/速率限制/认证条件的提供商很重要，
        例如 Anthropic 为"额外使用量不足"返回 HTTP 400。
        """
        pool = self._credential_pool
        if pool is None:
            return False, has_retried_429

        effective_reason = classified_reason
        if effective_reason is None:
            if status_code == 402:
                effective_reason = FailoverReason.billing
            elif status_code == 429:
                effective_reason = FailoverReason.rate_limit
            elif status_code == 401:
                effective_reason = FailoverReason.auth

        if effective_reason == FailoverReason.billing:
            rotate_status = status_code if status_code is not None else 402
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (billing) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, has_retried_429

        if effective_reason == FailoverReason.rate_limit:
            if not has_retried_429:
                return False, True
            rotate_status = status_code if status_code is not None else 429
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (rate limit) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, True

        if effective_reason == FailoverReason.auth:
            refreshed = pool.try_refresh_current()
            if refreshed is not None:
                logger.info(f"Credential auth failure — refreshed pool entry {getattr(refreshed, 'id', '?')}")
                self._swap_credential(refreshed)
                return True, has_retried_429
            # 刷新失败 — 轮换到下一个凭证而不是放弃。
            # 失败的条目已被 try_refresh_current() 标记为耗尽。
            rotate_status = status_code if status_code is not None else 401
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (auth refresh failed) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False

        return False, has_retried_429

    def _anthropic_messages_create(self, api_kwargs: dict):
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        return self._anthropic_client.messages.create(**api_kwargs)

    def _interruptible_api_call(self, api_kwargs: dict):
        """
        在后台线程中运行 API 调用，以便主对话循环
        可以检测中断而无需等待完整的 HTTP 往返。

        每个工作线程都有自己的 OpenAI 客户端实例。
        中断仅关闭该工作线程本地客户端，因此重试和其他请求
        永远不会继承关闭的传输。

        包括陈旧调用检测器：如果在配置的超时内
        没有响应到达，则终止连接并引发错误，
        以便主重试循环可以尝试使用退避/凭证轮换/
        提供商回退。
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None}

        def _call():
            try:
                if self.api_mode == "codex_responses":
                    request_client_holder["client"] = self._create_request_openai_client(reason="codex_stream_request")
                    result["response"] = self._run_codex_stream(
                        api_kwargs,
                        client=request_client_holder["client"],
                        on_first_delta=getattr(self, "_codex_on_first_delta", None),
                    )
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)
                elif self.api_mode == "bedrock_converse":
                    # Bedrock 直接使用 boto3 — 不需要 OpenAI 客户端。
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        normalize_converse_response,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    raw_response = client.converse(**api_kwargs)
                    result["response"] = normalize_converse_response(raw_response)
                else:
                    request_client_holder["client"] = self._create_request_openai_client(reason="chat_completion_request")
                    result["response"] = request_client_holder["client"].chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="request_complete")

        # ── 陈旧调用超时（镜像流式陈旧检测器） ────────
        # 非流式调用在完整响应准备就绪之前不返回任何内容。
        # 没有这个，挂起的提供商可以阻塞完整的
        # httpx 超时（默认 1800 秒）而没有任何反馈。
        # 陈旧检测器提前终止连接，以便主重试循环
        # 可以应用更丰富的恢复（凭证轮换、提供商回退）。
        _stale_base = float(os.getenv("HERMES_API_CALL_STALE_TIMEOUT", 300.0))
        _base_url = getattr(self, "_base_url", None) or ""
        if _stale_base == 300.0 and _base_url and is_local_endpoint(_base_url):
            _stale_timeout = float("inf")
        else:
            _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
            if _est_tokens > 100_000:
                _stale_timeout = max(_stale_base, 600.0)
            elif _est_tokens > 50_000:
                _stale_timeout = max(_stale_base, 450.0)
            else:
                _stale_timeout = _stale_base

        _call_start = time.time()
        self._touch_activity("waiting for non-streaming API response")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        _poll_count = 0
        while t.is_alive():
            t.join(timeout=0.3)
            _poll_count += 1

            # 每 ~30 秒触摸一次活动，以便网关的非活动
            # 监视器知道我们在等待响应时仍然存活。
            if _poll_count % 100 == 0:  # 100 × 0.3s = 30s
                _elapsed = time.time() - _call_start
                self._touch_activity(
                    f"waiting for non-streaming response ({int(_elapsed)}s elapsed)"
                )

            # 陈旧调用检测器：如果在配置的超时内
            # 没有响应到达，则终止连接。
            _elapsed = time.time() - _call_start
            if _elapsed > _stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Non-streaming API call stale for %.0fs (threshold %.0fs). "
                    "model=%s context=~%s tokens. Killing connection.",
                    _elapsed, _stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_elapsed)}s "
                    f"(non-streaming, model: {api_kwargs.get('model', 'unknown')}). "
                    f"Aborting call."
                )
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        rc = request_client_holder.get("client")
                        if rc is not None:
                            self._close_request_openai_client(rc, reason="stale_call_kill")
                except Exception:
                    pass
                self._touch_activity(
                    f"stale non-streaming call killed after {int(_elapsed)}s"
                )
                # Wait briefly for the thread to notice the closed connection.
                t.join(timeout=2.0)
                if result["error"] is None and result["response"] is None:
                    result["error"] = TimeoutError(
                        f"Non-streaming API call timed out after {int(_elapsed)}s "
                        f"with no response (threshold: {int(_stale_timeout)}s)"
                    )
                break

            if self._interrupt_requested:
                # 强制关闭进行中的工作线程本地 HTTP 连接以停止
                # 令牌生成，而不会污染用于
                # 种子未来重试的共享客户端。
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    # ── Unified streaming API call ─────────────────────────────────────────

    def _reset_stream_delivery_tracking(self) -> None:
        """重置当前模型响应期间传递的文本的跟踪。"""
        self._current_streamed_assistant_text = ""

    def _record_streamed_assistant_text(self, text: str) -> None:
        """累积通过流回调发出的可见助手文本。"""
        if isinstance(text, str) and text:
            self._current_streamed_assistant_text = (
                getattr(self, "_current_streamed_assistant_text", "") + text
            )

    @staticmethod
    def _normalize_interim_visible_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _interim_content_was_streamed(self, content: str) -> bool:
        visible_content = self._normalize_interim_visible_text(
            self._strip_think_blocks(content or "")
        )
        if not visible_content:
            return False
        streamed = self._normalize_interim_visible_text(
            self._strip_think_blocks(getattr(self, "_current_streamed_assistant_text", "") or "")
        )
        return bool(streamed) and streamed == visible_content

    def _emit_interim_assistant_message(self, assistant_msg: Dict[str, Any]) -> None:
        """将真实的回合中助手评论消息呈现到 UI 层。"""
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None or not isinstance(assistant_msg, dict):
            return
        content = assistant_msg.get("content")
        visible = self._strip_think_blocks(content or "").strip()
        if not visible or visible == "(empty)":
            return
        already_streamed = self._interim_content_was_streamed(visible)
        try:
            cb(visible, already_streamed=already_streamed)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _fire_stream_delta(self, text: str) -> None:
        """触发所有注册的流增量回调（显示 + TTS）。"""
        # 如果工具迭代设置了中断标志，在第一个真实文本增量前
        # 前置单个段落中断。这防止了原始问题
        #（跨工具边界的文本连接），而不会在
        # 多个工具迭代背靠背运行时堆叠空行。
        if getattr(self, "_stream_needs_break", False) and text and text.strip():
            self._stream_needs_break = False
            text = "\n\n" + text
        callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
        delivered = False
        for cb in callbacks:
            try:
                cb(text)
                delivered = True
            except Exception:
                pass
        if delivered:
            self._record_streamed_assistant_text(text)

    def _fire_reasoning_delta(self, text: str) -> None:
        """如果注册了推理回调则触发。"""
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass

    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """通知显示层模型正在生成工具调用参数。

        当流式响应开始产生
        tool_call / tool_use 令牌时，每个工具名称触发一次。
        给 TUI 一个机会显示旋转器
        或状态行，以便用户在生成
        大型工具有效载荷（例如 45 KB 的 write_file）时
        不会盯着冻结的屏幕。
        """
        cb = self.tool_gen_callback
        if cb is not None:
            try:
                cb(tool_name)
            except Exception:
                pass

    def _has_stream_consumers(self) -> bool:
        """如果注册了任何流式消费者则返回 True。"""
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )

    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: callable = None
    ):
        """_interruptible_api_call 的流式变体，用于实时令牌传递。

        处理所有三种 api_modes：
        - chat_completions：在 OpenAI 兼容端点上使用 stream=True
        - anthropic_messages：通过 Anthropic SDK 使用 client.messages.stream()
        - codex_responses：委托给 _run_codex_stream（已经在流式传输）

        为每个文本令牌触发 stream_delta_callback 和 _stream_callback。
        工具调用回合抑制回调 — 仅纯文本最终响应
        流式传输给消费者。返回一个模仿
        非流式响应形状的 SimpleNamespace，以便代理循环的其余部分保持不变。

        在指示不支持流式传输的提供商错误上
        回退到 _interruptible_api_call。
        """
        if self.api_mode == "codex_responses":
            # Codex 通过 _run_codex_stream 内部流式传输。
            # _interruptible_api_call 中的主要调度已经调用它；
            # 我们只需要确保 on_first_delta 到达它。
            # 将其临时存储在实例上，以便 _run_codex_stream 可以拾取它。
            self._codex_on_first_delta = on_first_delta
            try:
                return self._interruptible_api_call(api_kwargs)
            finally:
                self._codex_on_first_delta = None

        # Bedrock Converse 使用 boto3 的 converse_stream() 和实时增量
        # 回调 — 与 Anthropic 和 chat_completions 流式传输相同的 UX。
        if self.api_mode == "bedrock_converse":
            result = {"response": None, "error": None}
            first_delta_fired = {"done": False}
            deltas_were_sent = {"yes": False}

            def _fire_first():
                if not first_delta_fired["done"] and on_first_delta:
                    first_delta_fired["done"] = True
                    try:
                        on_first_delta()
                    except Exception:
                        pass

            def _bedrock_call():
                try:
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        stream_converse_with_callbacks,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    raw_response = client.converse_stream(**api_kwargs)

                    def _on_text(text):
                        _fire_first()
                        self._fire_stream_delta(text)
                        deltas_were_sent["yes"] = True

                    def _on_tool(name):
                        _fire_first()
                        self._fire_tool_gen_started(name)

                    def _on_reasoning(text):
                        _fire_first()
                        self._fire_reasoning_delta(text)

                    result["response"] = stream_converse_with_callbacks(
                        raw_response,
                        on_text_delta=_on_text if self._has_stream_consumers() else None,
                        on_tool_start=_on_tool,
                        on_reasoning_delta=_on_reasoning if self.reasoning_callback or self.stream_delta_callback else None,
                        on_interrupt_check=lambda: self._interrupt_requested,
                    )
                except Exception as e:
                    result["error"] = e

            t = threading.Thread(target=_bedrock_call, daemon=True)
            t.start()
            while t.is_alive():
                t.join(timeout=0.3)
                if self._interrupt_requested:
                    raise InterruptedError("Agent interrupted during Bedrock API call")
            if result["error"] is not None:
                raise result["error"]
            return result["response"]

        result = {"response": None, "error": None, "partial_tool_names": []}
        request_client_holder = {"client": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}  # Track if any deltas were fired (for fallback)
        # 上次真实流式块的实际时间戳。外部
        # 轮询循环使用它来检测陈旧连接，这些连接不断接收
        # SSE keep-alive ping 但没有实际数据。
        last_chunk_time = {"t": time.time()}

        def _fire_first_delta():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _call_chat_completions():
            """流式传输聊天完成响应。"""
            import httpx as _httpx
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            # 本地提供商（Ollama、llama.cpp、vLLM）可能需要几分钟
            # 在大型上下文上进行预填充，然后产生第一个令牌。
            # 自动增加 httpx 读取超时，除非用户明确
            # 覆盖了 HERMES_STREAM_READ_TIMEOUT。
            if _stream_read_timeout == 120.0 and self.base_url and is_local_endpoint(self.base_url):
                _stream_read_timeout = _base_timeout
                logger.debug(
                    "Local provider detected (%s) — stream read timeout raised to %.0fs",
                    self.base_url, _stream_read_timeout,
                )
            stream_kwargs = {
                **api_kwargs,
                "stream": True,
                "stream_options": {"include_usage": True},
                "timeout": _httpx.Timeout(
                    connect=30.0,
                    read=_stream_read_timeout,
                    write=_base_timeout,
                    pool=30.0,
                ),
            }
            request_client_holder["client"] = self._create_request_openai_client(
                reason="chat_completion_stream_request"
            )
            # 重置陈旧流计时器，以便检测器从
            # 此尝试的开始测量，而不是前一次尝试的最后一个块。
            last_chunk_time["t"] = time.time()
            self._touch_activity("waiting for provider response (streaming)")
            stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

            # 从初始 HTTP 响应中捕获速率限制头部。
            # OpenAI SDK Stream 对象在消耗任何块之前
            # 通过 .response 暴露底层的 httpx 响应。
            self._capture_rate_limits(getattr(stream, "response", None))

            content_parts: list = []
            tool_calls_acc: dict = {}
            tool_gen_notified: set = set()
            # Ollama 兼容端点为并行批次中的每个工具调用
            # 重用索引 0，仅通过 id 区分它们。
            # 跟踪每个原始索引最后看到的 id，以便我们可以检测
            # 在相同索引开始的新工具调用并将其重定向到新槽。
            _last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
            _active_slot_by_idx: dict = {}  # raw_index -> current slot in tool_calls_acc
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None
            for chunk in stream:
                last_chunk_time["t"] = time.time()
                self._touch_activity("receiving stream response")

                if self._interrupt_requested:
                    break

                if not chunk.choices:
                    if hasattr(chunk, "model") and chunk.model:
                        model_name = chunk.model
                    # Usage comes in the final chunk with empty choices
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model

                # Accumulate reasoning content
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    _fire_first_delta()
                    self._fire_reasoning_delta(reasoning_text)

                # Accumulate text content — fire callback only when no tool calls
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        _fire_first_delta()
                        self._fire_stream_delta(delta.content)
                        deltas_were_sent["yes"] = True
                    else:
                        # Tool calls suppress regular content streaming (avoids
                        # displaying chatty "I'll use the tool..." text alongside
                        # tool calls).  But reasoning tags embedded in suppressed
                        # content should still reach the display — otherwise the
                        # reasoning box only appears as a post-response fallback,
                        # rendering it confusingly after the already-streamed
                        # response.  Route suppressed content through the stream
                        # delta callback so its tag extraction can fire the
                        # reasoning display.  Non-reasoning text is harmlessly
                        # suppressed by the CLI's _stream_delta when the stream
                        # box is already closed (tool boundary flush).
                        if self.stream_delta_callback:
                            try:
                                self.stream_delta_callback(delta.content)
                                self._record_streamed_assistant_text(delta.content)
                            except Exception:
                                pass

                # Accumulate tool call deltas — notify display on first name
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        raw_idx = tc_delta.index if tc_delta.index is not None else 0
                        delta_id = tc_delta.id or ""

                        # Ollama fix: detect a new tool call reusing the same
                        # raw index (different id) and redirect to a fresh slot.
                        if raw_idx not in _active_slot_by_idx:
                            _active_slot_by_idx[raw_idx] = raw_idx
                        if (
                            delta_id
                            and raw_idx in _last_id_at_idx
                            and delta_id != _last_id_at_idx[raw_idx]
                        ):
                            new_slot = max(tool_calls_acc, default=-1) + 1
                            _active_slot_by_idx[raw_idx] = new_slot
                        if delta_id:
                            _last_id_at_idx[raw_idx] = delta_id
                        idx = _active_slot_by_idx[raw_idx]

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "extra_content": None,
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                # 使用赋值，而不是 +=。函数名是
                                # 在第一个块中完整传递的原子标识符（OpenAI 规范）。
                                # 一些提供商（通过 NVIDIA NIM 的 MiniMax M2.7）在每个块中
                                # 重新发送完整名称；连接会产生
                                # "read_fileread_file"。赋值
                                # （匹配 OpenAI Node SDK / LiteLLM /
                                # Vercel AI 模式）对此免疫。
                                entry["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments
                        extra = getattr(tc_delta, "extra_content", None)
                        if extra is None and hasattr(tc_delta, "model_extra"):
                            extra = (tc_delta.model_extra or {}).get("extra_content")
                        if extra is not None:
                            if hasattr(extra, "model_dump"):
                                extra = extra.model_dump()
                            entry["extra_content"] = extra
                        # 当完整名称可用时，每个工具触发一次
                        name = entry["function"]["name"]
                        if name and idx not in tool_gen_notified:
                            tool_gen_notified.add(idx)
                            _fire_first_delta()
                            self._fire_tool_gen_started(name)
                            # 记录部分工具调用名称，以便外部
                            # 存根构建器可以在流式传输在此工具的参数
                            # 完全传递之前死亡时显示用户可见的警告。
                            # 没有这个，工具调用 JSON 生成期间的停滞
                            # 会让第 ~6107 行的存根返回 `tool_calls=None`，
                            # 静默地丢弃尝试的操作。
                            result["partial_tool_names"].append(name)

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Usage in the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # 构建匹配非流式形状的模拟响应
            full_content = "".join(content_parts) or None
            mock_tool_calls = None
            has_truncated_tool_args = False
            if tool_calls_acc:
                mock_tool_calls = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    arguments = tc["function"]["arguments"]
                    if arguments and arguments.strip():
                        try:
                            json.loads(arguments)
                        except json.JSONDecodeError:
                            has_truncated_tool_args = True
                    mock_tool_calls.append(SimpleNamespace(
                        id=tc["id"],
                        type=tc["type"],
                        extra_content=tc.get("extra_content"),
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=arguments,
                        ),
                    ))

            effective_finish_reason = finish_reason or "stop"
            if has_truncated_tool_args:
                effective_finish_reason = "length"

            full_reasoning = "".join(reasoning_parts) or None
            mock_message = SimpleNamespace(
                role=role,
                content=full_content,
                tool_calls=mock_tool_calls,
                reasoning_content=full_reasoning,
            )
            mock_choice = SimpleNamespace(
                index=0,
                message=mock_message,
                finish_reason=effective_finish_reason,
            )
            return SimpleNamespace(
                id="stream-" + str(uuid.uuid4()),
                model=model_name,
                choices=[mock_choice],
                usage=usage_obj,
            )

        def _call_anthropic():
            """流式传输 Anthropic Messages API 响应。

            为实时令牌传递触发增量回调，但
            从 get_final_message() 返回
            原生 Anthropic Message 对象，以便
            代理循环的其余部分（验证、工具提取等）
            正常工作。
            """
            has_tool_use = False

            # 为此尝试重置陈旧流计时器
            last_chunk_time["t"] = time.time()
            # 使用 Anthropic SDK 的流式上下文管理器
            with self._anthropic_client.messages.stream(**api_kwargs) as stream:
                for event in stream:
                    # 在每个事件上更新陈旧流计时器，以便
                    # 外部轮询循环知道数据正在流动。
                    # 没有这个，检测器会在 180 秒后杀死健康的
                    # 长时间运行的 Opus 流，即使事件正在
                    # 积极到达（chat_completions 路径
                    # 在其块循环顶部已经这样做）。
                    last_chunk_time["t"] = time.time()
                    self._touch_activity("receiving stream response")

                    if self._interrupt_requested:
                        break

                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            has_tool_use = True
                            tool_name = getattr(block, "name", None)
                            if tool_name:
                                _fire_first_delta()
                                self._fire_tool_gen_started(tool_name)

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text and not has_tool_use:
                                    _fire_first_delta()
                                    self._fire_stream_delta(text)
                                    deltas_were_sent["yes"] = True
                            elif delta_type == "thinking_delta":
                                thinking_text = getattr(delta, "thinking", "")
                                if thinking_text:
                                    _fire_first_delta()
                                    self._fire_reasoning_delta(thinking_text)

                # 返回原生 Anthropic Message 以进行下游处理
                return stream.get_final_message()

        def _call():
            import httpx as _httpx

            _max_stream_retries = int(os.getenv("HERMES_STREAM_RETRIES", 2))

            try:
                for _stream_attempt in range(_max_stream_retries + 1):
                    try:
                        if self.api_mode == "anthropic_messages":
                            self._try_refresh_anthropic_client_credentials()
                            result["response"] = _call_anthropic()
                        else:
                            result["response"] = _call_chat_completions()
                        return  # success
                    except Exception as e:
                        if deltas_were_sent["yes"]:
                            # 流式传输在一些令牌已经
                            # 传递后失败。不要重试或回退 — 部分
                            # 内容已经到达用户。
                            logger.warning(
                                "Streaming failed after partial delivery, not retrying: %s", e
                            )
                            result["error"] = e
                            return

                        _is_timeout = isinstance(
                            e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout)
                        )
                        _is_conn_err = isinstance(
                            e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError)
                        )

                        # 来自代理的 SSE 错误事件（例如 OpenRouter 发送
                        # {"error":{"message":"Network connection lost."}}）
                        # 被 OpenAI SDK 作为 APIError 引发。
                        # 这些在语义上与 httpx 连接断开相同 —
                        # 上游流死亡 — 应该使用
                        # 新连接重试。与 HTTP 错误区分：
                        # 来自 SSE 的 APIError 没有 status_code，而
                        # APIStatusError (4xx/5xx) 总是有。
                        _is_sse_conn_err = False
                        if not _is_timeout and not _is_conn_err:
                            from openai import APIError as _APIError
                            if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                _err_lower_sse = str(e).lower()
                                _SSE_CONN_PHRASES = (
                                    "connection lost",
                                    "connection reset",
                                    "connection closed",
                                    "connection terminated",
                                    "network error",
                                    "network connection",
                                    "terminated",
                                    "peer closed",
                                    "broken pipe",
                                    "upstream connect error",
                                )
                                _is_sse_conn_err = any(
                                    phrase in _err_lower_sse
                                    for phrase in _SSE_CONN_PHRASES
                                )

                        if _is_timeout or _is_conn_err or _is_sse_conn_err:
                            # 瞬态网络/超时错误。首先使用
                            # 新连接重试流式请求。
                            if _stream_attempt < _max_stream_retries:
                                logger.info(
                                    "Streaming attempt %s/%s failed (%s: %s), "
                                    "retrying with fresh connection...",
                                    _stream_attempt + 1,
                                    _max_stream_retries + 1,
                                    type(e).__name__,
                                    e,
                                )
                                self._emit_status(
                                    f"⚠️ Connection to provider dropped "
                                    f"({type(e).__name__}). Reconnecting… "
                                    f"(attempt {_stream_attempt + 2}/{_max_stream_retries + 1})"
                                )
                                self._touch_activity(
                                    f"stream retry {_stream_attempt + 2}/{_max_stream_retries + 1} "
                                    f"after {type(e).__name__}"
                                )
                                # 重试前关闭陈旧的请求客户端
                                stale = request_client_holder.get("client")
                                if stale is not None:
                                    self._close_request_openai_client(
                                        stale, reason="stream_retry_cleanup"
                                    )
                                    request_client_holder["client"] = None
                                # 同时重建主客户端以清除
                                # 池中的任何死连接。
                                try:
                                    self._replace_primary_openai_client(
                                        reason="stream_retry_pool_cleanup"
                                    )
                                except Exception:
                                    pass
                                self._emit_status("🔄 Reconnected — resuming…")
                                continue
                            self._emit_status(
                                "❌ Connection to provider failed after "
                                f"{_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                            logger.warning(
                                "Streaming exhausted %s retries on transient error: %s",
                                _max_stream_retries + 1,
                                e,
                            )
                        else:
                            _err_lower = str(e).lower()
                            _is_stream_unsupported = (
                                "stream" in _err_lower
                                and "not supported" in _err_lower
                            )
                            if _is_stream_unsupported:
                                self._disable_streaming = True
                                self._safe_print(
                                    "\n⚠  Streaming is not supported for this "
                                    "model/provider. Switching to non-streaming.\n"
                                    "   To avoid this delay, set display.streaming: false "
                                    "in config.yaml\n"
                                )
                            logger.info(
                                "Streaming failed before delivery: %s",
                                e,
                            )

                        # 将错误传播到主重试循环，而不是
                        # 内联回退到非流式。主循环有
                        # 更丰富的恢复：凭证轮换、提供商回退、
                        # 退避，并且 — 对于"不支持流式" — 将通过
                        # _disable_streaming 在下一次尝试时切换到非流式。
                        result["error"] = e
                        return
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="stream_request_complete")

        _stream_stale_timeout_base = float(os.getenv("HERMES_STREAM_STALE_TIMEOUT", 180.0))
        # 本地提供商（Ollama、oMLX、llama-cpp）可能需要 300+ 秒
        # 在大型上下文上进行预填充。禁用陈旧检测器，除非
        # 用户明确设置了 HERMES_STREAM_STALE_TIMEOUT。
        if _stream_stale_timeout_base == 180.0 and self.base_url and is_local_endpoint(self.base_url):
            _stream_stale_timeout = float("inf")
            logger.debug("Local provider detected (%s) — stale stream timeout disabled", self.base_url)
        else:
            # 为大型上下文扩展陈旧超时：慢速模型（如 Opus）
            # 在上下文大时可以在产生第一个令牌之前
            # 合法地思考几分钟。
            # 没有这个，陈旧检测器在模型的思考阶段
            # 杀死健康连接，产生
            # 虚假的 RemoteProtocolError（"对等方关闭连接"）。
            _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
            if _est_tokens > 100_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
            elif _est_tokens > 50_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
            else:
                _stream_stale_timeout = _stream_stale_timeout_base

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        _last_heartbeat = time.time()
        _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
        while t.is_alive():
            t.join(timeout=0.3)

            # 定期心跳：触摸代理的活动跟踪器，以便
            # 网关的非活动监视器知道我们在等待
            # 流块时仍然存活。
            # 没有这个，长时间的思考暂停（例如。
            # 推理模型）或本地提供商（Ollama）上的缓慢预填充
            # 会触发错误的非活动超时。_call 线程在每个块上
            # 触摸活动，但 API 调用开始和
            # 第一个块之间的间隔可能超过网关超时 — 特别是
            # 当陈旧流超时被禁用时（本地提供商）。
            _hb_now = time.time()
            if _hb_now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
                _last_heartbeat = _hb_now
                _waiting_secs = int(_hb_now - last_chunk_time["t"])
                self._touch_activity(
                    f"waiting for stream response ({_waiting_secs}s, no chunks yet)"
                )

            # 检测陈旧流：由 SSE ping 保持活跃
            # 但不传递真实块的连接。终止客户端以便
            # 内部重试循环可以开始新连接。
            _stale_elapsed = time.time() - last_chunk_time["t"]
            if _stale_elapsed > _stream_stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                    "model=%s context=~%s tokens. Killing connection.",
                    _stale_elapsed, _stream_stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                    f"(model: {api_kwargs.get('model', 'unknown')}, "
                    f"context: ~{_est_ctx:,} tokens). "
                    f"Reconnecting..."
                )
                try:
                    rc = request_client_holder.get("client")
                    if rc is not None:
                        self._close_request_openai_client(rc, reason="stale_stream_kill")
                except Exception:
                    pass
                # 同时重建主客户端 — 其连接池
                # 可能持有来自同一提供商中断的死套接字。
                try:
                    self._replace_primary_openai_client(reason="stale_stream_pool_cleanup")
                except Exception:
                    pass
                # 重置计时器，以便在
                # 内部线程处理关闭时我们不会重复终止。
                last_chunk_time["t"] = time.time()
                self._touch_activity(
                    f"stale stream detected after {int(_stale_elapsed)}s, reconnecting"
                )

            if self._interrupt_requested:
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="stream_interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during streaming API call")
        if result["error"] is not None:
            if deltas_were_sent["yes"]:
                # 流式传输在一些令牌已经传递到
                # 平台后失败。重新引发会让外部重试循环
                # 进行新的 API 调用，创建重复消息。
                # 返回部分 "stop" 响应，以便外部循环将此
                # 回合视为完成（不重试，不回退）。
                # 恢复已经流式传输给用户的任何内容。
                # _current_streamed_assistant_text 累积通过
                # _fire_stream_delta 触发的文本，因此它具有
                # 连接死亡前用户看到的确切内容。
                _partial_text = (
                    getattr(self, "_current_streamed_assistant_text", "") or ""
                ).strip() or None

                # 如果流在模型发出工具调用时死亡，
                # 下面的存根将静默设置 `tool_calls=None`，
                # 并且代理循环将把回合视为完成 — 尝试的
                # 操作丢失，没有面向用户的信号。
                # 向存根内容附加人类可见的警告，以便 (a) 用户
                # 知道某些事情失败了，并且 (b) 下一回合的模型在
                # 对话历史中看到尝试的内容并可以重试。
                _partial_names = list(result.get("partial_tool_names") or [])
                if _partial_names:
                    _name_str = ", ".join(_partial_names[:3])
                    if len(_partial_names) > 3:
                        _name_str += f", +{len(_partial_names) - 3} more"
                    _warn = (
                        f"\n\n⚠ Stream stalled mid tool-call "
                        f"({_name_str}); the action was not executed. "
                        f"Ask me to retry if you want to continue."
                    )
                    _partial_text = (_partial_text or "") + _warn
                    # 同时作为流式增量触发，以便用户现在看到它
                    # 而不仅仅是在持久化的记录中。
                    try:
                        self._fire_stream_delta(_warn)
                    except Exception:
                        pass
                    logger.warning(
                        "Partial stream dropped tool call(s) %s after %s chars "
                        "of text; surfaced warning to user: %s",
                        _partial_names, len(_partial_text or ""), result["error"],
                    )
                else:
                    logger.warning(
                        "Partial stream delivered before error; returning stub "
                        "response with %s chars of recovered content to prevent "
                        "duplicate messages: %s",
                        len(_partial_text or ""),
                        result["error"],
                    )
                _stub_msg = SimpleNamespace(
                    role="assistant", content=_partial_text, tool_calls=None,
                    reasoning_content=None,
                )
                return SimpleNamespace(
                    id="partial-stream-stub",
                    model=getattr(self, "model", "unknown"),
                    choices=[SimpleNamespace(
                        index=0, message=_stub_msg, finish_reason="stop",
                    )],
                    usage=None,
                )
            raise result["error"]
        return result["response"]

    # ── Provider fallback ──────────────────────────────────────────────────

    def _try_activate_fallback(self) -> bool:
        """切换到链中的下一个回退模型/提供商。

        在当前模型重试失败后调用。就地交换
        OpenAI 客户端、模型 slug 和提供商，以便重试循环
        可以使用新后端继续。在每次调用时
        推进链；耗尽时返回 False。

        使用集中式提供商路由器（resolve_provider_client）进行
        身份验证解析和客户端构建 — 没有重复的提供商→密钥
        映射。
        """
        if self._fallback_index >= len(self._fallback_chain):
            return False

        fb = self._fallback_chain[self._fallback_index]
        self._fallback_index += 1
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            return self._try_activate_fallback()  # skip invalid, try next

        # 使用集中式路由器进行客户端构建。
        # raw_codex=True 因为主代理需要直接的 responses.stream()
        # 访问 Codex 提供商。
        try:
            from agent.auxiliary_client import resolve_provider_client
            # 从回退配置传递 base_url 和 api_key，以便自定义
            # 端点（例如 Ollama Cloud）正确解析，而不是
            # 回退到 OpenRouter 默认值。
            fb_base_url_hint = (fb.get("base_url") or "").strip() or None
            fb_api_key_hint = (fb.get("api_key") or "").strip() or None
            # 对于 Ollama Cloud 端点，当回退配置中没有
            # 显式密钥时，从环境变量中提取 OLLAMA_API_KEY。
            if fb_base_url_hint and "ollama.com" in fb_base_url_hint.lower() and not fb_api_key_hint:
                fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None
            fb_client, _resolved_fb_model = resolve_provider_client(
                fb_provider, model=fb_model, raw_codex=True,
                explicit_base_url=fb_base_url_hint,
                explicit_api_key=fb_api_key_hint)
            if fb_client is None:
                logging.warning(
                    "Fallback to %s failed: provider not configured",
                    fb_provider)
                return self._try_activate_fallback()  # try next in chain
            try:
                from hermes_cli.model_normalize import normalize_model_for_provider

                fb_model = normalize_model_for_provider(fb_model, fb_provider)
            except Exception:
                pass

            # Determine api_mode from provider / base URL / model
            fb_api_mode = "chat_completions"
            fb_base_url = str(fb_client.base_url)
            if fb_provider == "openai-codex":
                fb_api_mode = "codex_responses"
            elif fb_provider == "anthropic" or fb_base_url.rstrip("/").lower().endswith("/anthropic"):
                fb_api_mode = "anthropic_messages"
            elif self._is_direct_openai_url(fb_base_url):
                fb_api_mode = "codex_responses"
            elif self._provider_model_requires_responses_api(
                fb_model,
                provider=fb_provider,
            ):
                # GPT-5.x 模型通常需要 Responses API，但保留
                # 提供商特定的例外，如聊天完成上的 Copilot gpt-5-mini。
                fb_api_mode = "codex_responses"
            elif fb_provider == "bedrock" or "bedrock-runtime" in fb_base_url.lower():
                fb_api_mode = "bedrock_converse"

            old_model = self.model
            self.model = fb_model
            self.provider = fb_provider
            self.base_url = fb_base_url
            self.api_mode = fb_api_mode
            self._fallback_activated = True

            if fb_api_mode == "anthropic_messages":
                # 构建原生 Anthropic 客户端而不是使用 OpenAI 客户端
                from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token, _is_oauth_token
                effective_key = (fb_client.api_key or resolve_anthropic_token() or "") if fb_provider == "anthropic" else (fb_client.api_key or "")
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = fb_base_url
                self._anthropic_client = build_anthropic_client(effective_key, self._anthropic_base_url)
                self._is_anthropic_oauth = _is_oauth_token(effective_key)
                self.client = None
                self._client_kwargs = {}
            else:
                # 就地交换 OpenAI 客户端和配置
                self.api_key = fb_client.api_key
                self.client = fb_client
                fb_headers = getattr(fb_client, "_custom_headers", None)
                if not fb_headers:
                    fb_headers = getattr(fb_client, "default_headers", None)
                self._client_kwargs = {
                    "api_key": fb_client.api_key,
                    "base_url": fb_base_url,
                    **({"default_headers": dict(fb_headers)} if fb_headers else {}),
                }

            # 为新的提供商/模型重新评估提示缓存
            is_native_anthropic = fb_api_mode == "anthropic_messages" and fb_provider == "anthropic"
            self._use_prompt_caching = (
                ("openrouter" in fb_base_url.lower() and "claude" in fb_model.lower())
                or is_native_anthropic
            )

            # 更新回退模型的上下文压缩器限制。
            # 没有这个，压缩决策使用主模型的
            # 上下文窗口（例如 200K）而不是回退的（例如 32K），
            # 导致过大的会话溢出回退。
            if hasattr(self, 'context_compressor') and self.context_compressor:
                from agent.model_metadata import get_model_context_length
                fb_context_length = get_model_context_length(
                    self.model, base_url=self.base_url,
                    api_key=self.api_key, provider=self.provider,
                )
                self.context_compressor.update_model(
                    model=self.model,
                    context_length=fb_context_length,
                    base_url=self.base_url,
                    api_key=getattr(self, "api_key", ""),
                    provider=self.provider,
                )

            self._emit_status(
                f"🔄 Primary model failed — switching to fallback: "
                f"{fb_model} via {fb_provider}"
            )
            logging.info(
                "Fallback activated: %s → %s (%s)",
                old_model, fb_model, fb_provider,
            )
            return True
        except Exception as e:
            logging.error("Failed to activate fallback %s: %s", fb_model, e)
            return self._try_activate_fallback()  # try next in chain

    # ── Per-turn primary restoration ─────────────────────────────────────

    def _restore_primary_runtime(self) -> bool:
        """在新回合开始时恢复主运行时。

        在长时间运行的 CLI 会话中，单个 AIAgent 实例跨越多个
        回合。没有恢复，一个瞬态故障会将会话
        固定到每个后续回合的回退提供商。
        在 ``run_conversation()`` 顶部调用此函数
        使回退成为回合作用域。

        网关在消息之间缓存代理（``gateway/run.py`` 中的
        ``_agent_cache``），因此那里也需要此恢复。
        """
        if not self._fallback_activated:
            return False

        rt = self._primary_runtime
        try:
            # ── 核心运行时状态 ──
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]           # setter 更新 _base_url_lower
            self.api_mode = rt["api_mode"]
            self.api_key = rt["api_key"]
            self._client_kwargs = dict(rt["client_kwargs"])
            self._use_prompt_caching = rt["use_prompt_caching"]

            # ── 为主提供商重建客户端 ──
            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="restore_primary",
                    shared=True,
                )

            # ── 恢复上下文引擎状态 ──
            cc = self.context_compressor
            cc.update_model(
                model=rt["compressor_model"],
                context_length=rt["compressor_context_length"],
                base_url=rt["compressor_base_url"],
                api_key=rt["compressor_api_key"],
                provider=rt["compressor_provider"],
            )

            # ── 为新回合重置回退链 ──
            self._fallback_activated = False
            self._fallback_index = 0

            logging.info(
                "Primary runtime restored for new turn: %s (%s)",
                self.model, self.provider,
            )
            return True
        except Exception as e:
            logging.warning("Failed to restore primary runtime: %s", e)
            return False

    # 哪些错误类型表示瞬态传输故障，值得
    # 使用重建的客户端/连接池再尝试一次。
    _TRANSIENT_TRANSPORT_ERRORS = frozenset({
        "ReadTimeout", "ConnectTimeout", "PoolTimeout",
        "ConnectError", "RemoteProtocolError",
        "APIConnectionError", "APITimeoutError",
    })

    def _try_recover_primary_transport(
        self, api_error: Exception, *, retry_count: int, max_retries: int,
    ) -> bool:
        """为瞬态传输故障尝试一个额外的主提供商恢复周期。

        在 ``max_retries`` 耗尽后，重建主客户端（清除
        陈旧连接池）并在回退前再给它
        一次机会。这对于直接端点（自定义、Z.AI、
        Anthropic、OpenAI、本地模型）最有用，其中 TCP 级别的
        小故障并不意味着提供商已关闭。

        跳过代理/聚合提供商（OpenRouter、Nous），它们
        已经在服务器端管理连接池和重试 — 如果我们
        通过它们的重试已耗尽，再重建一个客户端也无济于事。
        """
        if self._fallback_activated:
            return False

        # 仅用于瞬态传输错误
        error_type = type(api_error).__name__
        if error_type not in self._TRANSIENT_TRANSPORT_ERRORS:
            return False

        # 跳过聚合提供商 — 它们管理自己的重试基础设施
        if self._is_openrouter_url():
            return False
        provider_lower = (self.provider or "").strip().lower()
        if provider_lower in ("nous", "nous-research"):
            return False

        try:
            # 关闭现有客户端以释放陈旧连接
            if getattr(self, "client", None) is not None:
                try:
                    self._close_openai_client(
                        self.client, reason="primary_recovery", shared=True,
                    )
                except Exception:
                    pass

            # 从主快照重建
            rt = self._primary_runtime
            self._client_kwargs = dict(rt["client_kwargs"])
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]
            self.api_mode = rt["api_mode"]
            self.api_key = rt["api_key"]

            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="primary_recovery",
                    shared=True,
                )

            wait_time = min(3 + retry_count, 8)
            self._vprint(
                f"{self.log_prefix}🔁 Transient {error_type} on {self.provider} — "
                f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
                force=True,
            )
            time.sleep(wait_time)
            return True
        except Exception as e:
            logging.warning("Primary transport recovery failed: %s", e)
            return False

    # ── End provider fallback ──────────────────────────────────────────────

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        header, _, data = str(image_url or "").partition(",")
        mime = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                mime = mime_part
        suffix = {
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")
        tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
        with tmp:
            tmp.write(base64.b64decode(data))
        path = Path(tmp.name)
        return str(path), path

    def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
        cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
        cached = self._anthropic_image_fallback_cache.get(cache_key)
        if cached:
            return cached

        role_label = {
            "assistant": "assistant",
            "tool": "tool result",
        }.get(role, "user")
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, UI, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        vision_source = str(image_url or "")
        cleanup_path: Optional[Path] = None
        if vision_source.startswith("data:"):
            vision_source, cleanup_path = self._materialize_data_url_for_vision(vision_source)

        description = ""
        try:
            from tools.vision_tools import vision_analyze_tool

            result_json = asyncio.run(
                vision_analyze_tool(image_url=vision_source, user_prompt=analysis_prompt)
            )
            result = json.loads(result_json) if isinstance(result_json, str) else {}
            description = (result.get("analysis") or "").strip()
        except Exception as e:
            description = f"Image analysis failed: {e}"
        finally:
            if cleanup_path and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    pass

        if not description:
            description = "Image analysis failed."

        note = f"[The {role_label} attached an image. Here's what it contains:\n{description}]"
        if vision_source and not str(image_url or "").startswith("data:"):
            note += (
                f"\n[If you need a closer look, use vision_analyze with image_url: {vision_source}]"
            )

        self._anthropic_image_fallback_cache[cache_key] = note
        return note

    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        if not self._content_has_image_parts(content):
            return content

        text_parts: List[str] = []
        image_notes: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
                continue

            if ptype in {"image_url", "input_image"}:
                image_data = part.get("image_url", {})
                image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
                if image_url:
                    image_notes.append(self._describe_image_for_anthropic_fallback(image_url, role))
                else:
                    image_notes.append("[An image was attached but no image source was available.]")
                continue

            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)

        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(text for text in text_parts if text).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        if prefix:
            return prefix
        if suffix:
            return suffix
        return "[A multimodal message was converted to text for Anthropic compatibility.]"

    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    def _anthropic_preserve_dots(self) -> bool:
        """True when using an anthropic-compatible endpoint that preserves dots in model names.
        Alibaba/DashScope keeps dots (e.g. qwen3.5-plus).
        MiniMax keeps dots (e.g. MiniMax-M2.7).
        OpenCode Go/Zen keeps dots for non-Claude models (e.g. minimax-m2.5-free).
        ZAI/Zhipu keeps dots (e.g. glm-4.7, glm-5.1)."""
        if (getattr(self, "provider", "") or "").lower() in {"alibaba", "minimax", "minimax-cn", "opencode-go", "opencode-zen", "zai"}:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        return "dashscope" in base or "aliyuncs" in base or "minimax" in base or "opencode.ai/zen/" in base or "bigmodel.cn" in base

    def _is_qwen_portal(self) -> bool:
        """Return True when the base URL targets Qwen Portal."""
        return "portal.qwen.ai" in self._base_url_lower

    def _qwen_prepare_chat_messages(self, api_messages: list) -> list:
        prepared = copy.deepcopy(api_messages)
        if not prepared:
            return prepared

        for msg in prepared:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # Normalize: convert bare strings to text dicts, keep dicts as-is.
                # deepcopy already created independent copies, no need for dict().
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        # Inject cache_control on the last part of the system message.
        for msg in prepared:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        return prepared

    def _qwen_prepare_chat_messages_inplace(self, messages: list) -> None:
        """就地变体 — 改变已复制的消息列表。"""
        if not messages:
            return

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """为活动 API 模式构建关键字参数字典。"""
        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_kwargs
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            # 传递 context_length（总输入+输出窗口），以便适配器可以在
            # 用户配置了比模型原生输出限制更小的上下文窗口时
            # 限制 max_tokens（输出上限）。
            ctx_len = getattr(self, "context_compressor", None)
            ctx_len = ctx_len.context_length if ctx_len else None
            # 当 API 返回"给定提示的 max_tokens 过大"时，
            # _ephemeral_max_output_tokens 为一次调用设置 — 它将输出限制到
            # 可用窗口空间，而不触及 context_length。
            ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
            if ephemeral_out is not None:
                self._ephemeral_max_output_tokens = None  # consume immediately
            return build_anthropic_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=self.tools,
                max_tokens=ephemeral_out if ephemeral_out is not None else self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=self._is_anthropic_oauth,
                preserve_dots=self._anthropic_preserve_dots(),
                context_length=ctx_len,
                base_url=getattr(self, "_anthropic_base_url", None),
                fast_mode=(self.request_overrides or {}).get("speed") == "fast",
            )

        # AWS Bedrock 原生 Converse API — 完全绕过 OpenAI 客户端。
        # 适配器直接处理消息/工具转换和 boto3 调用。
        if self.api_mode == "bedrock_converse":
            from agent.bedrock_adapter import build_converse_kwargs
            region = getattr(self, "_bedrock_region", None) or "us-east-1"
            guardrail = getattr(self, "_bedrock_guardrail_config", None)
            return {
                "__bedrock_converse__": True,
                "__bedrock_region__": region,
                **build_converse_kwargs(
                    model=self.model,
                    messages=api_messages,
                    tools=self.tools,
                    max_tokens=self.max_tokens or 4096,
                    temperature=None,  # Let the model use its default
                    guardrail_config=guardrail,
                ),
            }

        if self.api_mode == "codex_responses":
            instructions = ""
            payload_messages = api_messages
            if api_messages and api_messages[0].get("role") == "system":
                instructions = str(api_messages[0].get("content") or "").strip()
                payload_messages = api_messages[1:]
            if not instructions:
                instructions = DEFAULT_AGENT_IDENTITY

            is_github_responses = (
                "models.github.ai" in self.base_url.lower()
                or "api.githubcopilot.com" in self.base_url.lower()
            )
            is_codex_backend = (
                self.provider == "openai-codex"
                or "chatgpt.com/backend-api/codex" in self.base_url.lower()
            )

            # 解析推理努力程度：配置 > 默认（中等）
            reasoning_effort = "medium"
            reasoning_enabled = True
            if self.reasoning_config and isinstance(self.reasoning_config, dict):
                if self.reasoning_config.get("enabled") is False:
                    reasoning_enabled = False
                elif self.reasoning_config.get("effort"):
                    reasoning_effort = self.reasoning_config["effort"]

            # 限制 Responses API 模型不支持的努力程度级别。
            # GPT-5.4 支持 none/low/medium/high/xhigh 但不支持 "minimal"。
            # "minimal" 在 OpenRouter 和 GPT-5 上有效，但在 5.2/5.4 上失败。
            _effort_clamp = {"minimal": "low"}
            reasoning_effort = _effort_clamp.get(reasoning_effort, reasoning_effort)

            kwargs = {
                "model": self.model,
                "instructions": instructions,
                "input": self._chat_messages_to_responses_input(payload_messages),
                "tools": self._responses_tools(),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "store": False,
            }

            if not is_github_responses:
                kwargs["prompt_cache_key"] = self.session_id

            is_xai_responses = self.provider == "xai" or "api.x.ai" in (self.base_url or "").lower()

            if reasoning_enabled and is_xai_responses:
                # xAI 自动推理 — 没有努力参数，只需包含加密内容
                kwargs["include"] = ["reasoning.encrypted_content"]
            elif reasoning_enabled:
                if is_github_responses:
                    # Copilot 的 Responses 路由宣传支持 reasoning-effort，
                    # 但不支持 OpenAI 特定的提示缓存或加密推理
                    # 字段。将有效负载保持在文档记录的子集内。
                    github_reasoning = self._github_models_reasoning_extra_body()
                    if github_reasoning is not None:
                        kwargs["reasoning"] = github_reasoning
                else:
                    kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                    kwargs["include"] = ["reasoning.encrypted_content"]
            elif not is_github_responses and not is_xai_responses:
                kwargs["include"] = []

            if self.request_overrides:
                kwargs.update(self.request_overrides)

            if self.max_tokens is not None and not is_codex_backend:
                kwargs["max_output_tokens"] = self.max_tokens

            if is_xai_responses and getattr(self, "session_id", None):
                kwargs["extra_headers"] = {"x-grok-conv-id": self.session_id}

            return kwargs

        sanitized_messages = api_messages
        needs_sanitization = False
        for msg in api_messages:
            if not isinstance(msg, dict):
                continue
            if "codex_reasoning_items" in msg:
                needs_sanitization = True
                break

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    if "call_id" in tool_call or "response_item_id" in tool_call:
                        needs_sanitization = True
                        break
                if needs_sanitization:
                    break

        if needs_sanitization:
            sanitized_messages = copy.deepcopy(api_messages)
            for msg in sanitized_messages:
                if not isinstance(msg, dict):
                    continue

                # 仅 Codex 的重放状态不能泄漏到严格的聊天完成 API 中。
                msg.pop("codex_reasoning_items", None)

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call.pop("call_id", None)
                            tool_call.pop("response_item_id", None)

        # Qwen portal：将内容标准化为字典列表，注入 cache_control。
        # 必须在 codex 清理之后运行，以便我们转换最终消息。
        # 如果清理已经深拷贝，则重用该副本（就地）。
        if self._is_qwen_portal():
            if sanitized_messages is api_messages:
                # 没有进行清理 — 我们需要自己的副本。
                sanitized_messages = self._qwen_prepare_chat_messages(sanitized_messages)
            else:
                # 已经是深拷贝 — 就地转换以避免第二次深拷贝。
                self._qwen_prepare_chat_messages_inplace(sanitized_messages)

        # GPT-5 和 Codex 模型对 'developer' 的响应比对 'system' 更好
        # 用于指令遵循。在 API 边界交换角色，以便
        # 内部消息表示保持统一（"system"）。
        _model_lower = (self.model or "").lower()
        if (
            sanitized_messages
            and sanitized_messages[0].get("role") == "system"
            and any(p in _model_lower for p in DEVELOPER_ROLE_MODELS)
        ):
            # 浅复制列表 + 仅第一条消息 — 其余保持共享。
            sanitized_messages = list(sanitized_messages)
            sanitized_messages[0] = {**sanitized_messages[0], "role": "developer"}

        provider_preferences = {}
        if self.providers_allowed:
            provider_preferences["only"] = self.providers_allowed
        if self.providers_ignored:
            provider_preferences["ignore"] = self.providers_ignored
        if self.providers_order:
            provider_preferences["order"] = self.providers_order
        if self.provider_sort:
            provider_preferences["sort"] = self.provider_sort
        if self.provider_require_parameters:
            provider_preferences["require_parameters"] = True
        if self.provider_data_collection:
            provider_preferences["data_collection"] = self.provider_data_collection

        api_kwargs = {
            "model": self.model,
            "messages": sanitized_messages,
            "timeout": float(os.getenv("HERMES_API_TIMEOUT", 1800.0)),
        }
        try:
            from agent.auxiliary_client import _fixed_temperature_for_model
        except Exception:
            _fixed_temperature_for_model = None
        if _fixed_temperature_for_model is not None:
            fixed_temperature = _fixed_temperature_for_model(self.model)
            if fixed_temperature is not None:
                api_kwargs["temperature"] = fixed_temperature
        if self._is_qwen_portal():
            api_kwargs["metadata"] = {
                "sessionId": self.session_id or "hermes",
                "promptId": str(uuid.uuid4()),
            }
        if self.tools:
            api_kwargs["tools"] = self.tools

        # ── chat_completions 的 max_tokens ──────────────────────────────
        # 优先级：临时覆盖（错误恢复 / 长度延续
        # 提升）> 用户配置的 max_tokens > 提供商特定默认值。
        _ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
        if _ephemeral_out is not None:
            self._ephemeral_max_output_tokens = None  # consume immediately
            api_kwargs.update(self._max_tokens_param(_ephemeral_out))
        elif self.max_tokens is not None:
            api_kwargs.update(self._max_tokens_param(self.max_tokens))
        elif "integrate.api.nvidia.com" in self._base_url_lower:
            # NVIDIA NIM 在省略时默认为非常低的 max_tokens，
            # 导致像 GLM-4.7 这样的模型立即截断（仅思考
            # 令牌就耗尽了预算）。16384 提供了足够的空间。
            api_kwargs.update(self._max_tokens_param(16384))
        elif self._is_qwen_portal():
            # Qwen Portal 在省略时默认为非常低的 max_tokens。
            # 推理模型（qwen3-coder-plus）仅凭思考令牌就耗尽了该预算，
            # 导致门户返回 finish_reason="stop" 和截断的输出 — 代理
            # 将此视为有意停止并退出循环。发送 65536
            #（qwen3-coder 模型的文档化最大输出）以便
            # 模型有足够的输出预算进行工具调用。
            api_kwargs.update(self._max_tokens_param(65536))
        elif (self._is_openrouter_url() or "nousresearch" in self._base_url_lower) and "claude" in (self.model or "").lower():
            # OpenRouter 和 Nous Portal 将请求转换为 Anthropic 的
            # Messages API，这需要 max_tokens 作为必填字段。
            # 当我们省略它时，代理选择一个可能太低的默认值 — 
            # 模型将其输出预算花费在思考上，几乎没有什么留给
            # 实际响应（尤其是像 write_file 这样的大型工具调用）。
            # 发送模型的实际输出限制确保满容量。
            try:
                from agent.anthropic_adapter import _get_anthropic_max_output
                _model_output_limit = _get_anthropic_max_output(self.model)
                api_kwargs["max_tokens"] = _model_output_limit
            except Exception:
                pass  # fail open — let the proxy pick its default

        extra_body = {}

        _is_openrouter = self._is_openrouter_url()
        _is_github_models = (
            "models.github.ai" in self._base_url_lower
            or "api.githubcopilot.com" in self._base_url_lower
        )

        # 提供商偏好（only、ignore、order、sort）是 OpenRouter-
        # 特定的。仅发送到 OpenAI 兼容端点。
        # TODO：Nous Portal 将添加透明代理支持 — 在其
        # 后端更新时为 _is_nous 重新启用。
        if provider_preferences and _is_openrouter:
            extra_body["provider"] = provider_preferences
        _is_nous = "nousresearch" in self._base_url_lower

        if self._supports_reasoning_extra_body():
            if _is_github_models:
                github_reasoning = self._github_models_reasoning_extra_body()
                if github_reasoning is not None:
                    extra_body["reasoning"] = github_reasoning
            else:
                if self.reasoning_config is not None:
                    rc = dict(self.reasoning_config)
                    # Nous Portal 需要启用推理 — 不要向其发送
                    # enabled=false（会导致 400）。
                    if _is_nous and rc.get("enabled") is False:
                        pass  # omit reasoning entirely for Nous when disabled
                    else:
                        extra_body["reasoning"] = rc
                else:
                    extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }

        # Nous Portal 产品归属
        if _is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        # Ollama num_ctx：覆盖 2048 默认值，以便模型实际
        # 使用其训练的上下文窗口。通过 OpenAI
        # SDK 的 extra_body → options.num_ctx 传递，Ollama 的 OpenAI 兼容
        # 端点将其作为 --ctx-size 转发给运行器。
        if self._ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = self._ollama_num_ctx
            extra_body["options"] = options

        # Ollama / 自定义提供商：在推理禁用时传递 think=false。
        # Ollama 不识别 OpenRouter 风格的 `reasoning` extra_body
        # 字段，因此我们改用其原生 `think` 参数。
        # 这防止具有思考能力的模型（Qwen3 等）在用户
        # 设置了 reasoning_effort: none 时生成 <think> 块并产生空响应错误。
        if self.provider == "custom" and self.reasoning_config and isinstance(self.reasoning_config, dict):
            _effort = (self.reasoning_config.get("effort") or "").strip().lower()
            _enabled = self.reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                extra_body["think"] = False

        if self._is_qwen_portal():
            extra_body["vl_high_resolution_images"] = True

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        # 优先处理 / 通用请求覆盖（例如 service_tier）。
        # 最后应用，以便覆盖胜过上面设置的任何默认值。
        if self.request_overrides:
            api_kwargs.update(self.request_overrides)

        return api_kwargs

    def _supports_reasoning_extra_body(self) -> bool:
        """当推理 extra_body 对此路由/模型安全发送时返回 True。

        OpenRouter 将未知的 extra_body 字段转发给上游提供商。
        一些提供商/路由以 400 拒绝 `reasoning`，因此将其限制到
        已知的推理能力模型系列和直接 Nous Portal。
        """
        if "nousresearch" in self._base_url_lower:
            return True
        if "ai-gateway.vercel.sh" in self._base_url_lower:
            return True
        if "models.github.ai" in self._base_url_lower or "api.githubcopilot.com" in self._base_url_lower:
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if "openrouter" not in self._base_url_lower:
            return False
        if "api.mistral.ai" in self._base_url_lower:
            return False

        model = (self.model or "").lower()
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "qwen/qwen3",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """为 GitHub Models/OpenAI 兼容路由格式化推理有效载荷。"""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = str(
                self.reasoning_config.get("effort", "medium")
            ).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}

    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """从 API 响应消息构建标准化的助手消息字典。

        处理推理提取、reasoning_details 和可选的 tool_calls，
        以便工具调用路径和最终响应路径共享一个构建器。
        """
        reasoning_text = self._extract_reasoning(assistant_message)
        _from_structured = bool(reasoning_text)

        # Fallback: extract inline <think> blocks from content when no structured
        # reasoning fields are present (some models/providers embed thinking
        # directly in the content rather than returning separate API fields).
        if not reasoning_text:
            content = assistant_message.content or ""
            think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            if think_blocks:
                combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
                reasoning_text = combined or None

        if reasoning_text and self.verbose_logging:
            logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

        if reasoning_text and self.reasoning_callback:
            # Skip callback when streaming is active — reasoning was already
            # displayed during the stream via one of two paths:
            #   (a) _fire_reasoning_delta (structured reasoning_content deltas)
            #   (b) _stream_delta tag extraction (<think>/<REASONING_SCRATCHPAD>)
            # When streaming is NOT active, always fire so non-streaming modes
            # (gateway, batch, quiet) still get reasoning.
            # Any reasoning that wasn't shown during streaming is caught by the
            # CLI post-response display fallback (cli.py _reasoning_shown_this_turn).
            if not self.stream_delta_callback and not self._stream_callback:
                try:
                    self.reasoning_callback(reasoning_text)
                except Exception:
                    pass

        # 清理 API 响应中的代理项 — 某些模型（例如通过 Ollama 的 Kimi/GLM）
        # 可能返回无效的代理码点，导致持久化时 json.dumps() 崩溃。
        _raw_content = assistant_message.content or ""
        _san_content = _sanitize_surrogates(_raw_content)
        if reasoning_text:
            reasoning_text = _sanitize_surrogates(reasoning_text)

        # Strip inline reasoning tags (<think>…</think> etc.) from the stored
        # assistant content.  Reasoning was already captured into
        # ``reasoning_text`` above (either from structured fields or the
        # inline-block fallback), so the raw tags in content are redundant.
        # Leaving them in place caused reasoning to leak to messaging
        # platforms (#8878, #9568), inflate context on subsequent turns
        # (#9306 observed 16% content-size reduction on a real MiniMax
        # session), and pollute generated session titles.  One strip at the
        # storage boundary cleans content for every downstream consumer:
        # API replay, session transcript, gateway delivery, CLI display,
        # compression, title generation.
        if isinstance(_san_content, str) and _san_content:
            _san_content = self._strip_think_blocks(_san_content).strip()

        msg = {
            "role": "assistant",
            "content": _san_content,
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            # 未经修改地传回 reasoning_details，以便提供商（OpenRouter、
            # Anthropic、OpenAI）可以在回合之间保持推理连续性。
            # 每个提供商可能包含必须完全保留的不透明字段
            # （signature、encrypted_content）。
            raw_details = assistant_message.reasoning_details
            preserved = []
            for d in raw_details:
                if isinstance(d, dict):
                    preserved.append(d)
                elif hasattr(d, "__dict__"):
                    preserved.append(d.__dict__)
                elif hasattr(d, "model_dump"):
                    preserved.append(d.model_dump())
            if preserved:
                msg["reasoning_details"] = preserved

        # Codex Responses API：保留加密推理项以实现
        # 多回合连续性。这些在下一回合作为输入重放。
        codex_items = getattr(assistant_message, "codex_reasoning_items", None)
        if codex_items:
            msg["codex_reasoning_items"] = codex_items

        if assistant_message.tool_calls:
            tool_calls = []
            for tool_call in assistant_message.tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        _fn = getattr(tool_call, "function", None)
                        _fn_name = getattr(_fn, "name", "") if _fn else ""
                        _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                        call_id = self._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
                call_id = call_id.strip()

                response_item_id = getattr(tool_call, "response_item_id", None)
                if not isinstance(response_item_id, str) or not response_item_id.strip():
                    _, embedded_response_item_id = self._split_responses_tool_id(raw_id)
                    response_item_id = embedded_response_item_id

                response_item_id = self._derive_responses_function_call_id(
                    call_id,
                    response_item_id if isinstance(response_item_id, str) else None,
                )

                tc_dict = {
                    "id": call_id,
                    "call_id": call_id,
                    "response_item_id": response_item_id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                }
                # 保留 extra_content（例如 Gemini thought_signature），以便
                # 它在后续 API 调用中发回。没有这个，Gemini 3
                # 思考模型会以 400 错误拒绝请求。
                extra = getattr(tool_call, "extra_content", None)
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        extra = extra.model_dump()
                    tc_dict["extra_content"] = extra
                tool_calls.append(tc_dict)
            msg["tool_calls"] = tool_calls

        return msg

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """为严格提供商从 tool_calls 中剥离 Codex Responses API 字段。

        像 Mistral、Fireworks 和其他严格的 OpenAI 兼容 API
        这样的提供商验证 Chat Completions 模式并以 400 或 422 错误
        拒绝未知字段（call_id、response_item_id）。这些字段保留在
        内部消息历史中 — 此方法仅修改传出的
        API 副本。

        创建新的 tool_call 字典而不是就地变异，以便
        原始消息列表为 Codex Responses API 兼容性保留
        call_id/response_item_id（例如，如果会话稍后回退到
        Codex 提供商）。

        剥离的字段：call_id、response_item_id
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        _STRIP_KEYS = {"call_id", "response_item_id"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg

    def _should_sanitize_tool_calls(self) -> bool:
        """确定 tool_calls 是否需要为严格 API 进行清理。

        Codex Responses API 使用像 call_id 和 response_item_id
        这样的字段，这些字段不是标准 Chat Completions 模式的一部分。
        调用任何其他 API 时必须剥离这些字段以避免
        验证错误（400 Bad Request）。

        返回：
            bool：如果需要清理则为 True（非 Codex API），否则为 False。
        """
        return self.api_mode != "codex_responses"

    def flush_memories(self, messages: list = None, min_turns: int = None):
        """在上下文丢失之前给模型一个回合来持久化内存。

        在压缩、会话重置或 CLI 退出之前调用。注入刷新消息，
        进行一次 API 调用，执行任何内存工具调用，然后
        从消息列表中剥离所有刷新产物。

        参数：
            messages：当前对话消息。如果为 None，则使用
                      self._session_messages（最后一次 run_conversation 状态）。
            min_turns：触发刷新所需的最少用户回合数。
                       None = 使用配置值（flush_min_turns）。
                       0 = 始终刷新（用于压缩）。
        """
        if self._memory_flush_min_turns == 0 and min_turns is None:
            return
        if "memory" not in self.valid_tool_names or not self._memory_store:
            return
        effective_min = min_turns if min_turns is not None else self._memory_flush_min_turns
        if self._user_turn_count < effective_min:
            return

        if messages is None:
            messages = getattr(self, '_session_messages', None)
        if not messages or len(messages) < 3:
            return

        flush_content = (
            "[System: The session is being compressed. "
            "Save anything worth remembering — prioritize user preferences, "
            "corrections, and recurring patterns over task-specific details.]"
        )
        _sentinel = f"__flush_{id(self)}_{time.monotonic()}"
        flush_msg = {"role": "user", "content": flush_content, "_flush_sentinel": _sentinel}
        messages.append(flush_msg)

        try:
            # 为刷新调用构建 API 消息
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                if msg.get("role") == "assistant":
                    reasoning = msg.get("reasoning")
                    if reasoning:
                        api_msg["reasoning_content"] = reasoning
                api_msg.pop("reasoning", None)
                api_msg.pop("finish_reason", None)
                api_msg.pop("_flush_sentinel", None)
                api_msg.pop("_thinking_prefill", None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            if self._cached_system_prompt:
                api_messages = [{"role": "system", "content": self._cached_system_prompt}] + api_messages

            # 仅使用内存工具进行一次 API 调用
            memory_tool_def = None
            for t in (self.tools or []):
                if t.get("function", {}).get("name") == "memory":
                    memory_tool_def = t
                    break

            if not memory_tool_def:
                messages.pop()  # remove flush msg
                return

            # 在可用时使用辅助客户端进行刷新调用 —
            # 它更便宜并避免 Codex Responses API 不兼容性。
            from agent.auxiliary_client import (
                call_llm as _call_llm,
                _fixed_temperature_for_model,
            )
            _aux_available = True
            # 如果模型有严格合同，使用固定温度覆盖（例如 kimi-for-coding → 0.6）；
            # 否则使用历史 0.3 默认值。
            _flush_temperature = _fixed_temperature_for_model(self.model)
            if _flush_temperature is None:
                _flush_temperature = 0.3
            try:
                response = _call_llm(
                    task="flush_memories",
                    messages=api_messages,
                    tools=[memory_tool_def],
                    temperature=_flush_temperature,
                    max_tokens=5120,
                    # timeout resolved from auxiliary.flush_memories.timeout config
                )
            except RuntimeError:
                _aux_available = False
                response = None

            if not _aux_available and self.api_mode == "codex_responses":
                # 没有辅助客户端 — 直接使用 Codex Responses 路径
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs["tools"] = self._responses_tools([memory_tool_def])
                codex_kwargs["temperature"] = _flush_temperature
                if "max_output_tokens" in codex_kwargs:
                    codex_kwargs["max_output_tokens"] = 5120
                response = self._run_codex_stream(codex_kwargs)
            elif not _aux_available and self.api_mode == "anthropic_messages":
                # 原生 Anthropic — 直接使用 Anthropic 客户端
                from agent.anthropic_adapter import build_anthropic_kwargs as _build_ant_kwargs
                ant_kwargs = _build_ant_kwargs(
                    model=self.model, messages=api_messages,
                    tools=[memory_tool_def], max_tokens=5120,
                    reasoning_config=None,
                    preserve_dots=self._anthropic_preserve_dots(),
                )
                response = self._anthropic_messages_create(ant_kwargs)
            elif not _aux_available:
                api_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    "tools": [memory_tool_def],
                    "temperature": _flush_temperature,
                    **self._max_tokens_param(5120),
                }
                from agent.auxiliary_client import _get_task_timeout
                response = self._ensure_primary_openai_client(reason="flush_memories").chat.completions.create(
                    **api_kwargs, timeout=_get_task_timeout("flush_memories")
                )

            # 从响应中提取工具调用，处理所有 API 格式
            tool_calls = []
            if self.api_mode == "codex_responses" and not _aux_available:
                assistant_msg, _ = self._normalize_codex_response(response)
                if assistant_msg and assistant_msg.tool_calls:
                    tool_calls = assistant_msg.tool_calls
            elif self.api_mode == "anthropic_messages" and not _aux_available:
                from agent.anthropic_adapter import normalize_anthropic_response as _nar_flush
                _flush_msg, _ = _nar_flush(response, strip_tool_prefix=self._is_anthropic_oauth)
                if _flush_msg and _flush_msg.tool_calls:
                    tool_calls = _flush_msg.tool_calls
            elif hasattr(response, "choices") and response.choices:
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    tool_calls = assistant_message.tool_calls

            for tc in tool_calls:
                if tc.function.name == "memory":
                    try:
                        args = json.loads(tc.function.arguments)
                        flush_target = args.get("target", "memory")
                        from tools.memory_tool import memory_tool as _memory_tool
                        _memory_tool(
                            action=args.get("action"),
                            target=flush_target,
                            content=args.get("content"),
                            old_text=args.get("old_text"),
                            store=self._memory_store,
                        )
                        if not self.quiet_mode:
                            print(f"  🧠 Memory flush: saved to {args.get('target', 'memory')}")
                    except Exception as e:
                        logger.debug("Memory flush tool call failed: %s", e)
        except Exception as e:
            logger.debug("Memory flush API call failed: %s", e)
        finally:
            # 剥离刷新产物：从刷新消息开始删除所有内容。
            # 使用哨兵标记而不是身份检查以实现鲁棒性。
            while messages and messages[-1].get("_flush_sentinel") != _sentinel:
                messages.pop()
                if not messages:
                    break
            if messages and messages[-1].get("_flush_sentinel") == _sentinel:
                messages.pop()

    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default", focus_topic: str = None) -> tuple:
        """压缩对话上下文并在 SQLite 中拆分会话。

        参数：
            focus_topic：用于引导压缩的可选焦点字符串 — 
                总结器将优先保留与此主题相关的信息。
                受 Claude Code 的 ``/compact <focus>`` 启发。

        返回：
            (compressed_messages, new_system_prompt) 元组
        """
        _pre_msg_count = len(messages)
        logger.info(
            "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
            self.session_id or "none", _pre_msg_count,
            f"{approx_tokens:,}" if approx_tokens else "unknown", self.model,
            focus_topic,
        )
        # 压缩前内存刷新：让模型在内存丢失前保存它们
        self.flush_memories(messages, min_turns=0)

        # 在压缩丢弃上下文之前通知外部内存提供商
        if self._memory_manager:
            try:
                self._memory_manager.on_pre_compress(messages)
            except Exception:
                pass

        compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens, focus_topic=focus_topic)

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # 将标题传播到新会话并自动编号
                old_title = self._session_db.get_session_title(self.session_id)
                # 在旧会话轮换之前触发内存提取。
                self.commit_memory_session(messages)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                # Update session_log_file to point to the new session's JSON file
                self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    parent_session_id=old_session_id,
                )
                # 为继续会话自动编号标题
                if old_title:
                    try:
                        new_title = self._session_db.get_next_title_in_lineage(old_title)
                        self._session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                self._session_db.update_system_prompt(self.session_id, new_system_prompt)
                # 重置刷新游标 — 新会话开始时没有写入消息
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # 警告重复压缩（每次传递质量下降）
        _cc = self.context_compressor.compression_count
        if _cc >= 2:
            self._vprint(
                f"{self.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh.",
                force=True,
            )

        # 压缩后更新令牌估计，以便压力计算
        # 使用压缩后计数，而不是陈旧的压缩前计数。
        _compressed_est = (
            estimate_tokens_rough(new_system_prompt)
            + estimate_messages_tokens_rough(compressed)
        )
        self.context_compressor.last_prompt_tokens = _compressed_est
        self.context_compressor.last_completion_tokens = 0

        # 清除文件读取去重缓存。压缩后原始
        # 读取内容被总结掉 — 如果模型重新读取相同
        # 的文件，它需要完整内容，而不是"文件未更改"存根。
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d tokens=~%s",
            self.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        return compressed, new_system_prompt

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """从助手消息执行工具调用并将结果附加到消息。

        仅调度看起来独立的批次进行并发执行：
        只读工具可能始终共享并行路径，而
        file reads/writes may do so only when their target paths do not overlap.
        """
        tool_calls = assistant_message.tool_calls

        # 即使有流式消费者也允许在工具执行期间使用 _vprint
        self._executing_tools = True
        try:
            if not _should_parallelize_tool_batch(tool_calls):
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            return self._execute_tool_calls_concurrent(
                assistant_message, messages, effective_task_id, api_call_count
            )
        finally:
            self._executing_tools = False

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None) -> str:
        """调用单个工具并返回结果字符串。无显示逻辑。

        处理代理级工具（todo、memory 等）和注册表分发的
        工具。由并发执行路径使用；顺序路径保留
        其自己的内联调用以实现向后兼容的显示处理。
        """
        # 在执行任何操作之前检查插件钩子的阻止指令。
        block_message: Optional[str] = None
        try:
            from hermes_cli.plugins import get_pre_tool_call_block_message
            block_message = get_pre_tool_call_block_message(
                function_name, function_args, task_id=effective_task_id or "",
            )
        except Exception:
            pass
        if block_message is not None:
            return json.dumps({"error": block_message}, ensure_ascii=False)

        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            if not self._session_db:
                return json.dumps({"success": False, "error": "Session database not available."})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=self._session_db,
                current_session_id=self.session_id,
            )
        elif function_name == "memory":
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=function_args.get("action"),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=self._memory_store,
            )
            # Bridge: notify external memory provider of built-in memory writes
            if self._memory_manager and function_args.get("action") in ("add", "replace"):
                try:
                    self._memory_manager.on_memory_write(
                        function_args.get("action", ""),
                        target,
                        function_args.get("content", ""),
                    )
                except Exception:
                    pass
            return result
        elif self._memory_manager and self._memory_manager.has_tool(function_name):
            return self._memory_manager.handle_tool_call(function_name, function_args)
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            from tools.delegate_tool import delegate_task as _delegate_task
            return _delegate_task(
                goal=function_args.get("goal"),
                context=function_args.get("context"),
                toolsets=function_args.get("toolsets"),
                tasks=function_args.get("tasks"),
                max_iterations=function_args.get("max_iterations"),
                parent_agent=self,
            )
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                tool_call_id=tool_call_id,
                session_id=self.session_id or "",
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                skip_pre_tool_call_hook=True,
            )

    @staticmethod
    def _wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
        """将详细工具输出自动换行以适应终端宽度。

        在现有换行符上分割 *text* 并单独包装每一行，
        保留有意换行符（例如漂亮打印的 JSON）。
        返回准备打印的字符串，第一行有 *label*，
        续行缩进。
        """
        import shutil as _shutil
        import textwrap as _tw
        cols = _shutil.get_terminal_size((120, 24)).columns
        wrap_width = max(40, cols - len(indent))
        out_lines: list[str] = []
        for raw_line in text.split("\n"):
            if len(raw_line) <= wrap_width:
                out_lines.append(raw_line)
            else:
                wrapped = _tw.wrap(raw_line, width=wrap_width,
                                   break_long_words=True,
                                   break_on_hyphens=False)
                out_lines.extend(wrapped or [raw_line])
        body = ("\n" + indent).join(out_lines)
        return f"{indent}{label}{body}"

    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """使用线程池并发执行多个工具调用。

        结果按原始工具调用顺序收集并附加到
        消息，以便 API 按预期顺序看到它们。
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)

        # ── 预检：中断检查 ──────────────────────────────────
        if self._interrupt_requested:
            print(f"{self.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "content": f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                    "tool_call_id": tc.id,
                })
            return

        # ── 解析参数 + 执行前簿记 ───────────────────────
        parsed_calls = []  # list of (tool_call, function_name, function_args)
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # Reset nudge counters
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self.skill_evolution.on_tool_invoked(function_name)

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # 文件变异工具的检查点
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    pass

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

            parsed_calls.append((tool_call, function_name, function_args))

        # ── 日志记录 / 回调 ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _ in parsed_calls)
        if not self.quiet_mode:
            print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    print(self._wrap_verbose("Args: ", json.dumps(args, indent=2, ensure_ascii=False)))
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

        for tc, name, args in parsed_calls:
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback("tool.started", name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        for tc, name, args in parsed_calls:
            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tc.id, name, args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

        # ── 并发执行 ─────────────────────────────────────────
        # 每个槽保存 (function_name, function_args, function_result, duration, error_flag)
        results = [None] * num_tools

        # 在启动工作线程之前触摸活动，以便网关知道
        # 我们正在执行工具（而不是卡住）。
        self._current_tool = tool_names_str
        self._touch_activity(f"executing {num_tools} tools concurrently: {tool_names_str}")

        def _run_tool(index, tool_call, function_name, function_args):
            """在线程中执行的工作函数。"""
            # 注册此工作线程 tid，以便代理可以向其
            # 扩散中断 — 见 AIAgent.interrupt()。
            # 必须首先发生，并且必须与 finally 块中的 discard + clear 配对。
            _worker_tid = threading.current_thread().ident
            with self._tool_worker_threads_lock:
                self._tool_worker_threads.add(_worker_tid)
            # 竞争：如果代理在扩散（快照了空/早期集合）
            # 和我们的注册之间被中断，现在将中断
            # 应用到我们自己的 tid，以便工具内的 is_interrupted()
            # 在下一次轮询时返回 True。
            if self._interrupt_requested:
                try:
                    from tools.interrupt import set_interrupt as _sif
                    _sif(True, _worker_tid)
                except Exception:
                    pass
            # 在此工作线程上设置活动回调，以便
            # _wait_for_process（终端命令）可以触发心跳。
            # 回调是线程本地的；主线程的回调
            # 对工作线程不可见。
            try:
                from tools.environments.base import set_activity_callback
                set_activity_callback(self._touch_activity)
            except Exception:
                pass
            start = time.time()
            try:
                result = self._invoke_tool(function_name, function_args, effective_task_id, tool_call.id)
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (function_name, function_args, result, duration, is_error)
            # 拆除工作线程 tid 跟踪。清除我们可能
            # 设置的任何中断位，以便调度到此回收 tid 的
            # 下一个任务以干净的状态开始。
            with self._tool_worker_threads_lock:
                self._tool_worker_threads.discard(_worker_tid)
            try:
                from tools.interrupt import set_interrupt as _sif
                _sif(False, _worker_tid)
            except Exception:
                pass

        # 为 CLI 模式启动旋转器（TUI 处理工具进度时跳过）
        spinner = None
        if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
            face = random.choice(KawaiiSpinner.get_waiting_faces())
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=self._print_fn)
            spinner.start()

        try:
            max_workers = min(num_tools, _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, (tc, name, args) in enumerate(parsed_calls):
                    f = executor.submit(_run_tool, i, tc, name, args)
                    futures.append(f)

                # 等待所有完成并定期心跳，以便
                # 网关的非活动监视器在长时间
                # 并发工具批次期间不会杀死我们。
                # 同时检查用户中断，
                # 以便当用户在并发工具执行期间发送 /stop
                # 或新消息时我们不会无限期阻塞。
                _conc_start = time.time()
                _interrupt_logged = False
                while True:
                    done, not_done = concurrent.futures.wait(
                        futures, timeout=5.0,
                    )
                    if not not_done:
                        break

                    # 检查中断 — 每线程中断信号
                    # 已经导致单个工具（terminal、execute_code）
                    # 中止，但没有中断检查的工具（web_search、
                    # read_file）将运行到完成。取消任何尚未
                    # 开始的期货，这样我们就不会阻塞它们。
                    if self._interrupt_requested:
                        if not _interrupt_logged:
                            _interrupt_logged = True
                            self._vprint(
                                f"{self.log_prefix}⚡ Interrupt: cancelling "
                                f"{len(not_done)} pending concurrent tool(s)",
                                force=True,
                            )
                        for f in not_done:
                            f.cancel()
                        # 给已经运行的工具一点时间来注意
                        # 每线程中断信号并优雅退出。
                        concurrent.futures.wait(not_done, timeout=3.0)
                        break

                    _conc_elapsed = int(time.time() - _conc_start)
                    # 每 ~30s 心跳一次（6 × 5s 轮询间隔）
                    if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                        _still_running = [
                            parsed_calls[futures.index(f)][1]
                            for f in not_done
                            if f in futures
                        ]
                        self._touch_activity(
                            f"concurrent tools running ({_conc_elapsed}s, "
                            f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                        )
        finally:
            if spinner:
                # 为旋转器停止构建摘要消息
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

        # ── 执行后：显示每个工具的结果 ─────────────────────
        for i, (tc, name, args) in enumerate(parsed_calls):
            r = results[i]
            if r is None:
                # 工具被取消（中断）或线程未返回
                if self._interrupt_requested:
                    function_result = f"[Tool execution cancelled — {name} was skipped due to user interrupt]"
                else:
                    function_result = f"Error executing tool '{name}': thread did not return a result"
                tool_duration = 0.0
            else:
                function_name, function_args, function_result, tool_duration, is_error = r

                if is_error:
                    result_preview = function_result[:200] if len(function_result) > 200 else function_result
                    logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

                if self.tool_progress_callback:
                    try:
                        self.tool_progress_callback(
                            "tool.completed", function_name, None, None,
                            duration=tool_duration, is_error=is_error,
                        )
                    except Exception as cb_err:
                        logging.debug(f"Tool progress callback error: {cb_err}")

                if self.verbose_logging:
                    logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # 为每个工具打印可爱消息
            if self._should_emit_quiet_tool_messages():
                cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
                self._safe_print(f"  {cute_msg}")
            elif not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                    print(self._wrap_verbose("Result: ", function_result))
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tc.id, name, args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=name,
                tool_use_id=tc.id,
                env=get_active_env(effective_task_id),
            )

            subdir_hints = self._subdirectory_hints.check_tool_call(name, args)
            if subdir_hints:
                function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools = len(parsed_calls)
        if num_tools > 0:
            turn_tool_msgs = messages[-num_tools:]
            enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id))

        # ── /steer 注入 ──────────────────────────────────────────────
        # 将任何待处理的用户转向文本附加到最后一个工具结果，以便
        # 代理在其下一次迭代中看到它。在预算执行之后运行，
        # 因此转向标记永远不会被截断。详情见 steer()。
        if num_tools > 0:
            self._apply_pending_steer_to_tool_results(messages, num_tools)

    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """顺序执行工具调用（原始行为）。用于单个调用或交互式工具。"""
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            # 安全：在开始每个工具之前检查中断。
            # 如果用户在先前工具的执行期间发送了"stop"，
            # 则不要启动任何更多工具 — 立即跳过它们。
            if self._interrupt_requested:
                remaining_calls = assistant_message.tool_calls[i-1:]
                if remaining_calls:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
                for skipped_tc in remaining_calls:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Check plugin hooks for a block directive before executing.
            _block_msg: Optional[str] = None
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                _block_msg = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                pass

            if _block_msg is not None:
                # 工具被插件策略阻止 — 跳过计数器重置。
                # 执行在下面的工具调度链中处理。
                pass
            else:
                # 当相关工具实际使用时重置提醒计数器
                if function_name == "memory":
                    self._turns_since_memory = 0
                elif function_name == "skill_manage":
                    self.skill_evolution.on_tool_invoked(function_name)

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                    print(self._wrap_verbose("Args: ", json.dumps(function_args, indent=2, ensure_ascii=False)))
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

            if _block_msg is None:
                self._current_tool = function_name
                self._touch_activity(f"executing tool: {function_name}")

            # 为长时间运行的工具执行（终端
            # 命令等）设置活动回调，以便网关的非活动监视器
            # 在命令运行时不会杀死代理。
            if _block_msg is None:
                try:
                    from tools.environments.base import set_activity_callback
                    set_activity_callback(self._touch_activity)
                except Exception:
                    pass

            if _block_msg is None and self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback("tool.started", function_name, preview, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if _block_msg is None and self.tool_start_callback:
                try:
                    self.tool_start_callback(tool_call.id, function_name, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

            # 检查点：在文件变异工具之前快照工作目录
            if _block_msg is None and function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(
                            work_dir, f"before {function_name}"
                        )
                except Exception:
                    pass  # never block tool execution

            # Checkpoint before destructive terminal commands
            if _block_msg is None and function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass  # never block tool execution

            tool_start_time = time.time()

            if _block_msg is not None:
                # 工具被插件策略阻止 — 返回错误而不执行。
                function_result = json.dumps({"error": _block_msg}, ensure_ascii=False)
                tool_duration = 0.0
            elif function_name == "todo":
                from tools.todo_tool import todo_tool as _todo_tool
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            elif function_name == "session_search":
                if not self._session_db:
                    function_result = json.dumps({"success": False, "error": "Session database not available."})
                else:
                    from tools.session_search_tool import session_search as _session_search
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter"),
                        limit=function_args.get("limit", 3),
                        db=self._session_db,
                        current_session_id=self.session_id,
                    )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
            elif function_name == "memory":
                target = function_args.get("target", "memory")
                from tools.memory_tool import memory_tool as _memory_tool
                function_result = _memory_tool(
                    action=function_args.get("action"),
                    target=target,
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    store=self._memory_store,
                )
                # 桥接：通知外部内存提供商内置内存写入
                if self._memory_manager and function_args.get("action") in ("add", "replace"):
                    try:
                        self._memory_manager.on_memory_write(
                            function_args.get("action", ""),
                            target,
                            function_args.get("content", ""),
                        )
                    except Exception:
                        pass
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            elif function_name == "clarify":
                from tools.clarify_tool import clarify_tool as _clarify_tool
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            elif function_name == "delegate_task":
                from tools.delegate_tool import delegate_task as _delegate_task
                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                self._delegate_spinner = spinner
                _delegate_result = None
                try:
                    function_result = _delegate_task(
                        goal=function_args.get("goal"),
                        context=function_args.get("context"),
                        toolsets=function_args.get("toolsets"),
                        tasks=tasks_arg,
                        max_iterations=function_args.get("max_iterations"),
                        parent_agent=self,
                    )
                    _delegate_result = function_result
                finally:
                    self._delegate_spinner = None
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._context_engine_tool_names and function_name in self._context_engine_tool_names:
                # 上下文引擎工具（lcm_grep、lcm_describe、lcm_expand 等）
                spinner = None
                if self._should_emit_quiet_tool_messages():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _ce_result = None
                try:
                    function_result = self.context_compressor.handle_tool_call(function_name, function_args, messages=messages)
                    _ce_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                    logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_ce_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._memory_manager and self._memory_manager.has_tool(function_name):
                # 内存提供商工具（hindsight_retain、honcho_search 等）
                # 这些不在工具注册表中 — 通过 MemoryManager 路由。
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _mem_result = None
                try:
                    function_result = self._memory_manager.handle_tool_call(function_name, function_args)
                    _mem_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                    logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self.quiet_mode:
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _spinner_result = None
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        skip_pre_tool_call_hook=True,
                    )
                    _spinner_result = function_result
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            else:
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        skip_pre_tool_call_hook=True,
                    )
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                tool_duration = time.time() - tool_start_time

            result_preview = function_result if self.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )

            # 将工具错误记录到持久错误日志，以便 UI 中的 [error] 标签
            # 始终在磁盘上有相应的详细条目。
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if _is_error_result:
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, len(function_result))

            if self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed", function_name, None, None,
                        duration=tool_duration, is_error=_is_error_result,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tool_call.id, function_name, function_args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=function_name,
                tool_use_id=tool_call.id,
                env=get_active_env(effective_task_id),
            )

            # 从工具参数发现子目录上下文文件
            subdir_hints = self._subdirectory_hints.check_tool_call(function_name, function_args)
            if subdir_hints:
                function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(self._wrap_verbose("Result: ", function_result))
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(self.tool_delay)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools_seq = len(assistant_message.tool_calls)
        if num_tools_seq > 0:
            enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id))

        # ── /steer 注入 ──────────────────────────────────────────────
        # 基本原理见 _execute_tool_calls_parallel。相同的钩子，
        # 也应用于顺序执行。
        if num_tools_seq > 0:
            self._apply_pending_steer_to_tool_results(messages, num_tools_seq)



    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """当达到最大迭代次数时请求摘要。返回最终响应文本。"""
        print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary...")

        summary_request = (
            "You've reached the maximum number of tool-calling iterations allowed. "
            "Please provide a final response summarizing what you've found and accomplished so far, "
            "without calling any more tools."
        )
        messages.append({"role": "user", "content": summary_request})

        try:
            # 构建 API 消息，剥离仅内部字段
            # （finish_reason、reasoning），严格 API 如 Mistral 以 422 拒绝
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                    api_msg.pop(internal_field, None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            effective_system = self._cached_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            summary_extra_body = {}
            try:
                from agent.auxiliary_client import _fixed_temperature_for_model
            except Exception:
                _fixed_temperature_for_model = None
            _summary_temperature = (
                _fixed_temperature_for_model(self.model)
                if _fixed_temperature_for_model is not None
                else None
            )
            _is_nous = "nousresearch" in self._base_url_lower
            if self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                assistant_message, _ = self._normalize_codex_response(summary_response)
                final_response = (assistant_message.content or "").strip() if assistant_message else ""
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if _summary_temperature is not None:
                    summary_kwargs["temperature"] = _summary_temperature
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))

                # Include provider routing preferences
                provider_preferences = {}
                if self.providers_allowed:
                    provider_preferences["only"] = self.providers_allowed
                if self.providers_ignored:
                    provider_preferences["ignore"] = self.providers_ignored
                if self.providers_order:
                    provider_preferences["order"] = self.providers_order
                if self.provider_sort:
                    provider_preferences["sort"] = self.provider_sort
                if provider_preferences:
                    summary_extra_body["provider"] = provider_preferences

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak, normalize_anthropic_response as _nar
                    _ant_kw = _bak(model=self.model, messages=api_messages, tools=None,
                                   max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                   is_oauth=self._is_anthropic_oauth,
                                   preserve_dots=self._anthropic_preserve_dots())
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _msg, _ = _nar(summary_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_msg.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                # 重试摘要生成
                if self.api_mode == "codex_responses":
                    codex_kwargs = self._build_api_kwargs(api_messages)
                    codex_kwargs.pop("tools", None)
                    retry_response = self._run_codex_stream(codex_kwargs)
                    retry_msg, _ = self._normalize_codex_response(retry_response)
                    final_response = (retry_msg.content or "").strip() if retry_msg else ""
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak2, normalize_anthropic_response as _nar2
                    _ant_kw2 = _bak2(model=self.model, messages=api_messages, tools=None,
                                    is_oauth=self._is_anthropic_oauth,
                                    max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                    preserve_dots=self._anthropic_preserve_dots())
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_msg, _ = _nar2(retry_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_retry_msg.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if _summary_temperature is not None:
                        summary_kwargs["temperature"] = _summary_temperature
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

                if final_response:
                    if "<think>" in final_response:
                        final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                    if final_response:
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        final_response = "I reached the iteration limit and couldn't generate a summary."
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."

        except Exception as e:
            logging.warning(f"Failed to get summary response: {e}")
            final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"

        return final_response

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行完整的工具调用对话直到完成。

        参数：
            user_message (str)：用户的消息/问题
            system_message (str)：自定义系统消息（可选，如果提供则覆盖 ephemeral_system_prompt）
            conversation_history (List[Dict])：先前的对话消息（可选）
            task_id (str)：此任务的唯一标识符，用于在并发任务之间隔离 VM（可选，如果未提供则自动生成）
            stream_callback：在流式传输期间使用每个文本增量调用的可选回调。
                TTS 管道使用它在完整响应之前开始音频生成。
                当为 None（默认）时，API 调用使用标准非流式路径。
            persist_user_message：可选的干净用户消息，用于在
                user_message 包含仅 API 的
                合成前缀时存储在记录/历史中。
                    或排队后续预取工作。

        返回：
            Dict：完整的对话结果，包含最终响应和消息历史
        """
        # 保护 stdio 免受管道破裂的 OSError（systemd/headless/daemon）。
        # 安装一次，流健康时透明，防止写入时崩溃。
        _install_safe_stdio()

        # 在此线程上使用会话 ID 标记所有日志记录，以便
        # ``hermes logs --session <id>`` 可以过滤单个对话。
        from hermes_logging import set_session_context
        set_session_context(self.session_id)

        # 如果上一回合激活了回退，恢复主
        # 运行时，以便此回合使用首选模型进行全新尝试。
        # 当 _fallback_activated 为 False 时无操作（网关、第一回合等）。
        self._restore_primary_runtime()

        # 清理用户输入中的代理项字符。从
        # 富文本编辑器（Google Docs、Word 等）剪贴板粘贴
        # 可以注入无效 UTF-8 的单独代理项，
        # 并导致 OpenAI SDK 中的 JSON 序列化崩溃。
        if isinstance(user_message, str):
            user_message = _sanitize_surrogates(user_message)
        if isinstance(persist_user_message, str):
            persist_user_message = _sanitize_surrogates(persist_user_message)

        # 从用户输入中剥离泄漏的 <memory-context> 块。当 Honcho 的
        # saveMessages 持久化包含注入上下文的回合时，该块
        # 可以通过消息历史在下一回合的用户消息中重新出现。
        # 此处剥离可防止过时的内存标签泄漏到
        # 对话中，并被用户或模型作为用户文本看到。
        if isinstance(user_message, str):
            user_message = sanitize_context(user_message)
        if isinstance(persist_user_message, str):
            persist_user_message = sanitize_context(persist_user_message)

        # 存储流式回调供 _interruptible_api_call 使用
        self._stream_callback = stream_callback
        self._persist_user_message_idx = None
        self._persist_user_message_override = persist_user_message
        # 如果未提供，则生成唯一的 task_id 以在并发任务之间隔离 VM
        effective_task_id = task_id or str(uuid.uuid4())
        
        # 在每个回合开始时重置重试计数器和迭代预算
        # 以便上一回合的子代理使用不会占用下一回合的预算。
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._incomplete_scratchpad_retries = 0
        self._codex_incomplete_retries = 0
        self._thinking_prefill_retries = 0
        self._post_tool_empty_retried = False
        self._last_content_with_tools = None
        self._last_content_tools_all_housekeeping = False
        self._mute_post_response = False
        self._unicode_sanitization_passes = 0

        # 回合前连接健康检查：检测并清理来自
        # 提供商中断或丢弃流留下的死 TCP
        # 连接。这防止下一次 API 调用在僵尸套接字上挂起。
        if self.api_mode != "anthropic_messages":
            try:
                if self._cleanup_dead_connections():
                    self._emit_status(
                        "🔌 Detected stale connections from a previous provider "
                        "issue — cleaned up automatically. Proceeding with fresh "
                        "connection."
                    )
            except Exception:
                pass
        # 通过 status_callback 为网关
        # 平台重放压缩警告（回调在 __init__ 期间未连接）。
        if self._compression_warning:
            self._replay_compression_warning()
            self._compression_warning = None  # send once

        # 注意：_turns_since_memory 和 _iters_since_skill 此处不重置。
        # 它们在 __init__ 中初始化，必须在 run_conversation
        # 调用之间持久存在，以便提醒逻辑在 CLI 模式下正确累积。
        self.iteration_budget = IterationBudget(self.max_iterations)

        # 记录对话回合开始以便调试/可观察性
        _msg_preview = (user_message[:80] + "...") if len(user_message) > 80 else user_message
        _msg_preview = _msg_preview.replace("\n", " ")
        logger.info(
            "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
            self.session_id or "none", self.model, self.provider or "unknown",
            self.platform or "unknown", len(conversation_history or []),
            _msg_preview,
        )

        # 初始化对话（复制以避免改变调用者的列表）
        messages = list(conversation_history) if conversation_history else []

        # 从对话历史中补充 todo 存储（网关为每条消息
        # 创建一个新的 AIAgent，因此内存存储为空 — 我们需要
        # 从历史记录中最近的 todo 工具响应恢复 todo 状态）
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)
        
        # 预填充消息（少样本引导）仅在 API 调用时注入，
        # 从不存储在消息列表中。这使它们保持临时：它们不会
        # 被保存到会话 DB、会话日志或批处理轨迹，但它们
        # 在每次 API 调用时自动重新应用（包括会话继续）。
        
        # 跟踪用户回合以进行内存刷新和定期提醒逻辑
        self._user_turn_count += 1

        # 保留原始用户消息（无提醒注入）。
        original_user_message = persist_user_message if persist_user_message is not None else user_message

        # 跟踪内存提醒触发器（基于回合，此处检查）。
        # 技能触发器在代理循环完成后检查，基于
        # 此回合使用的工具迭代次数。
        _should_review_memory = False
        if (self._memory_nudge_interval > 0
                and "memory" in self.valid_tool_names
                and self._memory_store):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                _should_review_memory = True
                self._turns_since_memory = 0

        # Add user message
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        current_turn_user_idx = len(messages) - 1
        self._persist_user_message_idx = current_turn_user_idx
        
        if not self.quiet_mode:
            self._safe_print(f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")
        
        # ── 系统提示（每个会话缓存以进行前缀缓存） ──
        # 在第一次调用时构建一次，所有后续调用重用。
        # 仅在上下文压缩事件后重建（这会
        # 使缓存无效并从磁盘重新加载内存）。
        #
        # 对于继续会话（网关为每条
        # 消息创建一个新的 AIAgent），我们从会话 DB
        # 加载存储的系统提示
        # 而不是重建。重建会拾取模型已经知道的
        # 磁盘内存更改（它写入了它们！），
        # 产生不同的系统提示并破坏 Anthropic
        # 前缀缓存。
        if self._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and self._session_db:
                try:
                    session_row = self._session_db.get_session(self.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass  # Fall through to build fresh

            if stored_prompt:
                # 继续会话 — 重用上一回合的
                # 完全相同的系统提示，以便 Anthropic 缓存前缀匹配。
                self._cached_system_prompt = stored_prompt
            else:
                # 新会话的第一回合 — 从头开始构建。
                self._cached_system_prompt = self._build_system_prompt(system_message)
                # 插件钩子：on_session_start
                # 在创建全新会话时触发一次（不是在
                # 继续时）。插件可以使用此来初始化
                # 会话范围状态（例如预热内存缓存）。
                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "on_session_start",
                        session_id=self.session_id,
                        model=self.model,
                        platform=getattr(self, "platform", None) or "",
                    )
                except Exception as exc:
                    logger.warning("on_session_start hook failed: %s", exc)

                # 在 SQLite 中存储系统提示快照
                if self._session_db:
                    try:
                        self._session_db.update_system_prompt(self.session_id, self._cached_system_prompt)
                    except Exception as e:
                        logger.debug("Session DB update_system_prompt failed: %s", e)

        active_system_prompt = self._cached_system_prompt

        # ── 预检上下文压缩 ──
        # 在进入主循环之前，检查加载的对话
        # 历史是否已经超过模型的上下文阈值。这处理
        # 用户切换到具有较小上下文窗口的模型
        # 同时拥有大型现有会话的情况 — 主动压缩
        # 而不是等待 API 错误（可能被捕获为不可重试的
        # 4xx 并完全中止请求）。
        if (
            self.compression_enabled
            and len(messages) > self.context_compressor.protect_first_n
                                + self.context_compressor.protect_last_n + 1
        ):
            # 包含工具模式令牌 — 对于许多工具，这些可以添加
            # 20-30K+ 令牌，旧的 sys+msg 估计完全错过了。
            _preflight_tokens = estimate_request_tokens_rough(
                messages,
                system_prompt=active_system_prompt or "",
                tools=self.tools or None,
            )

            if _preflight_tokens >= self.context_compressor.threshold_tokens:
                logger.info(
                    "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                    f"{_preflight_tokens:,}",
                    f"{self.context_compressor.threshold_tokens:,}",
                    self.model,
                    f"{self.context_compressor.context_length:,}",
                )
                if not self.quiet_mode:
                    self._safe_print(
                        f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                        f">= {self.context_compressor.threshold_tokens:,} threshold"
                    )
                # 对于具有小
                # 上下文窗口的非常大的会话，可能需要多次传递
                # （每次传递总结中间 N 个回合）。
                for _pass in range(3):
                    _orig_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message, approx_tokens=_preflight_tokens,
                        task_id=effective_task_id,
                    )
                    if len(messages) >= _orig_len:
                        break  # Cannot compress further
                    # 压缩创建了新会话 — 清除历史
                    # 引用，以便 _flush_messages_to_session_db 将所有
                    # 压缩消息写入新会话的 SQLite，而不是
                    # 因为 conversation_history 仍然是
                    # 压缩前长度而跳过它们。
                    conversation_history = None
                    # 修复：压缩后重置重试计数器，以便模型
                    # 在压缩上下文上获得新的预算。没有
                    # 这个，压缩前重试会延续，模型
                    # 在压缩导致的
                    # 上下文丢失后立即遇到"(empty)"。
                    self._empty_content_retries = 0
                    self._thinking_prefill_retries = 0
                    self._last_content_with_tools = None
                    self._last_content_tools_all_housekeeping = False
                    self._mute_post_response = False
                    # Re-estimate after compression
                    _preflight_tokens = estimate_request_tokens_rough(
                        messages,
                        system_prompt=active_system_prompt or "",
                        tools=self.tools or None,
                    )
                    if _preflight_tokens < self.context_compressor.threshold_tokens:
                        break  # Under threshold

        # 插件钩子：pre_llm_call
        # 在工具调用循环之前每回合触发一次。插件可以
        # 返回带有 ``context`` 键（或纯字符串）的字典，其
        # 值附加到当前回合的用户消息。
        #
        # 上下文始终注入到用户消息中，从不注入到
        # 系统提示中。这保留了提示缓存前缀 — 
        # 系统提示在回合之间保持相同，因此缓存的令牌
        # 被重用。系统提示是 Hermes 的领域；插件
        # 与用户输入一起贡献上下文。
        #
        # 所有注入的上下文都是临时的（不持久化到会话 DB）。
        _plugin_user_context = ""
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _pre_results = _invoke_hook(
                "pre_llm_call",
                session_id=self.session_id,
                user_message=original_user_message,
                conversation_history=list(messages),
                is_first_turn=(not bool(conversation_history)),
                model=self.model,
                platform=getattr(self, "platform", None) or "",
                sender_id=getattr(self, "_user_id", None) or "",
            )
            _ctx_parts: list[str] = []
            for r in _pre_results:
                if isinstance(r, dict) and r.get("context"):
                    _ctx_parts.append(str(r["context"]))
                elif isinstance(r, str) and r.strip():
                    _ctx_parts.append(r)
            if _ctx_parts:
                _plugin_user_context = "\n\n".join(_ctx_parts)
        except Exception as exc:
            logger.warning("pre_llm_call hook failed: %s", exc)

        # 主对话循环
        api_call_count = 0
        final_response = None
        interrupted = False
        codex_ack_continuations = 0
        length_continue_retries = 0
        truncated_tool_call_retries = 0
        truncated_response_prefix = ""
        compression_attempts = 0
        _turn_exit_reason = "unknown"  # 诊断：循环为何结束
        
        # 记录执行线程，以便 interrupt()/clear_interrupt() 可以
        # 将工具级中断信号仅限定到此代理的线程。
        # 必须在任何线程范围中断同步之前设置。
        self._execution_thread_id = threading.current_thread().ident

        # 始终清除上一回合的陈旧每线程状态。如果
        # 中断在启动完成之前到达，保留它并将其
        # 现在绑定到此执行线程，而不是丢弃它。
        _set_interrupt(False, self._execution_thread_id)
        if self._interrupt_requested:
            _set_interrupt(True, self._execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            self._interrupt_message = None
            self._interrupt_thread_signal_pending = False

        # 通知内存提供商新回合，以便节奏跟踪工作。
        # 必须在 prefetch_all() 之前发生，以便提供商知道这是哪一回合
        # 并可以通过 contextCadence/dialecticCadence 控制上下文/辩证刷新。
        if self._memory_manager:
            try:
                _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
                self._memory_manager.on_turn_start(self._user_turn_count, _turn_msg)
            except Exception:
                pass

        # 外部内存提供商：在工具循环之前预取一次。
        # 在每次迭代中重用缓存结果以避免在每次工具调用时
        # 重新调用 prefetch_all()（10 个工具调用 = 10 倍延迟 + 成本）。
        # 使用 original_user_message（干净输入）— user_message 可能包含
        # 注入的技能内容，这些内容会膨胀/破坏提供商查询。
        _ext_prefetch_cache = ""
        if self._memory_manager:
            try:
                _query = original_user_message if isinstance(original_user_message, str) else ""
                _ext_prefetch_cache = self._memory_manager.prefetch_all(_query) or ""
            except Exception:
                pass

        while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) or self._budget_grace_call:
            # 重置每回合检查点去重，以便每次迭代可以拍摄一个快照
            self._checkpoint_mgr.new_turn()

            # 检查中断请求（例如，用户发送新消息）
            if self._interrupt_requested:
                interrupted = True
                _turn_exit_reason = "interrupted_by_user"
                if not self.quiet_mode:
                    self._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
                break
            
            api_call_count += 1
            self._api_call_count = api_call_count
            self._touch_activity(f"starting API call #{api_call_count}")

            # 宽限调用：预算已耗尽，但我们给了模型
            # 另一次机会。消耗宽限标志，以便循环在此
            # 迭代后退出，无论结果如何。
            if self._budget_grace_call:
                self._budget_grace_call = False
            elif not self.iteration_budget.consume():
                _turn_exit_reason = "budget_exhausted"
                if not self.quiet_mode:
                    self._safe_print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
                break

            # 为网关钩子触发 step_callback（agent:step 事件）
            if self.step_callback is not None:
                try:
                    prev_tools = []
                    for _idx, _m in enumerate(reversed(messages)):
                        if _m.get("role") == "assistant" and _m.get("tool_calls"):
                            _fwd_start = len(messages) - _idx
                            _results_by_id = {}
                            for _tm in messages[_fwd_start:]:
                                if _tm.get("role") != "tool":
                                    break
                                _tcid = _tm.get("tool_call_id")
                                if _tcid:
                                    _results_by_id[_tcid] = _tm.get("content", "")
                            prev_tools = [
                                {
                                    "name": tc["function"]["name"],
                                    "result": _results_by_id.get(tc.get("id")),
                                    "arguments": tc["function"].get("arguments"),
                                }
                                for tc in _m["tool_calls"]
                                if isinstance(tc, dict)
                            ]
                            break
                    self.step_callback(api_call_count, prev_tools)
                except Exception as _step_err:
                    logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

            # 跟踪技能提醒的工具调用迭代。
            # 每当实际使用 skill_manage 时计数器重置。
            self.skill_evolution.on_agent_iteration(self.valid_tool_names)
            
            # 为 API 调用准备消息
            # 如果我们有临时系统提示，将其前置到消息
            # 注意：推理通过 <think> 标签嵌入内容中以进行轨迹存储。
            # 但是，像 Moonshot AI 这样的提供商需要在带有 tool_calls 的
            # 助手消息上有单独的 'reasoning_content' 字段。我们在这里处理两种情况。
            api_messages = []
            for idx, msg in enumerate(messages):
                api_msg = msg.copy()

                # 将临时上下文注入到当前回合的用户消息中。
                # 来源：内存管理器预取 + 插件 pre_llm_call 钩子
                # target="user_message"（默认）。两者都是
                # 仅 API 调用时 — `messages` 中的原始消息
                # 永远不会改变，因此没有任何内容泄漏到会话持久性中。
                if idx == current_turn_user_idx and msg.get("role") == "user":
                    _injections = []
                    if _ext_prefetch_cache:
                        _fenced = build_memory_context_block(_ext_prefetch_cache)
                        if _fenced:
                            _injections.append(_fenced)
                    if _plugin_user_context:
                        _injections.append(_plugin_user_context)
                    if _injections:
                        _base = api_msg.get("content", "")
                        if isinstance(_base, str):
                            api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

                # 对于所有助手消息，将推理传回 API
                # 这确保多回合推理上下文被保留
                if msg.get("role") == "assistant":
                    reasoning_text = msg.get("reasoning")
                    if reasoning_text:
                        # 添加 reasoning_content 以实现 API 兼容性（Moonshot AI、Novita、OpenRouter）
                        api_msg["reasoning_content"] = reasoning_text

                # 删除 'reasoning' 字段 - 它仅用于轨迹存储
                # 我们已将其复制到上面的 API 的 'reasoning_content'
                if "reasoning" in api_msg:
                    api_msg.pop("reasoning")
                # 删除 finish_reason - 严格 API 不接受（例如 Mistral）
                if "finish_reason" in api_msg:
                    api_msg.pop("finish_reason")
                # 剥离内部思考预填充标记
                api_msg.pop("_thinking_prefill", None)
                # 为严格的提供商（如 Mistral、Fireworks 等）剥离 Codex Responses API 字段（call_id、response_item_id），
                # 这些提供商拒绝未知字段。
                # 使用新字典，以便内部消息列表保留这些字段
                # 以实现 Codex Responses 兼容性。
                if self._should_sanitize_tool_calls():
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                # 保留 'reasoning_details' - OpenRouter 将其用于多回合推理上下文
                # 签名字段有助于维护推理连续性
                api_messages.append(api_msg)

            # 构建最终系统消息：缓存提示 + 临时系统提示。
            # 临时添加仅 API 调用时（不持久化到会话 DB）。
            # 外部召回上下文注入到用户消息中，而不是系统
            # 提示中，因此稳定的缓存前缀保持不变。
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            # 注意：来自 pre_llm_call 钩子的插件上下文注入到
            # 用户消息中（见上面的注入块），而不是系统提示中。
            # 这是故意的 — 系统提示修改会破坏提示
            # 缓存前缀。系统提示保留给 Hermes 内部使用。
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages

            # 在系统提示之后但在对话历史之前注入临时预填充消息。
            # 相同的仅 API 调时模式。
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # 通过 OpenRouter 为 Claude 模型应用 Anthropic 提示缓存。
            # 自动检测：如果模型名称包含 "claude" 且 base_url 是 OpenRouter，
            # 注入 cache_control 断点（系统 + 最后 3 条消息）以在
            # 多回合对话中减少约 75% 的输入令牌成本。
            if self._use_prompt_caching:
                api_messages = apply_anthropic_cache_control(api_messages, cache_ttl=self._cache_ttl, native_anthropic=(self.api_mode == 'anthropic_messages'))

            # 安全网：在发送到 API 之前剥离孤立工具结果 / 为缺失
            # 结果添加存根。无条件运行 — 不
            # 受 context_compressor 限制 — 因此会话加载或
            # 手动消息操作的孤立内容总是被捕获。
            api_messages = self._sanitize_api_messages(api_messages)

            # 标准化消息空白和工具调用 JSON 以实现一致的
            # 前缀匹配。确保跨回合的位完美前缀，
            # 这在本地推理服务器上启用 KV 缓存重用
            # （llama.cpp、vLLM、Ollama）并提高
            # 云提供商的缓存命中率。
            # 在 api_messages（API 副本）上操作，因此
            # `messages` 中的原始对话历史未被触及。
            for am in api_messages:
                if isinstance(am.get("content"), str):
                    am["content"] = am["content"].strip()
            for am in api_messages:
                tcs = am.get("tool_calls")
                if not tcs:
                    continue
                new_tcs = []
                for tc in tcs:
                    if isinstance(tc, dict) and "function" in tc:
                        try:
                            args_obj = json.loads(tc["function"]["arguments"])
                            tc = {**tc, "function": {
                                **tc["function"],
                                "arguments": json.dumps(
                                    args_obj, separators=(",", ":"),
                                    sort_keys=True,
                                ),
                            }}
                        except Exception:
                            pass
                    new_tcs.append(tc)
                am["tool_calls"] = new_tcs

            # 在 API 调用之前主动剥离任何代理项字符。
            # 通过 Ollama 提供的模型（Kimi K2.5、GLM-5、Qwen）可以返回
            # 单独的代理项（U+D800-U+DFFF），这些代理项会在 OpenAI SDK 内部
            # 导致 json.dumps() 崩溃。此处清理可防止 3 次重试循环。
            _sanitize_messages_surrogates(api_messages)

            # 计算请求的近似大小以供记录
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = estimate_messages_tokens_rough(api_messages)
            
            # 安静模式的思考旋转器（在 API 调用期间动画）
            thinking_spinner = None
            
            if not self.quiet_mode:
                self._vprint(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
                self._vprint(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
                self._vprint(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
            else:
                # 安静模式下的动画思考旋转器
                face = random.choice(KawaiiSpinner.get_thinking_faces())
                verb = random.choice(KawaiiSpinner.get_thinking_verbs())
                if self.thinking_callback:
                    # CLI TUI 模式：使用 prompt_toolkit 小部件而不是原始旋转器
                    # （在流式和非流式模式下都有效）
                    self.thinking_callback(f"{face} {verb}...")
                elif not self._has_stream_consumers() and self._should_start_quiet_spinner():
                    # 仅在没有流式消费者且
                    # 旋转器输出有安全接收器时使用原始 KawaiiSpinner。
                    spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                    thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=self._print_fn)
                    thinking_spinner.start()
            
            # 如果详细则记录请求详情
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = 3
            primary_recovery_attempted = False
            max_compression_attempts = 3
            codex_auth_retry_attempted=False
            anthropic_auth_retry_attempted=False
            nous_auth_retry_attempted=False
            thinking_sig_retry_attempted = False
            has_retried_429 = False
            restart_with_compressed_messages = False
            restart_with_length_continuation = False

            finish_reason = "stop"
            response = None  # Guard against UnboundLocalError if all retries fail
            api_kwargs = None  # Guard against UnboundLocalError in except handler

            while retry_count < max_retries:
                # ── Nous Portal 速率限制保护 ──────────────────────
                # 如果另一个会话已经记录了 Nous 被速率
                # 限制，则完全跳过 API 调用。每次尝试
                # （包括 SDK 级重试）都会计入 RPH 并
                # 加深速率限制漏洞。
                if self.provider == "nous":
                    try:
                        from agent.nous_rate_guard import (
                            nous_rate_limit_remaining,
                            format_remaining as _fmt_nous_remaining,
                        )
                        _nous_remaining = nous_rate_limit_remaining()
                        if _nous_remaining is not None and _nous_remaining > 0:
                            _nous_msg = (
                                f"Nous Portal rate limit active — "
                                f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                            )
                            self._vprint(
                                f"{self.log_prefix}⏳ {_nous_msg} Trying fallback...",
                                force=True,
                            )
                            self._emit_status(f"⏳ {_nous_msg}")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue
                            # No fallback available — return with clear message
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": (
                                    f"⏳ {_nous_msg}\n\n"
                                    "No fallback provider available. "
                                    "Try again after the reset, or add a "
                                    "fallback provider in config.yaml."
                                ),
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": _nous_msg,
                            }
                    except ImportError:
                        pass
                    except Exception:
                        pass  # 永不让速率保护破坏代理循环

                try:
                    self._reset_stream_delivery_tracking()
                    api_kwargs = self._build_api_kwargs(api_messages)
                    if self._force_ascii_payload:
                        _sanitize_structure_non_ascii(api_kwargs)
                    if self.api_mode == "codex_responses":
                        api_kwargs = self._preflight_codex_api_kwargs(api_kwargs, allow_stream=False)

                    try:
                        from hermes_cli.plugins import invoke_hook as _invoke_hook
                        _invoke_hook(
                            "pre_api_request",
                            task_id=effective_task_id,
                            session_id=self.session_id or "",
                            platform=self.platform or "",
                            model=self.model,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_mode=self.api_mode,
                            api_call_count=api_call_count,
                            message_count=len(api_messages),
                            tool_count=len(self.tools or []),
                            approx_input_tokens=approx_tokens,
                            request_char_count=total_chars,
                            max_tokens=self.max_tokens,
                        )
                    except Exception:
                        pass

                    if env_var_enabled("HERMES_DUMP_REQUESTS"):
                        self._dump_api_request_debug(api_kwargs, reason="preflight")

                    # 始终首选流式路径 — 即使没有流
                    # 消费者。流式传输为我们提供了细粒度的健康
                    # 检查（90 秒陈旧流检测、60 秒读取超时），
                    # 这是非流式路径所缺乏的。没有这个，
                    # 子代理和其他安静模式调用者可能会无限期挂起，
                    # 当提供商使用 SSE ping 保持连接活着
                    # 但从不传递响应时。
                    # 当没有
                    # 消费者注册时，流式路径对回调是无操作的，
                    # 如果提供商不支持，会自动回退到非
                    # 流式。
                    def _stop_spinner():
                        nonlocal thinking_spinner
                        if thinking_spinner:
                            thinking_spinner.stop("")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")

                    _use_streaming = True
                    # 提供商在之前的
                    # 尝试中发出"不支持流式"信号 — 在此
                    # 会话的其余时间切换到非流式，
                    # 而不是每次重试都失败。
                    if getattr(self, "_disable_streaming", False):
                        _use_streaming = False
                    elif not self._has_stream_consumers():
                        # 没有显示/TTS 消费者。仍然首选流式进行
                        # 健康检查，但跳过测试中的 Mock 客户端
                        # （mock 返回 SimpleNamespace，而不是流迭代器）。
                        from unittest.mock import Mock
                        if isinstance(getattr(self, "client", None), Mock):
                            _use_streaming = False

                    if _use_streaming:
                        response = self._interruptible_streaming_api_call(
                            api_kwargs, on_first_delta=_stop_spinner
                        )
                    else:
                        response = self._interruptible_api_call(api_kwargs)
                    
                    api_duration = time.time() - api_start_time
                    
                    # 静默停止思考旋转器 — 后续的响应框或工具
                    # 执行消息信息更丰富。
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                    
                    if self.verbose_logging:
                        # 如果可用，记录带有提供商信息的响应
                        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                    
                    # 在继续之前验证响应形状
                    response_invalid = False
                    error_details = []
                    if self.api_mode == "codex_responses":
                        output_items = getattr(response, "output", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(output_items, list):
                            response_invalid = True
                            error_details.append("response.output is not a list")
                        elif not output_items:
                            # 流式回填可能已失败，但
                            # _normalize_codex_response 仍可以从 response.output_text 恢复。
                            # 仅当该回退也不存在时才标记为无效。
                            _out_text = getattr(response, "output_text", None)
                            _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                            if _out_text_stripped:
                                logger.debug(
                                    "Codex response.output is empty but output_text is present "
                                    "(%d chars); deferring to normalization.",
                                    len(_out_text_stripped),
                                )
                            else:
                                _resp_status = getattr(response, "status", None)
                                _resp_incomplete = getattr(response, "incomplete_details", None)
                                logger.warning(
                                    "Codex response.output is empty after stream backfill "
                                    "(status=%s, incomplete_details=%s, model=%s). %s",
                                    _resp_status, _resp_incomplete,
                                    getattr(response, "model", None),
                                    f"api_mode={self.api_mode} provider={self.provider}",
                                )
                                response_invalid = True
                                error_details.append("response.output is empty")
                    elif self.api_mode == "anthropic_messages":
                        content_blocks = getattr(response, "content", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(content_blocks, list):
                            response_invalid = True
                            error_details.append("response.content is not a list")
                        elif not content_blocks:
                            response_invalid = True
                            error_details.append("response.content is empty")
                    else:
                        if response is None or not hasattr(response, 'choices') or response.choices is None or not response.choices:
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            elif not hasattr(response, 'choices'):
                                error_details.append("response has no 'choices' attribute")
                            elif response.choices is None:
                                error_details.append("response.choices is None")
                            else:
                                error_details.append("response.choices is empty")

                    if response_invalid:
                        # 在打印错误消息之前停止旋转器
                        if thinking_spinner:
                            thinking_spinner.stop("(´;ω;`) oops, retrying...")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")
                        
                        # 无效响应 — 可能是速率限制、提供商超时、
                        # 上游服务器错误或格式错误的响应。
                        retry_count += 1
                        
                        # 主动回退：空/格式错误的响应是常见的
                        # 速率限制症状。立即切换到回退
                        # 而不是使用扩展退避重试。
                        if self._fallback_index < len(self._fallback_chain):
                            self._emit_status("⚠️ Empty/malformed response — switching to fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue

                        # 检查响应中的错误字段（某些提供商包含此字段）
                        error_msg = "Unknown"
                        provider_name = "Unknown"
                        if response and hasattr(response, 'error') and response.error:
                            error_msg = str(response.error)
                            # 尝试从错误元数据中提取提供商
                            if hasattr(response.error, 'metadata') and response.error.metadata:
                                provider_name = response.error.metadata.get('provider_name', 'Unknown')
                        elif response and hasattr(response, 'message') and response.message:
                            error_msg = str(response.message)
                        
                        # 尝试从模型字段获取提供商（OpenRouter 通常返回实际使用的模型）
                        if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                            provider_name = f"model={response.model}"
                        
                        # 检查 x-openrouter-provider 或类似元数据
                        if provider_name == "Unknown" and response:
                            # 记录所有响应属性以供调试
                            resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                            if self.verbose_logging:
                                logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                        
                        # 从响应中提取错误代码以进行上下文诊断
                        _resp_error_code = None
                        if response and hasattr(response, 'error') and response.error:
                            _code_raw = getattr(response.error, 'code', None)
                            if _code_raw is None and isinstance(response.error, dict):
                                _code_raw = response.error.get('code')
                            if _code_raw is not None:
                                try:
                                    _resp_error_code = int(_code_raw)
                                except (TypeError, ValueError):
                                    pass

                        # 从错误代码
                        # 和响应时间构建人类可读的失败提示，
                        # 而不是总是假设速率限制。
                        if _resp_error_code == 524:
                            _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
                        elif _resp_error_code == 504:
                            _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
                        elif _resp_error_code == 429:
                            _failure_hint = f"rate limited by upstream provider (429)"
                        elif _resp_error_code in (500, 502):
                            _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                        elif _resp_error_code in (503, 529):
                            _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
                        elif _resp_error_code is not None:
                            _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
                        elif api_duration < 10:
                            _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
                        elif api_duration > 60:
                            _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
                        else:
                            _failure_hint = f"response time {api_duration:.1f}s"

                        self._vprint(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}", force=True)
                        self._vprint(f"{self.log_prefix}   🏢 Provider: {provider_name}", force=True)
                        cleaned_provider_error = self._clean_error_message(error_msg)
                        self._vprint(f"{self.log_prefix}   📝 Provider message: {cleaned_provider_error}", force=True)
                        self._vprint(f"{self.log_prefix}   ⏱️  {_failure_hint}", force=True)
                        
                        if retry_count >= max_retries:
                            # 在放弃之前尝试回退
                            self._emit_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue
                            self._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                            logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
                                "failed": True  # Mark as failure for filtering
                            }
                        
                        # 重试前退避 — 抖动指数：5 秒基数，120 秒上限
                        wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                        self._vprint(f"{self.log_prefix}⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...", force=True)
                        logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                        
                        # 以小增量睡眠以保持对中断的响应
                        sleep_end = time.time() + wait_time
                        _backoff_touch_counter = 0
                        while time.time() < sleep_end:
                            if self._interrupt_requested:
                                self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                                self._persist_session(messages, conversation_history)
                                self.clear_interrupt()
                                return {
                                    "final_response": f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "interrupted": True,
                                }
                            time.sleep(0.2)
                            # 每 ~30 秒触摸一次活动，以便网关的非活动
                            # 监视器知道我们在退避等待期间仍然活着。
                            _backoff_touch_counter += 1
                            if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                                self._touch_activity(
                                    f"retry backoff ({retry_count}/{max_retries}), "
                                    f"{int(sleep_end - time.time())}s remaining"
                                )
                        continue  # Retry the API call

                    # 在继续之前检查 finish_reason
                    if self.api_mode == "codex_responses":
                        status = getattr(response, "status", None)
                        incomplete_details = getattr(response, "incomplete_details", None)
                        incomplete_reason = None
                        if isinstance(incomplete_details, dict):
                            incomplete_reason = incomplete_details.get("reason")
                        else:
                            incomplete_reason = getattr(incomplete_details, "reason", None)
                        if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"
                    elif self.api_mode == "anthropic_messages":
                        stop_reason_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}
                        finish_reason = stop_reason_map.get(response.stop_reason, "stop")
                    else:
                        finish_reason = response.choices[0].finish_reason
                        assistant_message = response.choices[0].message
                        if self._should_treat_stop_as_truncated(
                            finish_reason,
                            assistant_message,
                            messages,
                        ):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                                force=True,
                            )
                            finish_reason = "length"

                    if finish_reason == "length":
                        self._vprint(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens", force=True)

                        # ── 检测思考预算耗尽 ──────────────
                        # 当模型将所有输出令牌花费在推理上
                        # 而没有剩余令牌用于响应时，继续
                        # 重试是无意义的。早期检测并给出
                        # 针对性错误，而不是浪费 3 次 API 调用。
                        _trunc_content = None
                        _trunc_has_tool_calls = False
                        if self.api_mode in ("chat_completions", "bedrock_converse"):
                            _trunc_msg = response.choices[0].message if (hasattr(response, "choices") and response.choices) else None
                            _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                            _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False
                        elif self.api_mode == "anthropic_messages":
                            # Anthropic response.content 是一个块列表
                            _text_parts = []
                            for _blk in getattr(response, "content", []):
                                if getattr(_blk, "type", None) == "text":
                                    _text_parts.append(getattr(_blk, "text", ""))
                            _trunc_content = "\n".join(_text_parts) if _text_parts else None

                        # 仅当模型
                        # 实际产生推理块但在其后没有可见文本时，
                        # 响应才是"思考耗尽"。
                        # 不使用 <think> 标签的模型（例如 NVIDIA Build 上的 GLM-4.7、
                        # minimax）可能因无关原因返回 content=None 或空
                        # 字符串 — 将这些视为值得继续重试的
                        # 正常截断，而不是
                        # 思考预算耗尽。
                        _has_think_tags = bool(
                            _trunc_content and re.search(
                                r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                                _trunc_content,
                                re.IGNORECASE,
                            )
                        )
                        _thinking_exhausted = (
                            not _trunc_has_tool_calls
                            and _has_think_tags
                            and (
                                (_trunc_content is not None and not self._has_content_after_think_block(_trunc_content))
                                or _trunc_content is None
                            )
                        )

                        if _thinking_exhausted:
                            _exhaust_error = (
                                "Model used all output tokens on reasoning with none left "
                                "for the response. Try lowering reasoning effort or "
                                "increasing max_tokens."
                            )
                            self._vprint(
                                f"{self.log_prefix}💭 Reasoning exhausted the output token budget — "
                                f"no visible response was produced.",
                                force=True,
                            )
                            # 作为响应返回用户友好的消息，以便
                            # CLI（响应框）和网关（聊天消息）都
                            # 自然地显示它，而不是被抑制的错误。
                            _exhaust_response = (
                                "⚠️ **Thinking Budget Exhausted**\n\n"
                                "The model used all its output tokens on reasoning "
                                "and had none left for the actual response.\n\n"
                                "To fix this:\n"
                                "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                                "→ Or switch to a larger/non-reasoning model with `/model`"
                            )
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": _exhaust_response,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": _exhaust_error,
                            }

                        if self.api_mode in ("chat_completions", "bedrock_converse"):
                            assistant_message = response.choices[0].message
                            if not assistant_message.tool_calls:
                                length_continue_retries += 1
                                interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                                messages.append(interim_msg)
                                if assistant_message.content:
                                    truncated_response_prefix += assistant_message.content

                                if length_continue_retries < 3:
                                    self._vprint(
                                        f"{self.log_prefix}↻ Requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )
                                    continue_msg = {
                                        "role": "user",
                                        "content": (
                                            "[System: Your previous response was truncated by the output "
                                            "length limit. Continue exactly where you left off. Do not "
                                            "restart or repeat prior text. Finish the answer directly.]"
                                        ),
                                    }
                                    messages.append(continue_msg)
                                    self._session_messages = messages
                                    self._save_session_log(messages)
                                    restart_with_length_continuation = True
                                    break

                                partial_response = self._strip_think_blocks(truncated_response_prefix).strip()
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": partial_response or None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "Response remained truncated after 3 continuation attempts",
                                }

                        if self.api_mode in ("chat_completions", "bedrock_converse"):
                            assistant_message = response.choices[0].message
                            if assistant_message.tool_calls:
                                if truncated_tool_call_retries < 1:
                                    truncated_tool_call_retries += 1
                                    self._vprint(
                                        f"{self.log_prefix}⚠️  Truncated tool call detected — retrying API call...",
                                        force=True,
                                    )
                                    # 不要将损坏的响应附加到消息；
                                    # 只是从当前
                                    # 消息状态重新运行相同的 API 调用，给模型另一次机会。
                                    continue
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                                    force=True,
                                )
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "Response truncated due to output length limit",
                                }

                        # 如果我们有先前的消息，回滚到最后一个完整状态
                        if len(messages) > 1:
                            self._vprint(f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)

                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)

                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit"
                            }
                        else:
                            # 第一条消息被截断 - 标记为失败
                            self._vprint(f"{self.log_prefix}❌ First response truncated - cannot recover", force=True)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": "First response truncated due to output length limit"
                            }
                    
                    # 跟踪响应中的实际令牌使用量以进行上下文管理
                    if hasattr(response, 'usage') and response.usage:
                        canonical_usage = normalize_usage(
                            response.usage,
                            provider=self.provider,
                            api_mode=self.api_mode,
                        )
                        prompt_tokens = canonical_usage.prompt_tokens
                        completion_tokens = canonical_usage.output_tokens
                        total_tokens = canonical_usage.total_tokens
                        usage_dict = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                        self.context_compressor.update_from_response(usage_dict)

                        # 成功调用后缓存发现的上下文长度。
                        # 仅持久化提供商确认的限制（从
                        # 错误消息解析），而不是猜测的探测层级。
                        if getattr(self.context_compressor, "_context_probed", False):
                            ctx = self.context_compressor.context_length
                            if getattr(self.context_compressor, "_context_probe_persistable", False):
                                save_context_length(self.model, self.base_url, ctx)
                                self._safe_print(f"{self.log_prefix}💾 Cached context length: {ctx:,} tokens for {self.model}")
                            self.context_compressor._context_probed = False
                            self.context_compressor._context_probe_persistable = False

                        self.session_prompt_tokens += prompt_tokens
                        self.session_completion_tokens += completion_tokens
                        self.session_total_tokens += total_tokens
                        self.session_api_calls += 1
                        self.session_input_tokens += canonical_usage.input_tokens
                        self.session_output_tokens += canonical_usage.output_tokens
                        self.session_cache_read_tokens += canonical_usage.cache_read_tokens
                        self.session_cache_write_tokens += canonical_usage.cache_write_tokens
                        self.session_reasoning_tokens += canonical_usage.reasoning_tokens

                        # 记录 API 调用详情以供调试/可观察性
                        _cache_pct = ""
                        if canonical_usage.cache_read_tokens and prompt_tokens:
                            _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                        logger.info(
                            "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                            self.session_api_calls, self.model, self.provider or "unknown",
                            prompt_tokens, completion_tokens, total_tokens,
                            api_duration, _cache_pct,
                        )

                        cost_result = estimate_usage_cost(
                            self.model,
                            canonical_usage,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_key=getattr(self, "api_key", ""),
                        )
                        if cost_result.amount_usd is not None:
                            self.session_estimated_cost_usd += float(cost_result.amount_usd)
                        self.session_cost_status = cost_result.status
                        self.session_cost_source = cost_result.source

                        # 将令牌计数持久化到会话 DB 以供 /insights 使用。
                        # 对每个具有 session_id 的平台执行此操作，以便非 CLI
                        # 会话（网关、cron、委托运行）不会丢失
                        # 令牌/计费数据，如果更高级别的持久化路径
                        # 被跳过或失败。网关/会话存储写入使用
                        # 绝对总数，因此它们安全地覆盖这些每次调用
                        # 增量，而不是重复计算它们。
                        if self._session_db and self.session_id:
                            try:
                                self._session_db.update_token_counts(
                                    self.session_id,
                                    input_tokens=canonical_usage.input_tokens,
                                    output_tokens=canonical_usage.output_tokens,
                                    cache_read_tokens=canonical_usage.cache_read_tokens,
                                    cache_write_tokens=canonical_usage.cache_write_tokens,
                                    reasoning_tokens=canonical_usage.reasoning_tokens,
                                    estimated_cost_usd=float(cost_result.amount_usd)
                                    if cost_result.amount_usd is not None else None,
                                    cost_status=cost_result.status,
                                    cost_source=cost_result.source,
                                    billing_provider=self.provider,
                                    billing_base_url=self.base_url,
                                    billing_mode="subscription_included"
                                    if cost_result.status == "included" else None,
                                    model=self.model,
                                )
                            except Exception:
                                pass  # 永不阻止代理循环
                        
                        if self.verbose_logging:
                            logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                        
                        # 当提示缓存处于活动状态时记录缓存命中统计
                        if self._use_prompt_caching:
                            if self.api_mode == "anthropic_messages":
                                # Anthropic 使用 cache_read_input_tokens / cache_creation_input_tokens
                                cached = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
                                written = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                            else:
                                # OpenRouter 使用 prompt_tokens_details.cached_tokens
                                details = getattr(response.usage, 'prompt_tokens_details', None)
                                cached = getattr(details, 'cached_tokens', 0) or 0 if details else 0
                                written = getattr(details, 'cache_write_tokens', 0) or 0 if details else 0
                            prompt = usage_dict["prompt_tokens"]
                            hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                            if not self.quiet_mode:
                                self._vprint(f"{self.log_prefix}   💾 Cache: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% hit, {written:,} written)")
                    
                    has_retried_429 = False  # 成功时重置
                    # 在成功请求时清除 Nous 速率限制状态 —
                    # 证明限制已重置，其他会话可以
                    # 继续访问 Nous。
                    if self.provider == "nous":
                        try:
                            from agent.nous_rate_guard import clear_nous_rate_limit
                            clear_nous_rate_limit()
                        except Exception:
                            pass
                    self._touch_activity(f"API call #{api_call_count} completed")
                    break  # Success, exit retry loop

                except InterruptedError:
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    api_elapsed = time.time() - api_start_time
                    self._vprint(f"{self.log_prefix}⚡ Interrupted during API call.", force=True)
                    self._persist_session(messages, conversation_history)
                    interrupted = True
                    final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                    break

                except Exception as api_error:
                    # Stop spinner before printing error messages
                    if thinking_spinner:
                        thinking_spinner.stop("(╥_╥) error, retrying...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    # -----------------------------------------------------------
                    # UnicodeEncodeError 恢复。两个常见原因：
                    #   1. 来自剪贴板粘贴的单独代理项（U+D800..U+DFFF）
                    #      （Google Docs、富文本编辑器）— 清理并重试。
                    #   2. 具有 LANG=C 或非 UTF-8 语言环境的系统上的 ASCII 编解码器
                    #      （例如 Chromebooks）— 任何非 ASCII 字符都会失败。
                    #      通过提及 'ascii' 编解码器的错误消息检测。
                    # 我们就地清理消息并可能重试两次：
                    # 首先剥离代理项，然后如果需要再进行一次纯
                    # ASCII 仅语言环境清理。
                    # -----------------------------------------------------------
                    if isinstance(api_error, UnicodeEncodeError) and getattr(self, '_unicode_sanitization_passes', 0) < 2:
                        _err_str = str(api_error).lower()
                        _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                        # 检测代理项错误 — utf-8 编解码器拒绝
                        # 编码 U+D800..U+DFFF。错误文本是：
                        #   "'utf-8' codec can't encode characters in position
                        #    N-M: surrogates not allowed"
                        _is_surrogate_error = (
                            "surrogate" in _err_str
                            or ("'utf-8'" in _err_str and not _is_ascii_codec)
                        )
                        # 从规范 `messages` 列表和 `api_messages`（API 副本，可能携带
                        # 从 `reasoning` 转换的 `reasoning_content`/`reasoning_details` — 
                        # 规范列表没有的字段）中清理代理项。
                        # 如果已构建也清理 `api_kwargs`，如果存在也清理 `prefill_messages`。
                        # 镜像下面的 ASCII 编解码器恢复。
                        _surrogates_found = _sanitize_messages_surrogates(messages)
                        if isinstance(api_messages, list):
                            if _sanitize_messages_surrogates(api_messages):
                                _surrogates_found = True
                        if isinstance(api_kwargs, dict):
                            if _sanitize_structure_surrogates(api_kwargs):
                                _surrogates_found = True
                        if isinstance(getattr(self, "prefill_messages", None), list):
                            if _sanitize_messages_surrogates(self.prefill_messages):
                                _surrogates_found = True
                        # 根据错误类型而不是我们是否
                        # 找到任何内容来决定重试 — _force_ascii_payload / 上面的扩展
                        # 代理项遍历器覆盖所有已知路径，但
                        # 新的转换字段仍可能漏过。如果
                        # 错误是代理项编码失败，始终让
                        # 重试运行；第 ~8781 行的主动清理器
                        # 在下一次迭代时再次运行。受
                        # _unicode_sanitization_passes < 2（外部保护）限制。
                        if _surrogates_found or _is_surrogate_error:
                            self._unicode_sanitization_passes += 1
                            if _surrogates_found:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Stripped invalid surrogate characters from messages. Retrying...",
                                    force=True,
                                )
                            else:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Surrogate encoding error — retrying after full-payload sanitization...",
                                    force=True,
                                )
                            continue
                        if _is_ascii_codec:
                            self._force_ascii_payload = True
                            # ASCII 编解码器：系统编码根本无法处理
                            # 非 ASCII 字符。清理所有
                            # 来自消息/工具模式的非 ASCII 内容并重试。
                            # 清理规范 `messages` 列表和
                            # `api_messages`（在重试循环之前构建的 API 副本，
                            # 可能包含不在 `messages` 中的额外字段，
                            # 如 reasoning_content）。
                            _messages_sanitized = _sanitize_messages_non_ascii(messages)
                            if isinstance(api_messages, list):
                                _sanitize_messages_non_ascii(api_messages)
                            # 如果已经构建也清理最后的 api_kwargs，
                            # 以便转换字段中的剩余非 ASCII 值
                            # （例如 extra_body、reasoning_content）不会通过
                            # _build_api_kwargs 缓存路径存活到下一次尝试。
                            if isinstance(api_kwargs, dict):
                                _sanitize_structure_non_ascii(api_kwargs)
                            _prefill_sanitized = False
                            if isinstance(getattr(self, "prefill_messages", None), list):
                                _prefill_sanitized = _sanitize_messages_non_ascii(self.prefill_messages)

                            _tools_sanitized = False
                            if isinstance(getattr(self, "tools", None), list):
                                _tools_sanitized = _sanitize_tools_non_ascii(self.tools)

                            _system_sanitized = False
                            if isinstance(active_system_prompt, str):
                                _sanitized_system = _strip_non_ascii(active_system_prompt)
                                if _sanitized_system != active_system_prompt:
                                    active_system_prompt = _sanitized_system
                                    self._cached_system_prompt = _sanitized_system
                                    _system_sanitized = True
                            if isinstance(getattr(self, "ephemeral_system_prompt", None), str):
                                _sanitized_ephemeral = _strip_non_ascii(self.ephemeral_system_prompt)
                                if _sanitized_ephemeral != self.ephemeral_system_prompt:
                                    self.ephemeral_system_prompt = _sanitized_ephemeral
                                    _system_sanitized = True

                            _headers_sanitized = False
                            _default_headers = (
                                self._client_kwargs.get("default_headers")
                                if isinstance(getattr(self, "_client_kwargs", None), dict)
                                else None
                            )
                            if isinstance(_default_headers, dict):
                                _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

                            # 清理 API 密钥 — 凭据中的非 ASCII 字符
                            # （例如，来自错误复制粘贴的 ʋ 而不是 v）
                            # 会导致 httpx 在将
                            # Authorization 标头编码为 ASCII 时失败。这是
                            # 持续 UnicodeEncodeError 的最常见原因，
                            # 这种错误在消息/工具清理后仍然存在（#6843）。
                            _credential_sanitized = False
                            _raw_key = getattr(self, "api_key", None) or ""
                            if _raw_key:
                                _clean_key = _strip_non_ascii(_raw_key)
                                if _clean_key != _raw_key:
                                    self.api_key = _clean_key
                                    if isinstance(getattr(self, "_client_kwargs", None), dict):
                                        self._client_kwargs["api_key"] = _clean_key
                                    # 同时更新实时客户端 — 它持有
                                    # api_key 的自己的副本，auth_headers 在
                                    # 每个请求上动态读取。
                                    if getattr(self, "client", None) is not None and hasattr(self.client, "api_key"):
                                        self.client.api_key = _clean_key
                                    _credential_sanitized = True
                                    self._vprint(
                                        f"{self.log_prefix}⚠️  API key contained non-ASCII characters "
                                        f"(bad copy-paste?) — stripped them. If auth fails, "
                                        f"re-copy the key from your provider's dashboard.",
                                        force=True,
                                    )

                            # 在检测到 ASCII 编解码器时始终重试 —
                            # _force_ascii_payload 保证完整的
                            # api_kwargs 有效负载在下一次迭代（第 ~8475 行）被清理。
                            # 即使上面的
                            # 每个组件检查没有发现任何内容
                            # （例如，非 ASCII 仅在 api_messages 的
                            # reasoning_content 中），该标志也会捕获它。
                            # 受 _unicode_sanitization_passes < 2 限制。
                            self._unicode_sanitization_passes += 1
                            _any_sanitized = (
                                _messages_sanitized
                                or _prefill_sanitized
                                or _tools_sanitized
                                or _system_sanitized
                                or _headers_sanitized
                                or _credential_sanitized
                            )
                            if _any_sanitized:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                                    force=True,
                                )
                            else:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                                    force=True,
                                )
                            continue

                    status_code = getattr(api_error, "status_code", None)
                    error_context = self._extract_api_error_context(api_error)

                    # ── 对错误进行分类以进行结构化恢复决策 ──
                    _compressor = getattr(self, "context_compressor", None)
                    _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                    classified = classify_api_error(
                        api_error,
                        provider=getattr(self, "provider", "") or "",
                        model=getattr(self, "model", "") or "",
                        approx_tokens=approx_tokens,
                        context_length=_ctx_len,
                        num_messages=len(api_messages) if api_messages else 0,
                    )
                    logger.debug(
                        "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                        classified.reason.value, classified.status_code,
                        classified.retryable, classified.should_compress,
                        classified.should_rotate_credential, classified.should_fallback,
                    )

                    recovered_with_pool, has_retried_429 = self._recover_with_credential_pool(
                        status_code=status_code,
                        has_retried_429=has_retried_429,
                        classified_reason=classified.reason,
                        error_context=error_context,
                    )
                    if recovered_with_pool:
                        continue
                    if (
                        self.api_mode == "codex_responses"
                        and self.provider == "openai-codex"
                        and status_code == 401
                        and not codex_auth_retry_attempted
                    ):
                        codex_auth_retry_attempted = True
                        if self._try_refresh_codex_client_credentials(force=True):
                            self._vprint(f"{self.log_prefix}🔐 Codex auth refreshed after 401. Retrying request...")
                            continue
                    if (
                        self.api_mode == "chat_completions"
                        and self.provider == "nous"
                        and status_code == 401
                        and not nous_auth_retry_attempted
                    ):
                        nous_auth_retry_attempted = True
                        if self._try_refresh_nous_client_credentials(force=True):
                            print(f"{self.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                            continue
                    if (
                        self.api_mode == "anthropic_messages"
                        and status_code == 401
                        and hasattr(self, '_anthropic_api_key')
                        and not anthropic_auth_retry_attempted
                    ):
                        anthropic_auth_retry_attempted = True
                        from agent.anthropic_adapter import _is_oauth_token
                        if self._try_refresh_anthropic_client_credentials():
                            print(f"{self.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                            continue
                        # 凭据刷新没有帮助 — 显示诊断信息
                        key = self._anthropic_api_key
                        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                        print(f"{self.log_prefix}🔐 Anthropic 401 — authentication failed.")
                        print(f"{self.log_prefix}   Auth method: {auth_method}")
                        print(f"{self.log_prefix}   Token prefix: {key[:12]}..." if key and len(key) > 12 else f"{self.log_prefix}   Token: (empty or short)")
                        print(f"{self.log_prefix}   Troubleshooting:")
                        from hermes_constants import display_hermes_home as _dhh_fn
                        _dhh = _dhh_fn()
                        print(f"{self.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                        print(f"{self.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                        print(f"{self.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
                        print(f"{self.log_prefix}     • 对于 Claude Code：运行 'claude /login' 刷新，然后重试")
                        print(f"{self.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                        print(f"{self.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

                    # ── 思考块签名恢复 ─────────────────
                    # Anthropic 对完整回合
                    # 内容签名思考块。任何上游突变（上下文压缩、
                    # 会话截断、消息合并）都会使
                    # 签名无效 → HTTP 400。恢复：从所有消息中剥离 reasoning_details
                    # 以便下一次重试根本不发送思考
                    # 块。一次性 — 不要无限重试。
                    if (
                        classified.reason == FailoverReason.thinking_signature
                        and not thinking_sig_retry_attempted
                    ):
                        thinking_sig_retry_attempted = True
                        for _m in messages:
                            if isinstance(_m, dict):
                                _m.pop("reasoning_details", None)
                        self._vprint(
                            f"{self.log_prefix}⚠️  Thinking block signature invalid — "
                            f"stripped all thinking blocks, retrying...",
                            force=True,
                        )
                        logging.warning(
                            "%sThinking block signature recovery: stripped "
                            "reasoning_details from %d messages",
                            self.log_prefix, len(messages),
                        )
                        continue

                    retry_count += 1
                    elapsed_time = time.time() - api_start_time
                    self._touch_activity(
                        f"API error recovery (attempt {retry_count}/{max_retries})"
                    )
                    
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    _error_summary = self._summarize_api_error(api_error)
                    logger.warning(
                        "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                        retry_count,
                        max_retries,
                        error_type,
                        self._client_log_context(),
                        _error_summary,
                    )

                    _provider = getattr(self, "provider", "unknown")
                    _base = getattr(self, "base_url", "unknown")
                    _model = getattr(self, "model", "unknown")
                    _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                    self._vprint(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}", force=True)
                    self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    self._vprint(f"{self.log_prefix}   📝 Error: {_error_summary}", force=True)
                    if status_code and status_code < 500:
                        _err_body = getattr(api_error, "body", None)
                        _err_body_str = str(_err_body)[:300] if _err_body else None
                        if _err_body_str:
                            self._vprint(f"{self.log_prefix}   📋 Details: {_err_body_str}", force=True)
                    self._vprint(f"{self.log_prefix}   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

                    # OpenRouter "no tool endpoints" 错误的可操作提示。
                    # 无论回退是否成功都会触发 — 
                    # 用户需要知道他们的模型为什么失败，以便他们可以修复
                    # 他们的提供商路由，而不是仅仅静默回退。
                    if (
                        self._is_openrouter_url()
                        and "support tool use" in error_msg
                    ):
                        self._vprint(
                            f"{self.log_prefix}   💡 No OpenRouter providers for {_model} support tool calling with your current settings.",
                            force=True,
                        )
                        if self.providers_allowed:
                            self._vprint(
                                f"{self.log_prefix}      Your provider_routing.only restriction is filtering out tool-capable providers.",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      Try removing the restriction or adding providers that support tools for this model.",
                                force=True,
                            )
                        self._vprint(
                            f"{self.log_prefix}      Check which providers support tools: https://openrouter.ai/models/{_model}",
                            force=True,
                        )

                    # 在决定重试之前检查中断
                    if self._interrupt_requested:
                        self._vprint(f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: handling API error ({error_type}: {self._clean_error_message(str(api_error))}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    
                    # 在通用 4xx 处理程序之前检查 413 payload-too-large。
                    # 413 是有效负载大小错误 — 正确的响应是
                    # 压缩历史记录并重试，而不是立即中止。
                    status_code = getattr(api_error, "status_code", None)

                    # ── Anthropic Sonnet 长上下文层级门 ───────────
                    # 当 Claude Max（或类似）
                    # 订阅不包含 1M 上下文层级时，Anthropic 返回 HTTP 429 "Extra usage is required for
                    # long context requests"。这
                    # 不是瞬态速率限制 — 重试或切换
                    # 凭据不会有帮助。将上下文减少到 200k（
                    # 标准层级）并压缩。
                    if classified.reason == FailoverReason.long_context_tier:
                        _reduced_ctx = 200000
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length
                        if old_ctx > _reduced_ctx:
                            compressor.update_model(
                                model=self.model,
                                context_length=_reduced_ctx,
                                base_url=self.base_url,
                                api_key=getattr(self, "api_key", ""),
                                provider=self.provider,
                            )
                            # Context probing flags — only set on built-in
                            # compressor (plugin engines manage their own).
                            if hasattr(compressor, "_context_probed"):
                                compressor._context_probed = True
                                # 不要持久化 — 这是订阅层级
                                # 限制，而不是模型能力。如果
                                # 用户稍后启用额外使用，1M 限制
                                # 应该自动恢复。
                                compressor._context_probe_persistable = False
                            self._vprint(
                                f"{self.log_prefix}⚠️  Anthropic long-context tier "
                                f"requires extra usage — reducing context: "
                                f"{old_ctx:,} → {_reduced_ctx:,} tokens",
                                force=True,
                            )

                        compression_attempts += 1
                        if compression_attempts <= max_compression_attempts:
                            original_len = len(messages)
                            messages, active_system_prompt = self._compress_context(
                                messages, system_message,
                                approx_tokens=approx_tokens,
                                task_id=effective_task_id,
                            )
                            # 压缩创建了新会话 — 清除历史
                            # 以便 _flush_messages_to_session_db 将压缩的
                            # 消息写入新会话，而不是跳过它们。
                            conversation_history = None
                            if len(messages) < original_len or old_ctx > _reduced_ctx:
                                self._emit_status(
                                    f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                                    f"(was {old_ctx:,}), retrying..."
                                )
                                time.sleep(2)
                                restart_with_compressed_messages = True
                                break
                        # 如果压缩
                        # 耗尽或没有帮助，则进入正常错误处理。

                    # 速率限制错误（429 或配额耗尽）的主动回退。
                    # 当配置了回退模型时，立即切换而不是
                    # 通过指数退避消耗重试 — 
                    # 主要提供商不会在重试窗口内恢复。
                    is_rate_limited = classified.reason in (
                        FailoverReason.rate_limit,
                        FailoverReason.billing,
                    )
                    if is_rate_limited and self._fallback_index < len(self._fallback_chain):
                        # 如果凭据池轮换可能
                        # 仍然恢复，则不要主动回退。池的重试然后轮换周期
                        # 至少需要再尝试一次才能触发 — 在这里跳转到回退
                        # 提供商会短路它。
                        pool = self._credential_pool
                        pool_may_recover = pool is not None and pool.has_available()
                        if not pool_may_recover:
                            self._emit_status("⚠️ Rate limited — switching to fallback provider...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue

                    # ── Nous Portal：记录速率限制并跳过重试 ─────
                    # 当 Nous 返回 429 时，将重置时间记录到
                    # 共享文件，以便所有会话（cron、网关、辅助）
                    # 知道不要堆积。然后跳过进一步的重试 —
                    # 每一个都会消耗另一个 RPH 请求并加深
                    # 速率限制漏洞。重试循环的迭代顶部
                    # 守护将在下一次传递时捕获此情况并尝试
                    # 回退或以明确消息退出。
                    if (
                        is_rate_limited
                        and self.provider == "nous"
                        and classified.reason == FailoverReason.rate_limit
                        and not recovered_with_pool
                    ):
                        try:
                            from agent.nous_rate_guard import record_nous_rate_limit
                            _err_resp = getattr(api_error, "response", None)
                            _err_hdrs = (
                                getattr(_err_resp, "headers", None)
                                if _err_resp else None
                            )
                            record_nous_rate_limit(
                                headers=_err_hdrs,
                                error_context=error_context,
                            )
                        except Exception:
                            pass
                        # 直接跳到 max_retries — 循环顶部
                        # 守护将处理回退或干净退出。
                        retry_count = max_retries
                        continue

                    is_payload_too_large = (
                        classified.reason == FailoverReason.payload_too_large
                    )

                    if is_payload_too_large:
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        self._emit_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history
                        # so _flush_messages_to_session_db writes compressed
                        # messages to the new session, not skipping them.
                        conversation_history = None

                        if len(messages) < original_len:
                            self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            self._vprint(f"{self.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 payload too large. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Request payload too large (413). Cannot compress further.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }

                    # 在通用 4xx 处理程序之前检查上下文长度错误。
                    # 分类器从以下情况检测上下文溢出：明确错误
                    # 消息、通用 400 + 大会话启发式（#1630）和
                    # 服务器断开连接 + 大会话模式（#2153）。
                    is_context_length_error = (
                        classified.reason == FailoverReason.context_overflow
                    )

                    if is_context_length_error:
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length

                        # ── 区分两种非常不同的错误 ───────────
                        # 1. "Prompt too long"：输入超过上下文窗口。
                        #    修复：减少 context_length + 压缩历史。
                        # 2. "max_tokens too large"：输入正常，但
                        #    input_tokens + 请求的 max_tokens > context_window。
                        #    修复：为此调用减少 max_tokens（输出上限）。
                        #    不要缩小 context_length — 窗口未更改。
                        #
                        # 注意：max_tokens = 输出令牌上限（一个响应）。
                        #       context_length = 总窗口（输入 + 输出组合）。
                        available_out = parse_available_output_tokens_from_error(error_msg)
                        if available_out is not None:
                            # 错误纯粹是输出上限太大。
                            # 将输出限制到可用空间并重试，而不
                            # 触摸 context_length 或触发压缩。
                            safe_out = max(1, available_out - 64)  # 小安全边距
                            self._ephemeral_max_output_tokens = safe_out
                            self._vprint(
                                f"{self.log_prefix}⚠️  Output cap too large for current prompt — "
                                f"retrying with max_tokens={safe_out:,} "
                                f"(available_tokens={available_out:,}; context_length unchanged at {old_ctx:,})",
                                force=True,
                            )
                            # 仍然计入 compression_attempts，这样如果
                            # 错误持续发生，我们不会无限循环。
                            compression_attempts += 1
                            if compression_attempts > max_compression_attempts:
                                self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                                self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                                logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                                self._persist_session(messages, conversation_history)
                                return {
                                    "messages": messages,
                                    "completed": False,
                                    "api_calls": api_call_count,
                                    "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                    "partial": True,
                                    "failed": True,
                                    "compression_exhausted": True,
                                }
                            restart_with_compressed_messages = True
                            break

                        # 错误是关于输入太大 — 减少 context_length。
                        # 尝试从错误消息解析实际限制
                        parsed_limit = parse_context_limit_from_error(error_msg)
                        if parsed_limit and parsed_limit < old_ctx:
                            new_ctx = parsed_limit
                            self._vprint(f"{self.log_prefix}⚠️  Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})", force=True)
                        else:
                            # 降到下一个探测层级
                            new_ctx = get_next_probe_tier(old_ctx)

                        if new_ctx and new_ctx < old_ctx:
                            compressor.update_model(
                                model=self.model,
                                context_length=new_ctx,
                                base_url=self.base_url,
                                api_key=getattr(self, "api_key", ""),
                                provider=self.provider,
                            )
                            # Context probing flags — only set on built-in
                            # compressor (plugin engines manage their own).
                            if hasattr(compressor, "_context_probed"):
                                compressor._context_probed = True
                                # 仅持久化从提供商的
                                # 错误消息解析的限制（实数）。来自 get_next_probe_tier() 的猜测回退
                                # 层级应仅保留在内存中 — 持久化它们会
                                # 用错误的值污染缓存。
                                compressor._context_probe_persistable = bool(
                                    parsed_limit and parsed_limit == new_ctx
                                )
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...", force=True)

                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        self._emit_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history
                        # so _flush_messages_to_session_db writes compressed
                        # messages to the new session, not skipping them.
                        conversation_history = None

                        if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                            if len(messages) < original_len:
                                self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            # 无法进一步压缩且已处于最低层级
                            self._vprint(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                            logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }

                    # 检查不可重试的客户端错误。分类器
                    # 已经考虑了 413、429、529（瞬态）、上下文
                    # 溢出和通用 400 启发式。本地验证
                    # 错误（ValueError、TypeError）是编程错误。
                    is_local_validation_error = (
                        isinstance(api_error, (ValueError, TypeError))
                        and not isinstance(api_error, UnicodeEncodeError)
                    )
                    is_client_error = (
                        is_local_validation_error
                        or (
                            not classified.retryable
                            and not classified.should_compress
                            and classified.reason not in (
                                FailoverReason.rate_limit,
                                FailoverReason.billing,
                                FailoverReason.overloaded,
                                FailoverReason.context_overflow,
                                FailoverReason.payload_too_large,
                                FailoverReason.long_context_tier,
                                FailoverReason.thinking_signature,
                            )
                        )
                    ) and not is_context_length_error

                    if is_client_error:
                        # 在中止之前尝试回退 — 不同的提供商
                        # 可能没有相同的问题（速率限制、身份验证等）
                        self._emit_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        if api_kwargs is not None:
                            self._dump_api_request_debug(
                                api_kwargs, reason="non_retryable_client_error", error=api_error,
                            )
                        self._emit_status(
                            f"❌ Non-retryable error (HTTP {status_code}): "
                            f"{self._summarize_api_error(api_error)}"
                        )
                        self._vprint(f"{self.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                        self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                        self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                        # 常见身份验证错误的可操作指导
                        if classified.is_auth or classified.reason == FailoverReason.billing:
                            if _provider == "openai-codex" and status_code == 401:
                                self._vprint(f"{self.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                                self._vprint(f"{self.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                                self._vprint(f"{self.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                                self._vprint(f"{self.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                            else:
                                self._vprint(f"{self.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                                self._vprint(f"{self.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                                self._vprint(f"{self.log_prefix}      • Does your account have access to {_model}?", force=True)
                                if "openrouter" in str(_base).lower():
                                    self._vprint(f"{self.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                        logging.error(f"{self.log_prefix}Non-retryable client error: {api_error}")
                        # 当错误可能
                        # 与上下文溢出相关时跳过会话持久化（状态 400 + 大会话）。
                        # 持久化失败的用户消息会使
                        # 会话更大，导致下一次
                        # 尝试出现相同的失败。（#1630）
                        if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Skipping session persistence "
                                f"for large failed session to prevent growth loop.",
                                force=True,
                            )
                        else:
                            self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": str(api_error),
                        }

                    if retry_count >= max_retries:
                        # 在回退之前，尝试为主要
                        # 客户端重建一次，以处理瞬态传输错误（陈旧
                        # 连接池、TCP 重置）。每个 API 调用块
                        # 仅尝试一次。
                        if not primary_recovery_attempted and self._try_recover_primary_transport(
                            api_error, retry_count=retry_count, max_retries=max_retries,
                        ):
                            primary_recovery_attempted = True
                            retry_count = 0
                            continue
                        # 在完全放弃之前尝试回退
                        self._emit_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        _final_summary = self._summarize_api_error(api_error)
                        if is_rate_limited:
                            self._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                        else:
                            self._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                        self._vprint(f"{self.log_prefix}   💀 Final error: {_final_summary}", force=True)

                        # 检测 SSE 流丢弃模式（例如 "Network
                        # connection lost"）并提供可操作的指导。
                        # 这通常发生在模型生成
                        # 非常大的工具调用（包含大量内容的 write_file）
                        # 并且代理/CDN 在响应中途丢弃流时。
                        _is_stream_drop = (
                            not getattr(api_error, "status_code", None)
                            and any(p in error_msg for p in (
                                "connection lost", "connection reset",
                                "connection closed", "network connection",
                                "network error", "terminated",
                            ))
                        )
                        if _is_stream_drop:
                            self._vprint(
                                f"{self.log_prefix}   💡 The provider's stream "
                                f"connection keeps dropping. This often happens "
                                f"when the model tries to write a very large "
                                f"file in a single tool call.",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      Try asking the model "
                                f"to use execute_code with Python's open() for "
                                f"large files, or to write the file in smaller "
                                f"sections.",
                                force=True,
                            )

                        logging.error(
                            "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                            self.log_prefix, max_retries, _final_summary,
                            _provider, _model, len(api_messages), f"{approx_tokens:,}",
                        )
                        if api_kwargs is not None:
                            self._dump_api_request_debug(
                                api_kwargs, reason="max_retries_exhausted", error=api_error,
                            )
                        self._persist_session(messages, conversation_history)
                        _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                        if _is_stream_drop:
                            _final_response += (
                                "\n\nThe provider's stream connection keeps "
                                "dropping — this often happens when generating "
                                "very large tool call responses (e.g. write_file "
                                "with long content). Try asking me to use "
                                "execute_code with Python's open() for large "
                                "files, or to write in smaller sections."
                            )
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _final_summary,
                        }

                    # 对于速率限制，如果存在则尊重 Retry-After 标头
                    _retry_after = None
                    if is_rate_limited:
                        _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                        if _resp_headers and hasattr(_resp_headers, "get"):
                            _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                            if _ra_raw:
                                try:
                                    _retry_after = min(int(_ra_raw), 120)  # 上限 2 分钟
                                except (TypeError, ValueError):
                                    pass
                    wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                    if is_rate_limited:
                        self._emit_status(f"⏱️ Rate limited. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries})...")
                    else:
                        self._emit_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
                    logger.warning(
                        "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                        wait_time,
                        retry_count,
                        max_retries,
                        self._client_log_context(),
                        api_error,
                    )
                    # 以小增量睡眠，以便我们可以快速响应中断
                    # 而不是在一个 sleep() 调用中阻塞整个 wait_time
                    sleep_end = time.time() + wait_time
                    _backoff_touch_counter = 0
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            self._persist_session(messages, conversation_history)
                            self.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)  # 每 200ms 检查中断
                        # 每 ~30 秒触摸一次活动，以便网关的非活动
                        # 监视器知道我们在退避等待期间仍然活着。
                        _backoff_touch_counter += 1
                        if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                            self._touch_activity(
                                f"error retry backoff ({retry_count}/{max_retries}), "
                                f"{int(sleep_end - time.time())}s remaining"
                            )
            
            # 如果 API 调用被中断，跳过响应处理
            if interrupted:
                _turn_exit_reason = "interrupted_during_api_call"
                break

            if restart_with_compressed_messages:
                api_call_count -= 1
                self.iteration_budget.refund()
                # 将压缩重启计入重试限制以防止
                # 无限循环，当压缩减少消息但不足以
                # 适应上下文窗口时。
                retry_count += 1
                restart_with_compressed_messages = False
                continue

            if restart_with_length_continuation:
                # 在每次重试时逐步提高输出令牌预算。
                # 重试 1 → 2× 基数，重试 2 → 3× 基数，上限为 32 768。
                # 通过 _ephemeral_max_output_tokens 适用于所有提供商。
                _boost_base = self.max_tokens if self.max_tokens else 4096
                _boost = _boost_base * (length_continue_retries + 1)
                self._ephemeral_max_output_tokens = min(_boost, 32768)
                continue

            # 保护：如果所有重试都用尽而没有成功响应
            # （例如耗尽 retry_count 的重复上下文长度错误），
            # `response` 变量仍然是 None。干净地跳出。
            if response is None:
                _turn_exit_reason = "all_retries_exhausted_no_response"
                print(f"{self.log_prefix}❌ All API retries exhausted with no successful response.")
                self._persist_session(messages, conversation_history)
                break

            try:
                if self.api_mode == "codex_responses":
                    assistant_message, finish_reason = self._normalize_codex_response(response)
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import normalize_anthropic_response
                    assistant_message, finish_reason = normalize_anthropic_response(
                        response, strip_tool_prefix=self._is_anthropic_oauth
                    )
                else:
                    assistant_message = response.choices[0].message
                
                # 将内容标准化为字符串 — 一些 OpenAI 兼容服务器
                # （llama-server 等）将内容作为字典或列表而不是
                # 普通字符串返回，这会导致下游 .strip() 调用崩溃。
                if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                    raw = assistant_message.content
                    if isinstance(raw, dict):
                        assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                    elif isinstance(raw, list):
                        # 多模态内容列表 — 提取文本部分
                        parts = []
                        for part in raw:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and "text" in part:
                                parts.append(str(part["text"]))
                        assistant_message.content = "\n".join(parts)
                    else:
                        assistant_message.content = str(raw)

                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                    _assistant_text = assistant_message.content or ""
                    _invoke_hook(
                        "post_api_request",
                        task_id=effective_task_id,
                        session_id=self.session_id or "",
                        platform=self.platform or "",
                        model=self.model,
                        provider=self.provider,
                        base_url=self.base_url,
                        api_mode=self.api_mode,
                        api_call_count=api_call_count,
                        api_duration=api_duration,
                        finish_reason=finish_reason,
                        message_count=len(api_messages),
                        response_model=getattr(response, "model", None),
                        usage=self._usage_summary_for_api_request_hook(response),
                        assistant_content_chars=len(_assistant_text),
                        assistant_tool_call_count=len(_assistant_tool_calls),
                    )
                except Exception:
                    pass

                # Handle assistant response
                if assistant_message.content and not self.quiet_mode:
                    if self.verbose_logging:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content}")
                    else:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

                # 通知模型思考的进度回调（由子代理
                # 委托用于将子级的推理传递给父级显示）。
                if (assistant_message.content and self.tool_progress_callback):
                    _think_text = assistant_message.content.strip()
                    # 剥除不应泄露到父级显示的推理 XML 标签
                    _think_text = re.sub(
                        r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                    ).strip()
                    # 对于子代理：将第一行传递给父级显示（现有行为）。
                    # 对于具有结构化回调的所有代理：发出 reasoning.available 事件。
                    first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                    if first_line and getattr(self, '_delegate_depth', 0) > 0:
                        try:
                            self.tool_progress_callback("_thinking", first_line)
                        except Exception:
                            pass
                    elif _think_text:
                        try:
                            self.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                        except Exception:
                            pass
                
                # 检查不完整的 <REASONING_SCRATCHPAD>（已打开但从未关闭）
                # 这意味着模型在推理中途耗尽了输出令牌 — 最多重试 2 次
                if has_incomplete_scratchpad(assistant_message.content or ""):
                    self._incomplete_scratchpad_retries += 1
                    
                    self._vprint(f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                    
                    if self._incomplete_scratchpad_retries <= 2:
                        self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)...")
                        # 不要添加损坏的消息，只需重试
                        continue
                    else:
                        # 最大重试次数 - 丢弃此回合并保存为部分
                        self._vprint(f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                        self._incomplete_scratchpad_retries = 0
                        
                        rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)
                        
                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                        }
                
                # 在干净响应时重置不完整草稿板计数器
                self._incomplete_scratchpad_retries = 0

                if self.api_mode == "codex_responses" and finish_reason == "incomplete":
                    self._codex_incomplete_retries += 1

                    interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                    interim_has_content = bool((interim_msg.get("content") or "").strip())
                    interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))

                    if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning:
                        last_msg = messages[-1] if messages else None
                        # 重复检测：两个连续的不完整助手
                        # 消息具有相同的内容和推理将被合并。
                        # 对于仅推理消息（codex_reasoning_items 不同但
                        # 可见内容/推理都为空），我们还比较
                        # 加密项以避免静默丢弃新状态。
                        last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                        interim_codex_items = interim_msg.get("codex_reasoning_items")
                        duplicate_interim = (
                            isinstance(last_msg, dict)
                            and last_msg.get("role") == "assistant"
                            and last_msg.get("finish_reason") == "incomplete"
                            and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                            and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                            and last_codex_items == interim_codex_items
                        )
                        if not duplicate_interim:
                            messages.append(interim_msg)
                            self._emit_interim_assistant_message(interim_msg)

                    if self._codex_incomplete_retries < 3:
                        if not self.quiet_mode:
                            self._vprint(f"{self.log_prefix}↻ Codex response incomplete; continuing turn ({self._codex_incomplete_retries}/3)")
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    self._codex_incomplete_retries = 0
                    self._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Codex response remained incomplete after 3 continuation attempts",
                    }
                elif hasattr(self, "_codex_incomplete_retries"):
                    self._codex_incomplete_retries = 0
                
                # Check for tool calls
                if assistant_message.tool_calls:
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                    
                    # 验证工具调用名称 - 检测模型幻觉
                    # 在验证之前修复不匹配的工具名称
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in self.valid_tool_names:
                            repaired = self._repair_tool_call(tc.function.name)
                            if repaired:
                                print(f"{self.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                                tc.function.name = repaired
                    invalid_tool_calls = [
                        tc.function.name for tc in assistant_message.tool_calls
                        if tc.function.name not in self.valid_tool_names
                    ]
                    if invalid_tool_calls:
                        # Track retries for invalid tool calls
                        self._invalid_tool_retries += 1

                        # 向模型返回有用的错误 — 模型可以在下一回合自我纠正
                        available = ", ".join(sorted(self.valid_tool_names))
                        invalid_name = invalid_tool_calls[0]
                        invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                        self._vprint(f"{self.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for self-correction ({self._invalid_tool_retries}/3)")

                        if self._invalid_tool_retries >= 3:
                            self._vprint(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                            self._invalid_tool_retries = 0
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid tool call: {invalid_preview}"
                            }

                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        messages.append(assistant_msg)
                        for tc in assistant_message.tool_calls:
                            if tc.function.name not in self.valid_tool_names:
                                content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                            else:
                                content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": content,
                            })
                        continue
                    # 成功工具调用验证时重置重试计数器
                    self._invalid_tool_retries = 0
                    
                    # 验证工具调用参数是有效的 JSON
                    # 将空字符串作为空对象处理（常见的模型怪癖）
                    invalid_json_args = []
                    for tc in assistant_message.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, (dict, list)):
                            tc.function.arguments = json.dumps(args)
                            continue
                        if args is not None and not isinstance(args, str):
                            tc.function.arguments = str(args)
                            args = tc.function.arguments
                        # Treat empty/whitespace strings as empty object
                        if not args or not args.strip():
                            tc.function.arguments = "{}"
                            continue
                        try:
                            json.loads(args)
                        except json.JSONDecodeError as e:
                            invalid_json_args.append((tc.function.name, str(e)))
                    
                    if invalid_json_args:
                        # 检查无效 JSON 是否是由于截断而不是
                        # 模型格式错误。路由器有时
                        # 将 finish_reason 从 "length" 重写为 "tool_calls"，
                        # 从上面的长度处理器隐藏截断。
                        # 检测截断：不以 } 或 ] 结尾的参数
                        # （去除空白后）是在流中途被切断的。
                        _truncated = any(
                            not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                            for tc in assistant_message.tool_calls
                            if tc.function.name in {n for n, _ in invalid_json_args}
                        )
                        if _truncated:
                            self._vprint(
                                f"{self.log_prefix}⚠️  Truncated tool call arguments detected "
                                f"(finish_reason={finish_reason!r}) — refusing to execute.",
                                force=True,
                            )
                            self._invalid_json_retries = 0
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit",
                            }

                        # Track retries for invalid JSON arguments
                        self._invalid_json_retries += 1

                        tool_name, error_msg = invalid_json_args[0]
                        self._vprint(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                        if self._invalid_json_retries < 3:
                            self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            # 不是返回部分，而是注入工具错误结果以便模型可以恢复。
                            # 使用工具结果（不是用户消息）保持角色交替。
                            self._vprint(f"{self.log_prefix}⚠️  Injecting recovery tool results for invalid JSON...")
                            self._invalid_json_retries = 0  # Reset for next attempt
                            
                            # 附加助手消息及其（损坏的）tool_calls
                            recovery_assistant = self._build_assistant_message(assistant_message, finish_reason)
                            messages.append(recovery_assistant)
                            
                            # 对每个工具调用响应工具错误结果
                            invalid_names = {name for name, _ in invalid_json_args}
                            for tc in assistant_message.tool_calls:
                                if tc.function.name in invalid_names:
                                    err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                    tool_result = (
                                        f"Error: Invalid JSON arguments. {err}. "
                                        f"For tools with no required parameters, use an empty object: {{}}. "
                                        f"Please retry with valid JSON."
                                    )
                                else:
                                    tool_result = "Skipped: other tool call in this response had invalid JSON."
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                })
                            continue
                    
                    # 成功 JSON 验证时重置重试计数器
                    self._invalid_json_retries = 0

                    # ── 调用后护栏 ──────────────────────────
                    assistant_message.tool_calls = self._cap_delegate_task_calls(
                        assistant_message.tool_calls
                    )
                    assistant_message.tool_calls = self._deduplicate_tool_calls(
                        assistant_message.tool_calls
                    )

                    assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    # 如果此回合同时具有 content 和 tool_calls，则捕获内容
                    # 作为回退最终响应。常见模式：模型在同一回合中
                    # 传递其答案并调用内存/技能工具作为副作用。
                    # 如果工具后的后续回合为空，我们使用这个。
                    turn_content = assistant_message.content or ""
                    if turn_content and self._has_content_after_think_block(turn_content):
                        self._last_content_with_tools = turn_content
                        # Only mute subsequent output when EVERY tool call in
                        # this turn is post-response housekeeping (memory, todo,
                        # skill_manage, etc.).  If any substantive tool is present
                        # (search_files, read_file, write_file, terminal, ...),
                        # keep output visible so the user sees progress.
                        _HOUSEKEEPING_TOOLS = frozenset({
                            "memory", "todo", "skill_manage", "session_search",
                        })
                        _all_housekeeping = all(
                            tc.function.name in _HOUSEKEEPING_TOOLS
                            for tc in assistant_message.tool_calls
                        )
                        self._last_content_tools_all_housekeeping = _all_housekeeping
                        if _all_housekeeping and self._has_stream_consumers():
                            self._mute_post_response = True
                        elif self._should_emit_quiet_tool_messages():
                            clean = self._strip_think_blocks(turn_content).strip()
                            if clean:
                                self._vprint(f"  ┊ 💬 {clean}")
                    
                    # 在附加之前弹出仅思考的预填充消息
                    # （工具调用路径 — 与最终响应路径的理由相同）。
                    _had_prefill = False
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and messages[-1].get("_thinking_prefill")
                    ):
                        messages.pop()
                        _had_prefill = True

                    # 当工具调用跟在预填充
                    # 恢复之后时重置预填充计数器。
                    # 没有这个，计数器会在整个
                    # 对话中累积 — 间歇性
                    # 清空的模型（empty → prefill → tools → empty → prefill →
                    # tools）会消耗两次预填充尝试，第三次清空
                    # 得到零恢复。在这里重置将每个工具
                    # 调用成功视为新的开始。
                    if _had_prefill:
                        self._thinking_prefill_retries = 0
                        self._empty_content_retries = 0
                    # 成功的工具执行 — 重置工具后推动
                    # 标志，以便如果模型在稍后的
                    # 工具回合中清空，它可以再次触发。
                    self._post_tool_empty_retried = False

                    messages.append(assistant_msg)
                    self._emit_interim_assistant_message(assistant_msg)

                    # 在工具执行开始之前关闭任何打开的流式显示（响应框、推理
                    # 框）。中间回合可能
                    # 已经流式传输了打开响应框的早期内容；
                    # 在此处刷新可防止它包装工具馈送行。
                    # 仅向显示回调发出信号 — TTS（_stream_callback）
                    # 不应接收 None（它使用 None 作为流结束）。
                    if self.stream_delta_callback:
                        try:
                            self.stream_delta_callback(None)
                        except Exception:
                            pass

                    self._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                    # 成功工具执行后重置每回合重试计数器，
                    # 这样单次截断不会毒害
                    # 整个对话。
                    truncated_tool_call_retries = 0

                    # 发出信号表示在下一次
                    # 流式文本之前需要段落中断。
                    # 我们不立即发出它，因为
                    # 多个连续的工具迭代会堆积
                    # 冗余的空行。相反，_fire_stream_delta()
                    # 将在下一次真实文本
                    # 到达时前置单个 "\n\n"。
                    self._stream_needs_break = True

                    # 如果调用的唯一工具是
                    # execute_code（程序化工具调用），则退回迭代。
                    # 这些是廉价的 RPC 风格调用，不应消耗预算。
                    _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                    if _tc_names == {"execute_code"}:
                        self.iteration_budget.refund()
                    
                    # 使用来自 API 响应的真实令牌计数来决定
                    # 压缩。prompt_tokens + completion_tokens 是
                    # 提供商报告的实际上下文大小加上
                    # 助手回合 — 下一个提示的紧密下限。
                    # 上面附加的工具结果尚未计算，但
                    # 阈值（默认 50%）留有足够的余地；如果工具
                    # 结果超过它，下一次 API 调用将报告
                    # 实际总数并在那时触发压缩。
                    #
                    # 如果 last_prompt_tokens 为 0（API 断开连接后陈旧
                    # 或提供商未返回使用数据），则回退到粗略
                    # 估计以避免错过压缩。没有这个，
                    # 会话在断开连接后可以无限增长，因为
                    # should_compress(0) 永远不会触发。（#2153）
                    _compressor = self.context_compressor
                    if _compressor.last_prompt_tokens > 0:
                        _real_tokens = (
                            _compressor.last_prompt_tokens
                            + _compressor.last_completion_tokens
                        )
                    else:
                        _real_tokens = estimate_messages_tokens_rough(messages)

                    if self.compression_enabled and _compressor.should_compress(_real_tokens):
                        self._safe_print("  ⟳ compacting context…")
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message,
                            approx_tokens=self.context_compressor.last_prompt_tokens,
                            task_id=effective_task_id,
                        )
                        # 压缩创建了新会话 — 清除历史，以便
                        # _flush_messages_to_session_db 将压缩消息
                        # 写入新会话（参见预压缩注释）。
                        conversation_history = None
                    
                    # 增量保存会话日志（即使中断也可见进度）
                    self._session_messages = messages
                    self._save_session_log(messages)
                    
                    # Continue loop for next response
                    continue
                
                else:
                    # 没有工具调用 - 这是最终响应
                    final_response = assistant_message.content or ""
                    
                    # 修复：进入无工具调用分支时取消静音输出，
                    # 以便用户可以看到空响应警告和恢复
                    # 状态消息。_mute_post_response 是在之前的
                    # 清理工具回合中设置的，不应静默
                    # 最终响应路径。
                    self._mute_post_response = False
                    
                    # 检查响应是否只有思考块，其后没有实际内容
                    if not self._has_content_after_think_block(final_response):
                        # ── 部分流恢复 ─────────────────────
                        # 如果内容在连接
                        # 死亡之前已经流式传输给用户，
                        # 则将其用作最终响应，
                        # 而不是回退到先前回合的回退
                        # 或在重试上浪费 API 调用。
                        _partial_streamed = (
                            getattr(self, "_current_streamed_assistant_text", "") or ""
                        )
                        if self._has_content_after_think_block(_partial_streamed):
                            _turn_exit_reason = "partial_stream_recovery"
                            _recovered = self._strip_think_blocks(_partial_streamed).strip()
                            logger.info(
                                "Partial stream content delivered (%d chars) "
                                "— using as final response",
                                len(_recovered),
                            )
                            self._emit_status(
                                "↻ Stream interrupted — using delivered content "
                                "as final response"
                            )
                            final_response = _recovered
                            self._response_was_previewed = True
                            break

                        # 如果先前回合已经与清理工具调用一起
                        # 传递了真实内容（例如 "You're welcome!" + 内存保存），
                        # 模型没有更多要说的。立即使用较早的内容
                        # 而不是在重试上浪费 API 调用。
                        # 注意：仅当该回合中的所有工具都是
                        # 清理（内存、待办事项等）时才使用此快捷方式。
                        # 当调用了实质性工具
                        # （terminal、search_files 等）时，内容可能是
                        # 任务中段叙述（"I'll scan the directory..."），
                        # 空的后续意味着模型窒息 — 让下面的
                        # 工具后推动处理而不是提前退出。
                        fallback = getattr(self, '_last_content_with_tools', None)
                        if fallback and getattr(self, '_last_content_tools_all_housekeeping', False):
                            _turn_exit_reason = "fallback_prior_turn_content"
                            logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                            self._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                            self._last_content_with_tools = None
                            self._last_content_tools_all_housekeeping = False
                            self._empty_content_retries = 0
                            # 不要修改助手消息内容 — 
                            # 旧代码注入了 "Calling the X tools..."，
                            # 这毒化了对话历史。只需使用
                            # 回退文本作为最终响应并中断。
                            final_response = self._strip_think_blocks(fallback).strip()
                            self._response_was_previewed = True
                            break

                        # ── 工具调用后空响应推动 ───────────
                        # 模型在执行工具调用后返回空。
                        # 这涵盖两种情况：
                        #  (a) 根本没有先前回合的内容 — 模型静默
                        #  (b) 先前回合有内容 + 实质性工具（
                        #      上面的回退被跳过，因为内容
                        #      是任务中段叙述，而不是最终答案）
                        # 不是放弃，而是通过
                        # 附加用户级提示来推动模型继续。这是 #9400 情况：
                        # 较弱的模型（mimo-v2-pro、GLM-5 等）有时
                        # 在工具结果后返回空，而不是继续
                        # 到下一步。一次带推动的重试通常
                        # 可以修复它。
                        _prior_was_tool = any(
                            m.get("role") == "tool"
                            for m in messages[-5:]  # check recent messages
                        )
                        if (
                            _prior_was_tool
                            and not getattr(self, "_post_tool_empty_retried", False)
                        ):
                            self._post_tool_empty_retried = True
                            # 清除陈旧的叙述，以便在推动后的
                            # 稍后空响应中不会重新出现。
                            self._last_content_with_tools = None
                            self._last_content_tools_all_housekeeping = False
                            logger.info(
                                "Empty response after tool calls — nudging model "
                                "to continue processing"
                            )
                            self._emit_status(
                                "⚠️ Model returned empty after tool calls — "
                                "nudging to continue"
                            )
                            # 首先附加空助手消息，以便
                            # 消息序列保持有效：
                            #   tool(result) → assistant("(empty)") → user(nudge)
                            # 没有这个，我们将有 tool → user，大多数
                            # API 会将其拒绝为无效序列。
                            _nudge_msg = self._build_assistant_message(assistant_message, finish_reason)
                            _nudge_msg["content"] = "(empty)"
                            messages.append(_nudge_msg)
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You just executed tool calls but returned an "
                                    "empty response. Please process the tool "
                                    "results above and continue with the task."
                                ),
                            })
                            continue

                        # ── 仅思考预填充继续 ──────────
                        # 模型产生了结构化推理（通过 API
                        # 字段）但没有可见的文本内容。不是
                        # 放弃，而是按原样附加助手消息并
                        # 继续 — 模型将在下一回合看到自己的推理
                        # 并产生文本部分。
                        # 受 clawdbot 的 "incomplete-text" 恢复启发。
                        _has_structured = bool(
                            getattr(assistant_message, "reasoning", None)
                            or getattr(assistant_message, "reasoning_content", None)
                            or getattr(assistant_message, "reasoning_details", None)
                        )
                        if _has_structured and self._thinking_prefill_retries < 2:
                            self._thinking_prefill_retries += 1
                            logger.info(
                                "Thinking-only response (no visible content) — "
                                "prefilling to continue (%d/2)",
                                self._thinking_prefill_retries,
                            )
                            self._emit_status(
                                f"↻ Thinking-only response — prefilling to continue "
                                f"({self._thinking_prefill_retries}/2)"
                            )
                            interim_msg = self._build_assistant_message(
                                assistant_message, "incomplete"
                            )
                            interim_msg["_thinking_prefill"] = True
                            messages.append(interim_msg)
                            self._session_messages = messages
                            self._save_session_log(messages)
                            continue

                        # ── 空响应重试 ──────────────────────
                        # 模型没有返回任何有用的东西。在尝试回退之前重试最多 3
                        # 次。这涵盖
                        # 真正的空响应（没有内容、没有
                        # 推理）和预填充耗尽后的仅推理响应 — 
                        # 像 mimo-v2-pro 这样的模型
                        # 通过 OpenRouter 始终填充推理字段，
                        # 因此旧的 `not _has_structured` 守卫阻止了
                        # 预填充后每个推理模型的重试。
                        _truly_empty = not self._strip_think_blocks(
                            final_response
                        ).strip()
                        _prefill_exhausted = (
                            _has_structured
                            and self._thinking_prefill_retries >= 2
                        )
                        if _truly_empty and (not _has_structured or _prefill_exhausted) and self._empty_content_retries < 3:
                            self._empty_content_retries += 1
                            logger.warning(
                                "Empty response (no content or reasoning) — "
                                "retry %d/3 (model=%s)",
                                self._empty_content_retries, self.model,
                            )
                            self._emit_status(
                                f"⚠️ Empty response from model — retrying "
                                f"({self._empty_content_retries}/3)"
                            )
                            continue

                        # ── 重试耗尽 — 尝试回退提供商 ──
                        # 在用 "(empty)" 放弃之前，尝试
                        # 切换到回退链中的下一个提供商。
                        # 这涵盖模型（例如 GLM-4.5-Air）由于上下文退化或提供商问题
                        # 持续返回空的情况。
                        if _truly_empty and self._fallback_chain:
                            logger.warning(
                                "Empty response after %d retries — "
                                "attempting fallback (model=%s, provider=%s)",
                                self._empty_content_retries, self.model,
                                self.provider,
                            )
                            self._emit_status(
                                "⚠️ Model returning empty responses — "
                                "switching to fallback provider..."
                            )
                            if self._try_activate_fallback():
                                self._empty_content_retries = 0
                                self._emit_status(
                                    f"↻ Switched to fallback: {self.model} "
                                    f"({self.provider})"
                                )
                                logger.info(
                                    "Fallback activated after empty responses: "
                                    "now using %s on %s",
                                    self.model, self.provider,
                                )
                                continue

                        # 重试和回退链耗尽（或没有
                        # 配置回退）。进入
                        # "(empty)" 终端。
                        _turn_exit_reason = "empty_response_exhausted"
                        reasoning_text = self._extract_reasoning(assistant_message)
                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        assistant_msg["content"] = "(empty)"
                        messages.append(assistant_msg)

                        if reasoning_text:
                            reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                            logger.warning(
                                "Reasoning-only response (no visible content) "
                                "after exhausting retries and fallback. "
                                "Reasoning: %s", reasoning_preview,
                            )
                            self._emit_status(
                                "⚠️ Model produced reasoning but no visible "
                                "response after all retries. Returning empty."
                            )
                        else:
                            logger.warning(
                                "Empty response (no content or reasoning) "
                                "after %d retries. No fallback available. "
                                "model=%s provider=%s",
                                self._empty_content_retries, self.model,
                                self.provider,
                            )
                            self._emit_status(
                                "❌ Model returned no content after all retries"
                                + (" and fallback attempts." if self._fallback_chain else
                                   ". No fallback providers configured.")
                            )

                        final_response = "(empty)"
                        break
                    
                    # 成功内容时重置重试计数器/签名
                    self._empty_content_retries = 0
                    self._thinking_prefill_retries = 0

                    if (
                        self.api_mode == "codex_responses"
                        and self.valid_tool_names
                        and codex_ack_continuations < 2
                        and self._looks_like_codex_intermediate_ack(
                            user_message=user_message,
                            assistant_content=final_response,
                            messages=messages,
                        )
                    ):
                        codex_ack_continuations += 1
                        interim_msg = self._build_assistant_message(assistant_message, "incomplete")
                        messages.append(interim_msg)
                        self._emit_interim_assistant_message(interim_msg)

                        continue_msg = {
                            "role": "user",
                            "content": (
                                "[System: Continue now. Execute the required tool calls and only "
                                "send your final answer after completing the task.]"
                            ),
                        }
                        messages.append(continue_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    codex_ack_continuations = 0

                    final_response = truncated_response_prefix + final_response
                    truncated_response_prefix = ""
                    length_continue_retries = 0
                    
                    # 从面向用户的响应中剥离 <think> 块（在消息中为轨迹保留原始）
                    # Strip <think> blocks from user-facing response (keep raw in messages for trajectory)
                    final_response = self._strip_think_blocks(final_response).strip()
                    
                    final_msg = self._build_assistant_message(assistant_message, finish_reason)

                    # 在附加最终响应之前弹出仅思考的预填充消息。
                    # 在附加最终响应之前弹出仅思考的预填充消息。
                    # 这避免了连续的助手
                    # 消息，这些消息会破坏严格交替的提供商
                    #（Anthropic Messages API）并保持历史干净。
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and messages[-1].get("_thinking_prefill")
                    ):
                        messages.pop()

                    messages.append(final_msg)
                    
                    _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                    if not self.quiet_mode:
                        self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                try:
                    print(f"❌ {error_msg}")
                except (OSError, ValueError):
                    logger.error(error_msg)
                
                logger.debug("Outer loop error in API call #%d", api_call_count, exc_info=True)
                
                # 如果已经附加了带有 tool_calls 的助手消息，
                # API 期望每个 tool_call_id 都有 role="tool" 结果。
                # 为任何尚未回答的填充错误结果。
                for idx in range(len(messages) - 1, -1, -1):
                    msg = messages[idx]
                    if not isinstance(msg, dict):
                        break
                    if msg.get("role") == "tool":
                        continue
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        answered_ids = {
                            m["tool_call_id"]
                            for m in messages[idx + 1:]
                            if isinstance(m, dict) and m.get("role") == "tool"
                        }
                        for tc in msg["tool_calls"]:
                            if not tc or not isinstance(tc, dict): continue
                            if tc["id"] not in answered_ids:
                                err_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                messages.append(err_msg)
                    break
                
                # 非工具错误不需要注入合成消息。
                # 错误已经打印给用户（上文），
                # 重试循环继续。注入假的用户/助手
                # 消息会污染历史、消耗令牌，并有可能违反
                # 角色交替不变量。

                # 如果我们接近限制，中断以避免无限循环
                if api_call_count >= self.max_iterations - 1:
                    _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    # 作为助手附加，以便历史对于
                    # 会话恢复保持有效（避免连续的用户消息）。
                    messages.append({"role": "assistant", "content": final_response})
                    break
        
        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ):
            # 预算耗尽 — 通过一次额外的
            # API 调用要求模型总结，剥离工具。
            # _handle_max_iterations 注入一个
            # 用户消息并进行一次无工具请求。
            _turn_exit_reason = f"max_iterations_reached({api_call_count}/{self.max_iterations})"
            self._emit_status(
                f"⚠️ Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
                "— asking model to summarise"
            )
            if not self.quiet_mode:
                self._safe_print(
                    f"\n⚠️  Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
                    "— requesting summary..."
                )
            final_response = self._handle_max_iterations(messages, api_call_count)
        
        # 确定对话是否成功完成
        completed = final_response is not None and api_call_count < self.max_iterations

        # 如果启用则保存轨迹
        self._save_trajectory(messages, user_message, completed)

        # 对话完成后清理此任务的 VM 和浏览器
        self._cleanup_task_resources(effective_task_id)

        # 将会话持久化到 JSON 日志和 SQLite
        self._persist_session(messages, conversation_history)

        # ── 回合退出诊断日志 ─────────────────────────────────────
        # 始终在 INFO 级别记录，以便 agent.log 捕获每个回合结束的原因。
        # 当最后一条消息是工具结果（代理正在工作中）时，在
        # WARNING 级别记录 — 这是用户报告的"突然停止"场景。
        _last_msg_role = messages[-1].get("role") if messages else None
        _last_tool_name = None
        if _last_msg_role == "tool":
            # 回溯查找带有工具调用的助手消息
            for _m in reversed(messages):
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    _tcs = _m["tool_calls"]
                    if _tcs and isinstance(_tcs[0], dict):
                        _last_tool_name = _tcs[-1].get("function", {}).get("name")
                    break

        _turn_tool_count = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
        )
        _resp_len = len(final_response) if final_response else 0
        _budget_used = self.iteration_budget.used if self.iteration_budget else 0
        _budget_max = self.iteration_budget.max_total if self.iteration_budget else 0

        _diag_msg = (
            "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
            "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
        )
        _diag_args = (
            _turn_exit_reason, self.model, api_call_count, self.max_iterations,
            _budget_used, _budget_max,
            _turn_tool_count, _last_msg_role, _resp_len,
            self.session_id or "none",
        )

        if _last_msg_role == "tool" and not interrupted:
            # 代理正在工作中 — 这是"突然停止"的情况。
            logger.warning(
                "Turn ended with pending tool result (agent may appear stuck). "
                + _diag_msg + " last_tool=%s",
                *_diag_args, _last_tool_name,
            )
        else:
            logger.info(_diag_msg, *_diag_args)

        # 插件钩子：post_llm_call
        # 在工具调用循环完成后每回合触发一次。
        # 插件可以使用此钩子持久化对话数据（例如同步
        # 到外部内存系统）。
        if final_response and not interrupted:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "post_llm_call",
                    session_id=self.session_id,
                    user_message=original_user_message,
                    assistant_response=final_response,
                    conversation_history=list(messages),
                    model=self.model,
                    platform=getattr(self, "platform", None) or "",
                )
            except Exception as exc:
                logger.warning("post_llm_call hook failed: %s", exc)

        # 从最后一条助手消息中提取推理（如果有）
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        # 如果适用，使用中断信息构建结果
        result = {
            "final_response": final_response,
            "last_reasoning": last_reasoning,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "partial": False,  # 仅在由于无效工具调用而停止时为 True
            "interrupted": interrupted,
            "response_previewed": getattr(self, "_response_was_previewed", False),
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "input_tokens": self.session_input_tokens,
            "output_tokens": self.session_output_tokens,
            "cache_read_tokens": self.session_cache_read_tokens,
            "cache_write_tokens": self.session_cache_write_tokens,
            "reasoning_tokens": self.session_reasoning_tokens,
            "prompt_tokens": self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens": self.session_total_tokens,
            "last_prompt_tokens": getattr(self.context_compressor, "last_prompt_tokens", 0) or 0,
            "estimated_cost_usd": self.session_estimated_cost_usd,
            "cost_status": self.session_cost_status,
            "cost_source": self.session_cost_source,
        }
        # 如果 /steer 在最终助手回合之后到达（没有更多工具
        # 批次可排入），将其交还给调用者，以便它可以
        # 作为下一个用户回合传递，而不是静默丢失。
        _leftover_steer = self._drain_pending_steer()
        if _leftover_steer:
            result["pending_steer"] = _leftover_steer
        self._response_was_previewed = False
        
        # 如果有中断消息触发了中断，则包含它
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message
        
        # 处理后清除中断状态
        self.clear_interrupt()

        # 清除流回调，以免泄漏到未来的调用中
        self._stream_callback = None

        # 现在检查技能触发 — 基于此回合使用的工具迭代次数。
        _should_review_skills = self.skill_evolution.should_review_after_turn(
            self.valid_tool_names
        )

        # 外部内存提供商：同步完成的回合 + 排队下一次预取。
        # 使用 original_user_message（干净输入）— user_message 可能包含
        # 注入的技能内容，这些内容会膨胀/破坏提供商查询。
        if self._memory_manager and final_response and original_user_message:
            try:
                self._memory_manager.sync_all(original_user_message, final_response)
                self._memory_manager.queue_prefetch_all(original_user_message)
            except Exception:
                pass

        # 后台内存/技能审查 — 在响应交付后运行，
        # 因此它永远不会与用户的任务竞争模型注意力。
        if final_response and not interrupted and (_should_review_memory or _should_review_skills):
            try:
                self._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=_should_review_memory,
                    review_skills=_should_review_skills,
                )
            except Exception:
                pass  # 后台审查是尽力而为的

        # 注意：内存提供商的 on_session_end() + shutdown_all() 在这里
        # 不调用 — run_conversation() 在多回合会话中每条用户消息调用一次。
        # 每回合后关闭会在第二条消息之前杀死
        # 提供商。实际的会话结束清理
        # 由 CLI（atexit / /reset）和网关（会话过期 /
        # _reset_session）处理。

        # 插件钩子：on_session_end
        # 在每次 run_conversation 调用的最末尾触发。
        # 插件可以使用此钩子进行清理、刷新缓冲区等。
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_end",
                session_id=self.session_id,
                completed=completed,
                interrupted=interrupted,
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("on_session_end hook failed: %s", exc)

        return result

    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """
        简单的聊天界面，仅返回最终响应。

        Args:
            message (str): 用户消息
            stream_callback: 在流式传输期间为每个文本增量调用的可选回调。

        Returns:
            str: 最终助手响应
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]


def main(
    query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",
    max_turns: int = 10,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    save_sample: bool = False,
    verbose: bool = False,
    log_prefix_chars: int = 20
):
    """
    直接运行代理的主函数。

    Args:
        query (str): 代理的自然语言查询。默认为 Python 3.13 示例。
        model (str): 要使用的模型名称（OpenRouter 格式：provider/model）。默认为 anthropic/claude-sonnet-4.6。
        api_key (str): 身份验证的 API 密钥。如果未提供则使用 OPENROUTER_API_KEY 环境变量。
        base_url (str): 模型 API 的基本 URL。默认为 https://openrouter.ai/api/v1
        max_turns (int): API 调用迭代的最大次数。默认为 10。
        enabled_toolsets (str): 要启用的工具集的逗号分隔列表。支持预定义
                              工具集（例如 "research"、"development"、"safe"）。
                              可以组合多个工具集："web,vision"
        disabled_toolsets (str): 要禁用的工具集的逗号分隔列表（例如 "terminal"）
        list_tools (bool): 仅列出可用工具并退出
        save_trajectories (bool): 将对话轨迹保存到 JSONL 文件（附加到 trajectory_samples.jsonl）。默认为 False。
        save_sample (bool): 将单个轨迹样本保存到 UUID 命名的 JSONL 文件以供检查。默认为 False。
        verbose (bool): 启用详细日志记录以进行调试。默认为 False。
        log_prefix_chars (int): 在工具调用/响应的日志预览中显示的字符数。默认为 20。

    工具集示例：
        - "research": Web 搜索、提取、爬取 + 视觉工具
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)
    
    # 处理工具列表
    if list_tools:
        from model_tools import get_all_tool_names, get_toolset_for_tool, get_available_toolsets
        from toolsets import get_all_toolsets, get_toolset_info
        
        print("📋 Available Tools & Toolsets:")
        print("-" * 50)
        
        # 显示新工具集系统
        print("\n🎯 Predefined Toolsets (New System):")
        print("-" * 40)
        all_toolsets = get_all_toolsets()
        
        # 按类别分组
        basic_toolsets = []
        composite_toolsets = []
        scenario_toolsets = []
        
        for name, toolset in all_toolsets.items():
            info = get_toolset_info(name)
            if info:
                entry = (name, info)
                if name in ["web", "terminal", "vision", "creative", "reasoning"]:
                    basic_toolsets.append(entry)
                elif name in ["research", "development", "analysis", "content_creation", "full_stack"]:
                    composite_toolsets.append(entry)
                else:
                    scenario_toolsets.append(entry)
        
        # 打印基本工具集
        print("\n📌 Basic Toolsets:")
        for name, info in basic_toolsets:
            tools_str = ', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Tools: {tools_str}")
        
        # 打印复合工具集
        print("\n📂 Composite Toolsets (built from other toolsets):")
        for name, info in composite_toolsets:
            includes_str = ', '.join(info['includes']) if info['includes'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Includes: {includes_str}")
            print(f"    Total tools: {info['tool_count']}")
        
        # 打印场景特定工具集
        print("\n🎭 Scenario-Specific Toolsets:")
        for name, info in scenario_toolsets:
            print(f"  • {name:20} - {info['description']}")
            print(f"    Total tools: {info['tool_count']}")
        
        
        # 显示旧版工具集兼容性
        print("\n📦 Legacy Toolsets (for backward compatibility):")
        legacy_toolsets = get_available_toolsets()
        for name, info in legacy_toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")
        
        # 显示单个工具
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in sorted(all_tools):
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")
        
        print("\n💡 Usage Examples:")
        print("  # Use predefined toolsets")
        print("  python run_agent.py --enabled_toolsets=research --query='search for Python news'")
        print("  python run_agent.py --enabled_toolsets=development --query='debug this code'")
        print("  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'")
        print("  ")
        print("  # Combine multiple toolsets")
        print("  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'")
        print("  ")
        print("  # Disable toolsets")
        print("  python run_agent.py --disabled_toolsets=terminal --query='no command execution'")
        print("  ")
        print("  # Run with trajectory saving enabled")
        print("  python run_agent.py --save_trajectories --query='your question here'")
        return
    
    # 解析工具集选择参数
    enabled_toolsets_list = None
    disabled_toolsets_list = None
    
    if enabled_toolsets:
        enabled_toolsets_list = [t.strip() for t in enabled_toolsets.split(",")]
        print(f"🎯 Enabled toolsets: {enabled_toolsets_list}")
    
    if disabled_toolsets:
        disabled_toolsets_list = [t.strip() for t in disabled_toolsets.split(",")]
        print(f"🚫 Disabled toolsets: {disabled_toolsets_list}")
    
    if save_trajectories:
        print("💾 Trajectory saving: ENABLED")
        print("   - Successful conversations → trajectory_samples.jsonl")
        print("   - Failed conversations → failed_trajectories.jsonl")
    
    # 使用提供的参数初始化代理
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            verbose_logging=verbose,
            log_prefix_chars=log_prefix_chars
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # 使用提供的查询或默认为 Python 3.13 示例
    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )
    else:
        user_query = query
    
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)
    
    # 运行对话
    result = agent.run_conversation(user_query)
    
    print("\n" + "=" * 50)
    print("📋 CONVERSATION SUMMARY")
    print("=" * 50)
    print(f"✅ Completed: {result['completed']}")
    print(f"📞 API Calls: {result['api_calls']}")
    print(f"💬 Messages: {len(result['messages'])}")
    
    if result['final_response']:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result['final_response'])
    
    # 如果请求，将样本轨迹保存到 UUID 命名的文件
    if save_sample:
        sample_id = str(uuid.uuid4())[:8]
        sample_filename = f"sample_{sample_id}.json"
        
        # 将消息转换为轨迹格式（与 batch_runner 相同）
        trajectory = agent._convert_to_trajectory_format(
            result['messages'], 
            user_query, 
            result['completed']
        )
        
        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "completed": result['completed'],
            "query": user_query
        }
        
        try:
            with open(sample_filename, "w", encoding="utf-8") as f:
                # 使用缩进漂亮打印 JSON 以提高可读性
                f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            print(f"\n💾 Sample trajectory saved to: {sample_filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save sample: {e}")
    
    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    fire.Fire(main)
