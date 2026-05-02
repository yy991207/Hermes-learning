# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
source venv/bin/activate                    # 必须先激活虚拟环境
scripts/run_tests.sh                        # 全量测试 (CI 对齐)
scripts/run_tests.sh tests/agent/           # 按目录跑
scripts/run_tests.sh tests/agent/test_foo.py::test_x  # 单个测试
```

See `AGENTS.md` for full architecture, AIAgent loop, TUI design, tool registration, skin system, profiles, and known pitfalls.

## Web UI

Hermes 包含一个 Web Dashboard，入口为 `hermes dashboard` 或 `manage.sh`。

### 后端

- `hermes_cli/web_server.py` —— FastAPI + WebSocket，端口 9119，提供 `/api/*` 接口 + 静态前端资源
- `hermes_cli/config.py` —— DEFAULT_CONFIG 字典，配置项的 Schema 通过 `_build_schema_from_config()` 自动派生；字段级别的 select/textarea 选项定义在 `web_server.py` 的 `_SCHEMA_OVERRIDES` 中
- 配置加载链：`load_config()` 合并 config.yaml → DEFAULT_CONFIG，schema 推断时对 override 字段覆盖 type/options/category

### 前端

- `web/` —— React + TypeScript + Vite + Ant Design + Tailwind CSS
- 构建产物输出到 `hermes_cli/web_dist/`
- 开发时 Vite proxy 把 `/api` 指向 `127.0.0.1:9119`，通过 `.server.token` 注入 session 认证

```bash
cd web
npm run dev         # 开发模式（需要后端已启动）
npm run build       # 生产构建→ hermes_cli/web_dist/
npm run lint        # ESLint
```

### 管理脚本

```bash
./manage.sh start    # 启动后端（创建 venv + 安装 web extra + 后台运行）
./manage.sh stop     # 停止
./manage.sh restart  # 重启
./manage.sh status   # 检查运行状态
./manage.sh logs     # 查看日志
```

### 语音对话链路

```
用户语音 → VAD → STT(transcription_tools) → Agent(LLM) → _run_tts()
                                                               │
                                    ┌──────────────────────────┼──────────────────────┐
                                    ▼                          ▼                      ▼
                                edge-tts                 aliyun/openai       elevenlabs/neutts
                             (免费/默认/回退)             (配置驱动)            (需对应 API Key)
```

TTS provider 在 `web_server.py` 中分发，失败自动回退到 edge-tts。语音 WebSocket 端点：`/api/voice/ws`。

### system_prompt 传递

`agent.system_prompt`（ConfigPage 中配置）→ `AIAgent(ephemeral_system_prompt=...)`，chat 和 voice 两条链路各自传入。

## Key Conventions

- 测试统一用 `scripts/run_tests.sh`，不要直接 `pytest`——wrapper 会清理 API key 环境变量、切换 TZ=UTC、限制 4 个 xdist worker
- 所有 `HERMES_HOME` 路径都用 `hermes_constants.get_hermes_home()`，不能硬编码 `~/.hermes`
- Python 环境：安装用 `venv` + `uv pip install -e ".[all,dev]"`；跑自动化测试前先 `conda activate deepagent`
- 工具注册：在 `tools/` 下创建 `.py` 文件 → 调用 `registry.register()` → 在 `toolsets.py` 中注册到对应 toolset
- 配置新增：改 `DEFAULT_CONFIG`（`config.py`）→ 如需特殊 UI 控件，加 `_SCHEMA_OVERRIDES`（`web_server.py`）
