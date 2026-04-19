# Hermes Agent 项目结构解析

本文档面向首次接触 `hermes-agent` 代码库的开发者，说明仓库的整体职责划分、核心调用链、主要入口、扩展点和测试方式。

## 1. 项目定位

`hermes-agent` 是一个以 Python 为核心的 AI Agent 项目，提供多种使用入口：

- 终端交互 CLI / TUI
- Telegram、Discord、Slack、WhatsApp、Signal 等消息平台网关
- ACP 适配器，用于 VS Code / Zed / JetBrains 等编辑器集成
- MCP、浏览器、终端、文件、代码执行、记忆、技能、定时任务等工具能力
- 批处理、轨迹压缩和 RL 环境，用于研究与训练

项目主体运行时是 Python，TUI 与部分 Web 前端使用 TypeScript / React。

## 2. 顶层目录总览

```text
hermes-agent/
├── run_agent.py              # AIAgent 核心类与主对话循环
├── cli.py                    # 经典交互式 CLI 编排器
├── model_tools.py            # 工具发现、过滤、调用分发入口
├── toolsets.py               # 工具集定义与核心工具清单
├── hermes_state.py           # SQLite 会话库与 FTS5 全文搜索
├── hermes_constants.py       # HERMES_HOME、路径与 profile 常量
├── hermes_logging.py         # 日志能力
├── hermes_time.py            # 时间处理
├── batch_runner.py           # 批处理运行入口
├── trajectory_compressor.py  # 轨迹压缩
├── rl_cli.py                 # RL 相关 CLI
├── agent/                    # Agent 内部组件
├── tools/                    # 工具实现与工具注册体系
├── hermes_cli/               # `hermes` 命令行子命令、配置、认证、安装向导
├── gateway/                  # 多消息平台网关
├── ui-tui/                   # Ink / React 终端 UI
├── tui_gateway/              # TUI 的 Python JSON-RPC 后端
├── acp_adapter/              # ACP 协议适配器
├── cron/                     # 定时任务调度
├── environments/             # RL / benchmark / tool-calling 环境
├── skills/                   # 内置技能
├── optional-skills/          # 可选技能包
├── plugins/                  # 插件，当前主要包含 memory provider 插件
├── tests/                    # Pytest 测试套件
├── docs/                     # 既有设计/迁移/规格文档
├── website/                  # 文档站点
├── web/                      # Web dashboard 前端
├── scripts/                  # 安装、测试、发布和辅助脚本
├── docker/                   # Docker 相关资源
├── packaging/                # 打包发布资源
└── pyproject.toml            # Python 包元数据、依赖与命令入口
```

## 3. 核心运行链路

### 3.1 Agent 对话主链路

核心类是 `run_agent.py` 中的 `AIAgent`。

```text
用户输入
  ↓
AIAgent.run_conversation()
  ↓
构造 system / user / assistant / tool 消息
  ↓
调用模型 chat.completions.create()
  ↓
如果模型返回 tool_calls
  ↓
model_tools.handle_function_call()
  ↓
tools.registry.dispatch()
  ↓
具体 tools/*.py handler 执行
  ↓
tool result 写回 messages
  ↓
继续下一轮模型调用，直到得到最终 assistant 响应
```

关键文件：

- `run_agent.py`：Agent 初始化、模型调用循环、上下文压缩、记忆、工具结果整合。
- `model_tools.py`：工具发现、根据 toolset 和配置过滤工具、执行工具调用。
- `tools/registry.py`：工具注册中心，负责保存 schema、handler、可用性检查和 dispatch。
- `toolsets.py`：定义内置工具集和工具分组。

### 3.2 工具注册链路

工具按文件拆分在 `tools/` 下。每个工具文件在 import 时调用 `registry.register()` 注册 schema 和 handler。

```text
tools/registry.py
  ↑
tools/*.py 顶层 registry.register()
  ↑
model_tools.py 触发工具发现与 schema 汇总
  ↑
run_agent.py / cli.py / gateway / batch_runner 使用工具
```

新增工具通常需要：

1. 在 `tools/your_tool.py` 中实现 handler，并调用 `registry.register()`。
2. 在 `toolsets.py` 中加入核心工具清单或新增 toolset。
3. 确保 handler 返回 JSON 字符串。
4. 如工具需要持久状态，使用 `get_hermes_home()`，不要硬编码 `~/.hermes`。

## 4. 主要模块职责

### 4.1 `agent/`

`agent/` 存放 Agent 内部能力，主要服务于 `run_agent.py`。

```text
agent/
├── prompt_builder.py         # 系统提示词组装
├── context_compressor.py     # 自动上下文压缩
├── prompt_caching.py         # Anthropic prompt caching 支持
├── auxiliary_client.py       # 辅助 LLM 客户端，覆盖视觉、总结等场景
├── model_metadata.py         # 模型上下文长度、token 估算
├── models_dev.py             # models.dev 注册表集成
├── memory_manager.py         # 记忆管理
├── memory_provider.py        # 记忆 provider 抽象
├── credential_pool.py        # 凭据池
├── smart_model_routing.py    # 模型路由
├── skill_commands.py         # 技能 slash command 注入
├── display.py                # CLI spinner 和工具输出展示
└── trajectory.py             # 轨迹保存辅助
```

重点理解：

- `prompt_builder.py` 决定 Agent 每轮看到的稳定系统上下文。
- `context_compressor.py` 是唯一会主动改写历史上下文的核心机制。
- `skill_commands.py` 将技能内容作为用户消息注入，避免破坏 prompt caching。
- `model_metadata.py` 和 `models_dev.py` 用于模型能力、上下文长度和 provider 信息判断。

### 4.2 `tools/`

`tools/` 是项目最大的能力层，包含工具 schema、执行逻辑、安全检查和运行环境。

主要类别：

- 文件与代码：`file_tools.py`、`file_operations.py`、`patch_parser.py`
- 终端与进程：`terminal_tool.py`、`process_registry.py`
- 浏览器与网页：`browser_tool.py`、`browser_cdp_tool.py`、`web_tools.py`
- MCP：`mcp_tool.py`、`mcp_oauth.py`、`mcp_oauth_manager.py`
- 代码执行：`code_execution_tool.py`
- 子 Agent：`delegate_tool.py`
- 记忆与会话搜索：`memory_tool.py`、`session_search_tool.py`
- 技能系统：`skills_tool.py`、`skill_manager_tool.py`、`skills_hub.py`
- 语音与多模态：`vision_tools.py`、`transcription_tools.py`、`tts_tool.py`、`voice_mode.py`
- 定时任务：`cronjob_tools.py`
- 安全与审批：`approval.py`、`tirith_security.py`、`url_safety.py`、`path_security.py`
- 环境后端：`tools/environments/`

`tools/environments/` 提供 local、Docker、SSH、Modal、Daytona、Singularity 等终端后端抽象。

### 4.3 `hermes_cli/`

`hermes_cli/` 是 `hermes` 命令入口背后的 CLI 子系统。

```text
hermes_cli/
├── main.py              # `hermes` 主入口与 profile override
├── config.py            # DEFAULT_CONFIG、环境变量元数据、迁移
├── commands.py          # Slash command 注册表
├── setup.py             # 交互式安装/配置向导
├── auth.py              # provider 凭据解析
├── models.py            # 模型 catalog 与 provider 列表
├── model_switch.py      # /model 切换管线
├── tools_config.py      # `hermes tools`
├── skills_config.py     # `hermes skills`
├── gateway.py           # gateway 子命令
├── skin_engine.py       # CLI 皮肤主题系统
├── doctor.py            # 诊断命令
├── profiles.py          # profile 管理
└── web_server.py        # Web dashboard 后端能力
```

`pyproject.toml` 中的脚本入口：

```text
hermes       -> hermes_cli.main:main
hermes-agent -> run_agent:main
hermes-acp   -> acp_adapter.entry:main
```

### 4.4 `cli.py`

`cli.py` 是经典 prompt_toolkit 交互式 CLI 的主体，核心类是 `HermesCLI`。

职责包括：

- 加载 CLI 配置。
- 初始化皮肤、banner、spinner、回调。
- 管理会话、slash command 和输入循环。
- 将用户输入交给 `AIAgent`。
- 处理审批、澄清、sudo、工具进度展示等终端交互。

Slash command 的来源是 `hermes_cli/commands.py` 中的 `COMMAND_REGISTRY`。新增 alias 通常只需要改注册表。

### 4.5 `gateway/`

`gateway/` 是消息平台入口，核心文件是 `gateway/run.py`。

```text
gateway/
├── run.py                # Gateway 主循环、消息分发、slash command、Agent 调用
├── session.py            # 消息平台会话持久化
├── config.py             # Gateway 配置
├── platforms/            # 平台适配器
├── hooks.py              # hook 系统
├── delivery.py           # 消息投递
├── status.py             # 状态与锁
└── stream_consumer.py    # 流式输出消费
```

典型消息处理链路：

```text
平台 adapter 收到消息
  ↓
gateway.run.MessageEvent
  ↓
slash command 或普通消息分流
  ↓
创建 / 恢复 gateway session
  ↓
调用 AIAgent
  ↓
流式进度、工具结果、最终响应回传平台
```

`gateway/platforms/` 存放 Telegram、Discord、Slack、WhatsApp、Signal、Home Assistant、QQBot 等平台适配器。新增平台时应参考 `gateway/platforms/ADDING_A_PLATFORM.md`。

### 4.6 `ui-tui/` 与 `tui_gateway/`

TUI 是完整替代经典 CLI 的终端 UI，由 TypeScript 负责渲染，Python 负责 Agent 运行。

```text
hermes --tui
  └─ Node / Ink / React 前端
       └─ stdio JSON-RPC
            └─ tui_gateway Python 后端
                 └─ AIAgent + tools + session
```

`ui-tui/` 重点目录：

```text
ui-tui/src/
├── entry.tsx          # TTY gate 与 render()
├── app.tsx            # UI 入口
├── gatewayClient.ts   # 子进程和 JSON-RPC bridge
├── app/               # 应用状态机和事件处理
├── components/        # Ink 组件
├── hooks/             # 输入、补全、队列、虚拟历史 hooks
├── lib/               # 纯辅助函数
└── protocol/          # 协议类型与处理
```

`tui_gateway/` 重点文件：

- `entry.py`：stdio 后端入口。
- `server.py`：JSON-RPC 方法和事件处理。
- `slash_worker.py`：持久化 CLI 子进程，用于复用 slash command 逻辑。
- `render.py`：可选 rich / ANSI 渲染桥接。

### 4.7 `acp_adapter/`

`acp_adapter/` 提供 Agent Client Protocol 服务，使 Hermes 可被编辑器或其他 ACP 客户端调用。

```text
acp_adapter/
├── entry.py        # hermes-acp 入口
├── server.py       # ACP server 方法实现
├── session.py      # ACP session 管理
├── tools.py        # ACP tool 暴露
├── permissions.py  # 权限处理
├── events.py       # 事件转换
└── auth.py         # 认证辅助
```

### 4.8 `cron/`

`cron/` 实现自然语言自动化和定时任务。

```text
cron/
├── jobs.py         # job 数据结构、持久化和执行信息
└── scheduler.py    # 调度循环与 croniter 集成
```

相关入口还包括：

- `tools/cronjob_tools.py`
- `hermes_cli/cron.py`
- gateway 中的 background / watcher 通知逻辑

### 4.9 `environments/`

`environments/` 是研究和训练相关环境层，服务于 RL、benchmark 和 tool-calling 评估。

```text
environments/
├── hermes_base_env.py
├── agent_loop.py
├── agentic_opd_env.py
├── web_research_env.py
├── tool_context.py
├── benchmarks/
├── hermes_swe_env/
├── terminal_test_env/
└── tool_call_parsers/
```

这部分与普通 CLI/Gateway 使用链路相对独立，更偏向训练数据、评测和研究。

### 4.10 `skills/`、`optional-skills/`、`plugins/`

`skills/` 是随项目分发的内置技能库，按领域组织，例如：

- `skills/software-development/`
- `skills/github/`
- `skills/creative/`
- `skills/mlops/`
- `skills/research/`

`optional-skills/` 是可选技能集合，安装或启用后再参与 Agent 工作。

`plugins/` 当前主要包含 memory provider 插件，例如：

- `plugins/memory/honcho/`
- `plugins/memory/mem0/`
- `plugins/memory/supermemory/`
- `plugins/memory/byterover/`

技能系统的运行时连接点主要在：

- `agent/skill_commands.py`
- `agent/skill_utils.py`
- `tools/skills_tool.py`
- `tools/skills_hub.py`
- `hermes_cli/skills_hub.py`

### 4.11 `web/` 与 `website/`

`web/` 是 Web dashboard 前端项目，使用 Vite / React / TypeScript。

`website/` 是文档站点源码，包含用户指南、参考文档、开发者指南、集成说明等。

两者和核心 Agent 运行时解耦，但共享 Hermes 的配置、工具和文档语义。

### 4.12 `tests/`

测试按功能域拆分：

```text
tests/
├── agent/
├── cli/
├── cron/
├── gateway/
├── hermes_cli/
├── tools/
├── tui_gateway/
├── acp/
├── environments/
├── plugins/
├── skills/
├── e2e/
└── integration/
```

项目要求优先使用包装脚本运行测试：

```bash
source venv/bin/activate
scripts/run_tests.sh
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/agent/test_foo.py::test_x
```

该脚本会清理凭据环境变量、隔离 HOME / HERMES_HOME、设置 UTC 和固定 xdist worker，以接近 CI 环境。

## 5. 配置与状态目录

用户配置默认位于 Hermes home 下：

```text
~/.hermes/config.yaml
~/.hermes/.env
~/.hermes/skills/
~/.hermes/sessions/
```

项目支持 profiles。代码中涉及 Hermes 状态路径时应遵守：

- 读写状态使用 `get_hermes_home()`。
- 面向用户展示路径使用 `display_hermes_home()`。
- 不要硬编码 `~/.hermes` 或 `Path.home() / ".hermes"`。
- profile 列表根目录是 HOME 锚定，而不是当前 `HERMES_HOME` 锚定。

## 6. 关键扩展点

### 6.1 新增工具

涉及文件：

- `tools/your_tool.py`
- `toolsets.py`

基本模式：

```python
from tools.registry import registry

def your_tool(param: str, task_id: str | None = None) -> str:
    return '{"success": true}'

registry.register(
    name="your_tool",
    toolset="your_toolset",
    schema={
        "name": "your_tool",
        "description": "Describe what the tool does.",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            },
        },
    },
    handler=lambda args, **kw: your_tool(
        param=args.get("param", ""),
        task_id=kw.get("task_id"),
    ),
)
```

### 6.2 新增 Slash Command

涉及文件：

- `hermes_cli/commands.py`
- `cli.py`
- 如需 Gateway 支持，再改 `gateway/run.py`

命令定义集中在 `COMMAND_REGISTRY`。CLI、Gateway help、Telegram 菜单、Slack routing、autocomplete 都从该注册表派生。

### 6.3 新增 Gateway 平台

涉及目录：

- `gateway/platforms/`
- `gateway/platforms/ADDING_A_PLATFORM.md`
- `gateway/run.py`
- `gateway/config.py`

如果平台使用唯一 token 或 API key，适配器启动时应使用 `gateway.status.acquire_scoped_lock()`，停止时释放，避免多个 profile 同时使用同一凭据。

### 6.4 新增配置项

涉及文件：

- `hermes_cli/config.py`
- 可能涉及 `cli.py` 的 `load_cli_config()`
- 可能涉及 `gateway/run.py` 的直接 YAML 加载逻辑

新增 `config.yaml` 字段时需要更新 `DEFAULT_CONFIG` 并 bump `_config_version`。

新增 `.env` 变量时需要加入 `OPTIONAL_ENV_VARS`。

## 7. 开发命令

Python 开发：

```bash
source venv/bin/activate
scripts/run_tests.sh
```

TUI 开发：

```bash
cd ui-tui
npm install
npm run dev
npm run type-check
npm run lint
npm test
```

Web dashboard 开发：

```bash
cd web
npm install
npm run dev
```

包入口检查：

```bash
hermes
hermes --tui
hermes gateway
hermes-acp
```

## 8. 阅读代码的推荐顺序

如果目标是理解主流程，建议按以下顺序阅读：

1. `pyproject.toml`：确认安装入口、依赖和包结构。
2. `run_agent.py`：理解 `AIAgent` 和主对话循环。
3. `model_tools.py` 与 `tools/registry.py`：理解工具如何被发现、过滤、调用。
4. `toolsets.py`：理解工具集分组。
5. `cli.py`：理解经典 CLI 如何接入 Agent。
6. `hermes_cli/main.py`：理解 `hermes` 命令入口、profile 和子命令。
7. `gateway/run.py`：理解消息平台如何接入 Agent。
8. `ui-tui/src/gatewayClient.ts` 与 `tui_gateway/server.py`：理解 TUI 前后端协议。
9. `hermes_state.py`：理解会话存储与搜索。
10. `tests/`：按功能域寻找对应测试样例。

## 9. 维护注意事项

- 不要在运行中改变历史上下文、重载 memory 或重建系统 prompt，除非走既有上下文压缩机制，否则会破坏 prompt caching。
- 不要在工具 schema 描述中硬编码引用其他 toolset 的工具名，因为目标工具可能未启用。
- 不要硬编码 `~/.hermes`，使用 `get_hermes_home()` 和 `display_hermes_home()`。
- 不要直接调用 `pytest` 作为默认测试方式，优先使用 `scripts/run_tests.sh`。
- 不要在 spinner / display 代码中使用 `\033[K`，应使用空格补齐清行。
- Gateway 平台适配器涉及唯一凭据时应实现 scoped lock。

## 10. 一句话架构总结

Hermes Agent 的核心是 `AIAgent` 同步对话循环；CLI、TUI、Gateway、ACP、批处理和研究环境都是不同入口层；工具能力通过 `tools.registry` 统一注册，再由 `model_tools.handle_function_call()` 统一分发；会话、记忆、技能、配置和 profile 共同构成跨入口一致的运行上下文。
