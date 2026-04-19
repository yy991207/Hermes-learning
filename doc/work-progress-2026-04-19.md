# 2026-04-19 工作进度总结

本文档记录本次围绕 Hermes Agent 项目结构、Skill 自我优化机制分析与重构的工作进展。

## 1. 项目结构解析

已完成一份项目结构解析文档：

```text
doc/project-structure.md
```

该文档整理了 Hermes Agent 的整体架构，包括：

- 项目定位：Python Agent Runtime + CLI / TUI / Gateway / ACP / Web 等多入口。
- 顶层目录职责：`run_agent.py`、`agent/`、`tools/`、`hermes_cli/`、`gateway/`、`ui-tui/`、`tui_gateway/`、`acp_adapter/`、`cron/`、`environments/`、`skills/`、`plugins/`、`tests/` 等。
- 核心调用链：用户输入进入 `AIAgent.run_conversation()`，模型返回 `tool_calls` 后经 `model_tools.handle_function_call()` 分发到工具注册表。
- 主要扩展点：新增工具、新增 Slash Command、新增 Gateway 平台、新增配置项。
- 开发与测试入口：强调项目要求优先使用 `scripts/run_tests.sh`。

## 2. Agent 框架判断

已确认 Hermes Agent 的 Agent 编排框架是自研的，不是基于 LangChain、LlamaIndex、AutoGen、CrewAI 等现成框架。

核心判断依据：

```text
run_agent.py
    AIAgent 类和 run_conversation 主循环

model_tools.py
    工具发现和 handle_function_call

tools/registry.py
    ToolRegistry 工具注册中心
```

底层模型调用依赖 OpenAI SDK、Anthropic SDK 和自定义 provider adapter，但 Agent loop、工具编排、上下文压缩、记忆、技能、会话持久化、Gateway/TUI 回调等都是项目自研 runtime。

## 3. `run_agent.py` 作用分析

已梳理 `run_agent.py` 的主要职责：

- 定义核心类 `AIAgent`。
- 构建和缓存 system prompt。
- 执行完整 Agent 对话循环。
- 适配多种 provider / API mode。
- 准备 API 请求消息，处理 reasoning、tool_calls、prompt caching、strict API 字段兼容。
- 校验并执行工具调用。
- 管理上下文压缩、memory、session、trajectory、usage、成本统计。
- 支持流式输出、打断、steer、Gateway/TUI callback。
- 提供 `chat()` 和 `main()` 作为简单使用入口。

结论：`run_agent.py` 是 Hermes Agent 的 Agent Runtime Kernel，也是当前代码库中职责最重的中枢文件。

## 4. Skill 自我优化机制解析

已完成独立文档：

```text
doc/skill-self-optimization.md
```

文档中用中文说明了 Hermes 的 Skill 自我优化机制：

```text
复杂任务执行
    |
    v
Agent 发现可复用经验、踩坑、用户纠正或非平凡流程
    |
    v
调用 skill_manage 创建或修补 Skill
    |
    v
写入 ~/.hermes/skills/<skill-name>/SKILL.md
    |
    v
后续系统提示词扫描 Skill 索引
    |
    v
相关任务通过 skill_view 加载 Skill
    |
    v
按沉淀经验执行，并在发现问题时继续 patch
```

关键结论：

- 自我优化不是训练模型权重，而是文档型 procedural memory。
- 进行中优化由模型根据 `SKILLS_GUIDANCE` 和工具 schema 主动调用 `skill_manage`。
- 后台优化由 `run_agent.py` 的计数器和 review Agent 触发。
- `5+ tool calls` 是自然语言提示中的复杂任务判断标准。
- `10` 是默认后台 review 触发阈值，对应 `skills.creation_nudge_interval`。
- 真正落盘逻辑在 `tools/skill_manager_tool.py`。
- 后续生效路径在 `agent/prompt_builder.py` 和 `tools/skills_tool.py`。

## 5. Skill 自我优化重构进展

用户提出希望将自我进化功能落成独立类。已完成初步重构：

```text
agent/skill_evolution.py
```

新增类：

```python
class SkillEvolutionManager:
    ...
```

该类当前承担以下职责：

- 维护 Skill 自我优化的计数器。
- 从配置读取 `skills.creation_nudge_interval`。
- 在 Agent 迭代时累计 `iters_since_skill`。
- 在实际调用 `skill_manage` 后重置计数器。
- 在回合结束时判断是否需要触发后台 Skill review。
- 选择 memory review、skill review 或 combined review prompt。
- 启动静默后台 review Agent。
- 从后台 review 的 tool result 中提取简短摘要。

`run_agent.py` 已接入该类：

- 新增 `from agent.skill_evolution import SkillEvolutionManager`。
- 在 `AIAgent.__init__` 中创建 `self.skill_evolution`。
- 将 `_iters_since_skill` 和 `_skill_nudge_interval` 保留为兼容属性，内部转发到 `SkillEvolutionManager`。
- 工具调用中 `skill_manage` 的重置逻辑委托给 `self.skill_evolution.on_tool_invoked()`。
- Agent loop 中的 Skill 迭代计数委托给 `self.skill_evolution.on_agent_iteration()`。
- 回合结束时的后台 review 触发判断委托给 `self.skill_evolution.should_review_after_turn()`。
- `_spawn_background_review()` 兼容入口已转发到 `self.skill_evolution.spawn_background_review()`。

## 6. 新增测试

已新增轻量单元测试：

```text
tests/agent/test_skill_evolution.py
```

覆盖内容：

- 从配置读取 `creation_nudge_interval`。
- 只有 `skill_manage` 可用时才累计 Skill 迭代计数。
- 达到阈值后触发 review 并重置计数器。
- 调用 `skill_manage` 后重置计数器。
- 后台 review 摘要提取逻辑。

## 7. 当前验证状态

本地尝试验证时遇到环境阻塞：

- 当前 Windows shell 中 `python` 指向 `C:\Users\75550\AppData\Local\Microsoft\WindowsApps\python.exe` 占位符，不能执行项目 Python。
- 未发现 `venv` 或 `.venv` 下的 Python 可执行文件。
- 尝试使用 WSL 运行 `scripts/run_tests.sh`，但 WSL 未安装可用 Linux 发行版。

因此尚未完成 `py_compile` 或 pytest 验证。

建议后续在可用 Python / WSL / CI 环境中运行：

```bash
source venv/bin/activate
scripts/run_tests.sh tests/agent/test_skill_evolution.py tests/run_agent/test_run_agent.py::TestMemoryNudgeCounterPersistence -q
```

## 8. 当前风险与后续建议

当前工作区中 `run_agent.py` 已存在大量文本变化，主要表现为注释和文档字符串中文化。重构时已尽量只做最小接入，但仍需注意：

- `run_agent.py` 的 diff 较大，提交前建议人工 review。
- `_spawn_background_review()` 已转发到 `SkillEvolutionManager`，但旧方法体中仍保留了一段不可达旧代码，原因是当前文件存在编码显示异常，直接大段删除风险较高。
- 后续建议在单独 PR 中清理 `run_agent.py` 中不可达旧代码。
- 后续建议将 memory review 也进一步拆出，形成独立的 `MemoryEvolutionManager` 或统一 `ExperienceReviewManager`。

## 9. 本次新增/修改文件

```text
doc/project-structure.md
doc/skill-self-optimization.md
doc/work-progress-2026-04-19.md
agent/skill_evolution.py
tests/agent/test_skill_evolution.py
run_agent.py
```

其中 `run_agent.py` 包含本次接入 `SkillEvolutionManager` 的改动，同时工作区中也有此前存在的大量中文化文本变化。
