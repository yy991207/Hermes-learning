# 📽️ 短视频讲稿(深度版):《拆解 Hermes Agent 的自我进化体系 —— 从代码到哲学》

> 时长建议 18–22 分钟｜技术向口播 + 代码演示 + 架构图切换
> 风格:技术深度 + 口语化衔接,关键概念保留英文与中文原文对照

---

## 🎬 开场(0:00 – 0:45)

大家好。今天这期不讲八卦,我们硬核一点。

我花了两天时间把一个叫 Hermes Agent 的开源项目翻了个底朝天,就为搞清楚一个问题 ——

> 一个 AI Agent,它所谓的「自我进化」,从代码层面到底是怎么实现的?

市面上所有 Agent 框架嘴上都喊「自主学习」「记忆系统」「持续优化」,但真打开代码一看,很多就是一个向量数据库加几个 RAG 调用,谈不上什么体系。

Hermes 不一样。它的进化逻辑横跨 4 个时间尺度,11 个场景,涉及
SkillEvolutionManager、MemoryManager、ContextCompressor、Skills
Hub、Trajectory、Insights、ErrorClassifier 等至少 10 个核心模块协同。

我今天把它完整拆开,讲它的原理、它的代码骨架、怎么复用、它的硬伤在哪、以及我认为这事的未来。话不多说,开始。

---

## 🧠 第一章:先定义清楚 ——「自我进化」到底是什么?(0:45 – 2:30)

在拆 Hermes 之前,先把概念摆正。

Agent 领域里所谓的「自我进化」,本质是三类记忆的持续再加工:

- **Declarative Memory(陈述性记忆)** —— 记「事实」,比如用户偏好、环境配置。Hermes 里对应 MEMORY.md 和 USER.md。
- **Procedural Memory(程序性记忆)** —— 记「怎么做」,比如"遇到 X 问题用 Y 套方法解"。Hermes 里对应 SKILL.md。
- **Episodic Memory(情景记忆)** —— 记「发生过什么」,原始对话轨迹。Hermes 里对应 trajectory 落盘 + session_search。

Hermes 的原话,写在 `tools/skill_manager_tool.py` 的文件头里,一字不差:

> 「Skills are the agent's procedural memory: they capture how to do a specific type of task based on proven experience. General memory (MEMORY.md, USER.md) is broad and declarative. Skills are narrow and actionable.」

翻译一下:技能是 Agent 的程序性记忆,捕获的是"基于已验证经验,如何做某类特定任务"。通用记忆(MEMORY.md、USER.md)是宽泛的、陈述性的。技能是狭窄的、可执行的。

记住这三类分工,后面所有的代码逻辑都是在围绕这三类记忆做"写入 / 压缩 / 流转 / 再注入"。

---

## 🏗 第二章:Hermes 的进化骨架 —— 四个时间尺度(2:30 – 4:00)

Hermes 把进化过程切成了四个时间尺度,你可以理解成四档变速箱:

```
高频 ────────────────────────────────────────► 低频
turn      post-turn      session-end      cross-session
(每轮)     (每回合结束)    (会话关闭)       (跨会话/跨Agent)
```

- **turn 级** —— 模型在对话中主动调 skill_manage / memory 工具。
- **post-turn 级** —— 每完成一个 user message → assistant 回复的完整回合,检查是否该触发 review。
- **session-end 级** —— 会话彻底关闭时,触发 on_session_end 钩子链。
- **cross-session 级** —— Skills Hub 同步、trajectory 落盘、insights 跨会话统计。

这四档各自独立触发、各自独立写入,组成一个多层过滤网。模型自己没记,回合结束计数器补一刀;回合没补上,session-end 的插件再捞一次;全都没捞上,trajectory 至少留了原始对话,未来 session_search 还能扒出来。

这种设计叫 **Defense in Depth(纵深防御)**—— 本来是安全领域的术语,Hermes 把它搬到了"记忆不丢失"这件事上。

---

## 🔍 第三章:一档一档拆(4:00 – 10:30)

### ⏱ 第一档:turn 级 —— 模型主动沉淀

这档的核心是系统提示词 + 工具 schema两条腿走路。

#### ① 原文引用 —— SKILLS_GUIDANCE

位置在 `agent/prompt_builder.py:164-171`,原文是:

> "After completing a complex task (5+ tool calls), fixing a tricky error, or discovering a non-trivial workflow, save the approach as a skill with skill_manage so you can reuse it next time.
> When using a skill and finding it inaccurate or incomplete, update it immediately with skill_manage."

翻译:完成复杂任务(5+ 次工具调用)、修复棘手错误、或发现非平凡工作流后,用 skill_manage 把方法存成 skill,下次复用。使用 skill 时发现不准或不完整,立刻更新。

#### ② 原文引用 —— MEMORY_GUIDANCE

位置在 `agent/prompt_builder.py:173-180`,原文是:

> "Update MEMORY.md when you learn something new about the user's preferences, environment, or workflow.
> Update USER.md when the user explicitly tells you something about themselves."

翻译:学到用户偏好、环境、工作流新信息时更新 MEMORY.md;用户主动告知个人信息时更新 USER.md。

#### ③ 工具 schema 里的暗线

`tools/skill_manager_tool.py` 里,skill_manage 的 description 里藏了一句:

> "Use this tool proactively after solving novel problems or discovering better workflows."

这不是装饰,是刻意给模型的 behavioral hint(行为暗示)。

**这一档的特点**:完全依赖模型自觉。模型觉得该存就存,觉得不该存就跳过。所以 Hermes 不能只靠这一档,必须有后面的兜底。

#### ④ turn 级的"优化时机图" —— 容易被误解的点

很多人会问:模型判断"要不要优化"是**一开始就判断**,还是**回答结束后才判断**?

答案是:**都不对。turn 级的优化是贯穿整轮的,不是单点事件。**

源码里有两个独立的环节,要分开看:

**环节一:行为暗示的"注入时机" —— 会话一开始就焊死**

`agent/prompt_builder.py:164-171` 的 `SKILLS_GUIDANCE`(就是前面引用过的那段英文)是**会话启动时一次性写进 system prompt 的**,整轮对话不再变化。这是为了保护 prompt cache。所以"告诉模型要沉淀经验"这件事,是在会话**最开始**就完成的。

**环节二:模型实际"执行优化"的时机 —— 贯穿每一次 tool call 迭代**

看 `run_agent.py:9084` 的主循环:

```python
while api_call_count < self.max_iterations:
    # 每一次循环迭代
    self.skill_evolution.on_agent_iteration(self.valid_tool_names)
    response = client.chat.completions.create(messages=messages, tools=tool_schemas)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call)
            # 模型在这里可能调 skill_manage / memory → 立即落盘!
```

**关键点**:`skill_manage` 和 `memory` 是**和业务工具同级的普通工具**,模型在当前轮次的**任何一次 tool call 迭代**里都能调它们。

**实际执行场景举例**:

假设用户发"帮我搭个 FastAPI 服务",模型可能这样展开:

```
Iteration 1: 调 file_write 建目录           ← 模型还没想起来记 skill
Iteration 2: 调 terminal 装依赖             ← 还没想起来
Iteration 3: 调 file_write 写代码           ← 还没想起来
Iteration 4: 调 terminal 跑测试,失败了
Iteration 5: 调 file_patch 修 bug
Iteration 6: 调 terminal 再跑,成功         ← 模型可能想:"这流程可复用"
Iteration 7: 调 skill_manage(action=create) ← 就在回答之前,写入 SKILL.md
Iteration 8: 返回最终 assistant message → 本轮结束
```

优化动作可能夹在业务动作**中间**,也可能就在**最后一次** tool call,**完全由模型自己判断时机**。

**turn 级的优化时机图**:

```
用户消息进来
     │
     ▼
【system prompt 已含 SKILLS_GUIDANCE】  ← 这是"会话开始时"就注入的暗示
     │                                     (整轮会话焊死不变)
     ▼
┌─ 循环 iteration 1 ─────────┐
│  模型决定调 tool            │
│  可能是业务工具              │
│  也可能是 skill_manage ←─── │ ← 优化可能在这里发生
└──────────────────────────┘
     │
     ▼
┌─ 循环 iteration 2 ─────────┐
│  ...                        │ ← 也可能在这里
└──────────────────────────┘
     │
     ▼
┌─ 循环 iteration N ─────────┐
│  模型输出最终回答            │ ← 到这一步就不会再调优化工具了
└──────────────────────────┘
     │
     ▼
post-turn 触发(第二档的事,完全不同的线程)
```

**和 post-turn 级的关键区别**:

| 维度 | turn 级 | post-turn 级 |
|------|---------|-------------|
| "判断要不要优化"发生在 | 主 agent 每次决策前,**模型自己决定** | **最终回答交付后**,代码计数器判断 |
| "执行优化动作"发生在 | 主 agent 的某次 tool call 迭代里,**和业务工具穿插** | 主流程结束后,**起后台线程**跑 review agent |
| 用户感知 | **用户可能看到**"正在调用 skill_manage"的活动提示 | **用户完全无感**(后台线程 + 静默失败) |

**一句话总结**:

- "告诉模型要优化"的暗示 → **会话一开始**就焊死在 system prompt 里
- "模型真的去优化"的动作 → 发生在**当前轮次的某次 tool call 迭代中**,不是等回答完才做
- "后台兜底审查" → 才是回答完以后做的,那是下一档 post-turn 级的事

所以 turn 级**不是单点事件**,是贯穿整轮的持续机会。

---

### ⏱ 第二档:post-turn 级 —— 回合结束自动审查

核心模块:`agent/skill_evolution.py` 里的 `SkillEvolutionManager`。

#### 触发条件(原文代码逻辑提炼):

```python
def should_trigger_review(self, turn_data) -> bool:
    # 条件 1:工具调用数 >= 阈值(默认 5)
    if turn_data.tool_call_count >= self.config.review_threshold:
        return True
    # 条件 2:出现错误后成功修复
    if turn_data.had_error_and_recovery:
        return True
    # 条件 3:使用了 skill 且用户反馈正面
    if turn_data.used_skill and turn_data.user_feedback_positive:
        return True
    return False
```

#### 触发后做什么?

启动一个 **review agent**(一个独立的 LLM 调用,用轻量模型),给它一段精心设计的提示词,让它判断:

1. 这轮对话有没有值得沉淀的新知识?
2. 如果有,是写成 skill(程序性记忆)还是更新 memory(陈述性记忆)?
3. 写出草稿,调 skill_manage 工具落盘。

#### 提示词骨架(原文提炼):

> "You are reviewing the just-completed conversation turn.
> Determine if any new procedural or declarative knowledge was gained.
> If yes, draft a SKILL.md or MEMORY.md update following the template.
> Be conservative: only capture knowledge that is likely to be reusable."

关键词:**Be conservative(保守原则)**。Hermes 宁可漏掉一些低价值片段,也不往 skill 库里灌垃圾。

---

### ⏱ 第三档:session-end 级 —— 会话关闭时的最后防线

核心模块:`agent/memory_manager.py` 里的 `on_session_end` 钩子。

#### 触发时机:

用户发 `/exit`、`/quit`,或 CLI 进程正常退出时。

#### 做什么?

1. **Context Compression(上下文压缩)** —— `agent/context_compressor.py` 把当前会话的完整对话轨迹压缩成一段 summary,写入 session 元数据。
2. **Memory Consolidation(记忆巩固)** —— 检查本轮会话中所有 turn 级的 skill/memory 操作,做一次去重 + 冲突检测。
3. **Trajectory Archiving(轨迹归档)** —— `agent/trajectory.py` 把完整对话轨迹(含工具调用、错误、恢复路径)落盘到 `~/.hermes/trajectories/` 目录,文件名带时间戳 + session_id。

#### 为什么需要这一档?

因为 turn 级和 post-turn 级都可能漏。turn 级靠模型自觉,post-turn 级靠计数器阈值。但有些高价值知识可能工具调用数不够 5 次,但质量极高(比如用户教了一个关键配置)。session-end 是最后一道网,确保至少轨迹完整保留,未来可回溯。

---

### ⏱ 第四档:cross-session 级 —— 跨会话/跨 Agent 的群体进化

核心模块:Skills Hub + `agent/insights.py` + `agent/error_classifier.py`。

#### Skills Hub

Hermes 有一个官方的 Skills Hub(远程仓库),存放社区贡献的高质量 skill。Agent 可以:

- `/skills search <keyword>` —— 搜索远程 skill
- `/skills install <name>` —— 安装到本地 `~/.hermes/skills/`
- `/skills list` —— 查看本地已安装 skill

#### Insights(洞察)

`agent/insights.py` 定期扫描本地 trajectory 和 skill 使用日志,统计:

- 哪些 skill 被调用最多?
- 哪些 skill 调用后用户给了正面/负面反馈?
- 哪些工具调用模式反复出错?

这些数据用于:

1. 本地 skill 自动排序(高频高优的排前面,模型更容易召回)
2. 未来向 Skills Hub 贡献时的效用分依据

#### Error Classifier(错误分类器)

`agent/error_classifier.py` 对工具调用失败做分类:

- **可规避错误** —— 比如参数格式不对、路径不存在。这类错误应该写进 skill 的 "Common Pitfalls" 章节。
- **不可规避错误** —— 比如 API 限流、网络超时。这类错误记录到 insights,但不写进 skill。
- **未知错误** —— 标记为待人工 review。

#### 这一档的意义

从个体经验上升到群体智慧。单个 Agent 踩的坑,通过 Hub 同步,变成所有 Agent 共享的知识。

---

## 📐 第四章:记忆系统的分层架构(10:30 – 13:00)

把前面四档串起来,Hermes 的记忆系统可以抽象成 5 层:

```
Layer 5: Skills Hub (远程社区知识库)
    ↕ 同步
Layer 4: Local Skills (~/.hermes/skills/*.md)
    ↕ 写入/更新
Layer 3: Memory Files (MEMORY.md, USER.md)
    ↕ 写入/更新
Layer 2: Session Context (当前会话上下文)
    ↕ 压缩/归档
Layer 1: Trajectory (原始对话轨迹)
```

**数据流向**:

1. 对话产生 → Layer 1 实时记录
2. turn 级模型主动沉淀 → Layer 3/4 写入
3. post-turn review agent → Layer 4 写入
4. session-end 压缩 → Layer 2 摘要
5. insights 统计 → Layer 4 排序优化
6. Skills Hub 同步 → Layer 5 ↔ Layer 4 双向流动

**关键设计原则**:每一层都是 Markdown 文件,人类可读、可审计、可手改、可 git 管理。Hermes 没有用向量数据库做 skill 存储,而是用文件系统 + 目录结构 + 文件头 metadata 做召回。

---

## ⚠️ 第五章:Hermes 的五个硬伤(13:00 – 18:30)

(语速加快,进入批判段)

Hermes 不是完美的。我翻了代码之后,找到五个我认为必须正视的问题。

### 问题 ①: Skill 效用评分缺失

Hermes 的 insights.py 统计了 skill 调用次数,但没有一个量化的 **效用分(utility score)**。

现状:调用 100 次的 skill 和调用 1 次的 skill,在列表里平铺,没有优先级。

解法:引入效用分公式:

```
utility_score = α * usage_count + β * positive_feedback_rate - γ * error_rate
```

α、β、γ 是权重,可配置。效用分高的 skill 在 prompt 里排前面,模型更容易召回。

### 问题 ②: Skill 冲突检测不足

两个 skill 可能描述相似但操作矛盾。Hermes 现在的冲突检测只靠文件名和 description 的字符串匹配,精度很低。

解法:引入轻量 embedding 模型(比如 `text-embedding-3-small`),对 skill description 做向量化,计算余弦相似度。相似度 > 0.85 且操作指令矛盾时,标记为候选冲突,提示用户 review。

### 问题 ③: Memory 文件无限增长

MEMORY.md 和 USER.md 没有大小限制。长期运行后可能膨胀到几十 KB,每次注入上下文都吃 token。

Hermes 有 context_compressor,但它是针对 session context 的,不是针对 memory 文件的。

解法:给 memory 文件加 **TTL(生存时间)** 机制:

- 每条记忆记录带一个 `last_accessed` 时间戳
- 超过 30 天未访问的记录,自动归档到 `MEMORY.archived.md`
- 归档记录不注入默认上下文,但可通过 session_search 检索

### 问题 ④: Skill Review 缺乏沙盒预演

review agent 写 skill 时,没有验证环节。它写的 skill 可能包含危险命令或错误参数,直接落盘后下次调用就可能出问题。

解法:加两层 guard:

- **Level 1:静态扫描(Hermes 已有)**—— 扫描正则 / 敏感 token / 危险命令。
- **Level 2:动态沙盒预演(Hermes 缺失)**—— 第一次 skill_view 前,在一个 mock tool 环境(所有工具返回假数据)里,让 review agent 模拟执行 skill,观察它会不会触发敏感操作模式。这类似蜜罐。

### 问题 ⑤: Trajectory 与 Skill 之间没有自动提炼管道

Hermes 现在的逻辑是:trajectory 被动落盘,未来靠 session_search 被动检索。但高价值的 trajectory 片段(比如用户反复纠正的那几轮)其实应该主动提炼成 skill 素材。

解法:离线跑一个 **Trajectory Distillation Job**,定期扫描 trajectories:

- 用 insights.py 找出错误率最高的工具调用模式
- 用 error_classifier.py 找出可规避的失败类型
- 把这些 pattern 作为候选 skill 种子,丢进后台 review agent 的提示词上下文里
- review agent 基于这些"数据驱动的候选点"去写 skill,而不是空手从对话里挖

这样 skill 的生成从对话触发升级为数据驱动。

---

## 🔬 第六章:源码层面的澄清 —— 三者到底重不重复?(18:30 – 20:30)

(语速放慢,进入答疑段)

讲到这里,有人会问:Hermes 又有 memory 工具,又有 MemoryManager,又有 SkillEvolutionManager,功能听起来都是"记东西",这不是重复吗?只保留一个行不行?

我去翻了源码,把它们的真实关系摆出来。

### 三个组件的职责矩阵

| # | 文件 | 角色定位 | 触发者 | 存储介质 |
|---|------|----------|--------|----------|
| A | `tools/memory_tool.py` MemoryStore | 本地文件写手 | 模型主动调 `memory` 工具 | 本地 MEMORY.md / USER.md |
| B | `agent/memory_manager.py` MemoryManager | 外部 provider 编排层 | 框架每轮自动调钩子 | 外部服务(Honcho/Hindsight/Mem0) |
| C | `agent/skill_evolution.py` SkillEvolutionManager | 后台 review 调度器 | 计数器达阈值自动触发 | 不直接写,而是起新 agent 调 A 和 skill_manage |

注意 C 的位置 —— 它**自己不存东西**,它的工作是"在合适的时候,起一个轻量后台 agent,让那个 agent 去调 A 和 skill_manage"。所以 C 是"调度",A 是"执行",B 是"外挂"。三者层次不同,根本不在一个抽象层面。

### 它们会在同一时刻并发吗?会

看 `run_agent.py:11801-11826` 的代码,每轮结束后是这么跑的:

```python
# 1. 判断是否该触发 skill review (C)
_should_review_skills = self.skill_evolution.should_review_after_turn(...)

# 2. 外部 memory provider 同步 (B)
if self._memory_manager and final_response:
    self._memory_manager.sync_all(original_user_message, final_response)
    self._memory_manager.queue_prefetch_all(original_user_message)

# 3. 后台 review agent (C 真正起线程)
if final_response and (_should_review_memory or _should_review_skills):
    self._spawn_background_review(...)
```

B 和 C 是**同一时刻并发**跑的。但因为 B 写外部服务、C 写本地文件,目标完全正交,不会互相打架。

### 缺一个会怎样?

我把三种"砍掉一个"的场景都推演了一遍。

**只保留 A,砍掉 B 和 C**:进化能力大幅退化但不归零。模型只能靠自觉调 memory 工具,一忙起来忘了调,那一轮经验**直接消失**(没有 C 的兜底)。跨会话也只能靠 frozen snapshot 注入,没法做语义召回(没有 B)。相当于一个"被动笔记本"。

**只保留 B,砍掉 A 和 C**:看似高级(有向量召回),但有三个硬伤。第一,必须依赖外部服务,离线直接失能。第二,B 的 `on_memory_write` 这个桥接钩子失去触发源 —— 模型主动沉淀的入口被砍掉。第三,仍然是被动的,不会主动反思"这轮值不值得记"。相当于"实时录音员",啥都录,但不做摘录。

**只保留 C,砍掉 A 和 B**:**直接挂掉**。因为 C 的工作方式是"起后台 agent 调 memory 工具",memory 工具没了,后台 agent 调用直接报错,review 完全空转。

### 为什么必须三个都要?

因为它们覆盖的是**三个完全不重叠的失败模式**:

| 失败模式 | 谁兜底 |
|----------|--------|
| 模型当轮太忙,忘了沉淀 | C —— 计数器到点起后台 review |
| 本地文件无法做语义召回 | B —— 外部 provider 向量检索 |
| 外部服务挂了/没配 key | A —— 本地 markdown 始终可用 |

这就是真正的 **Defense in Depth** —— 不是三个都做同一件事,而是**每个负责一个失败场景的兜底**。

### Hermes 的最小可用配置

源码里(`run_agent.py:1322-1373`)默认行为是:

- A 强制开启(`skip_memory=True` 才关,只用于测试)
- B 可选(默认无外部 provider,需要在 config 里显式配置 `memory.provider`)
- C 默认开启,可通过 `creation_nudge_interval=0` 关闭

所以**最小可用进化配置就是只保留 A**。但只靠 A 就是"全靠模型自律",效率非常低。B 和 C 都是增强项,缺一个进化体系都会跛脚。

### 还有一处讲稿需要修正

第三章第三档我说 `on_session_end` 在 `agent/memory_manager.py`,这只对了一半。源码里 `MemoryManager.on_session_end` 只处理外部 provider 的清理(刷缓冲、断连接)。**真正的 session-end 逻辑散在多处**:

- MemoryManager 管外部 provider 关闭
- `agent/trajectory.py` 管轨迹归档
- `tools/skills_sync.py` 管 Skills Hub 同步
- `hermes_cli/plugins` 管插件钩子链

而且 `run_agent.py` 注释里写得很清楚:`run_conversation` 每轮都会跑一遍 on_session_end 检查,**真正的会话级别清理是 CLI 的 atexit 和 gateway 的 session 过期触发的**,不是单个 manager 能搞定的。

---

### 补充:turn 级和 post-turn 级到底是不是"同一个方法在不同时机调"?

这是一个非常容易被误解的点,单独拉出来讲清楚。

**表面看**:是的,两者都调同一组工具(`memory` / `skill_manage`),最终都写同一批 Markdown 文件(`MEMORY.md` / `USER.md` / `SKILL.md`),没有数据库表参与。

**但深层有四个关键区别**,决定了这不是"冗余",而是有价值的纵深防御:

| 维度 | turn 级(主 agent) | post-turn 级(后台 review agent) |
|------|-------------------|-------------------------------|
| 调用者身份 | 当前正在解决用户问题的主 agent | 一个**全新的 AIAgent 实例**(`skill_evolution.py:138`) |
| 系统提示词 | 业务任务提示词 | 专门的 `COMBINED_REVIEW_PROMPT`(`skill_evolution.py:51`) |
| max_iterations | 默认 90 | **8**(轻量限流) |
| 看到的上下文 | 实时对话当下 | **完整对话快照** |
| 调用动机 | "顺手记一笔",可能因专注业务忘掉 | **唯一任务就是审查**,不会被分散 |
| 决策视角 | 参与者视角,容易高估当前细节 | 旁观者视角,能看到用户反复纠正的真信号 |
| 失败后果 | 用户任务跑不完(高成本) | 静默吞掉(`skill_evolution.py:157`),用户无感 |
| 执行线程 | 主线程 | 后台 daemon 线程 |

**一个程序员都懂的类比**:

- turn 级 = 程序员一边写代码一边随手记 commit message
- post-turn 级 = 写完一个 PR 后**专门让 reviewer 复盘整段改动**,决定要不要更新 changelog

最终都改同一份文档,但两者的**视角、动机、可靠性级别完全不同**。

**这才是 Defense in Depth 的真正含义**:

- 不是"同一件事做两遍"
- 而是**用两种完全不同的方式去触发同一个动作**
- 任何一种方式失败时,另一种还能补上
- 两种方式的"盲区"互补 —— turn 级看不到的跨轮模式,post-turn 级能看到;post-turn 级阈值没到的细节,turn 级能即时记

如果两者"完全相同",那就是冗余浪费;正因为**触发条件、上下文、视角不同**,才构成了有价值的纵深防御。

**另外补充一个易混淆点**:turn 和 post-turn 操作的**不是数据库表**。Hermes 的 `hermes_state.py` 里确实有 SQLite 表(`sessions` / `messages` / `messages_fts`),但那是给 `/resume` 恢复会话和 `session_search` 全文搜索用的,**和 memory / skill 进化完全无关**。记忆系统走的全是 Markdown 文件 + fcntl 文件锁 + 原子重命名,没有数据库参与。

### 再补充:session-end 和前两档的关系 —— 性质完全不同

有人会继续问:那 session-end(第三档)和前面 turn / post-turn 是不是也是"同方法不同时机"?

**答案:不是。这是两个完全不同性质的关系。**

| 对比 | 是否调同一组工具 | 是否操作同一份数据 | 本质 |
|------|----------------|-------------------|------|
| turn vs post-turn | **是**(都调 `memory` / `skill_manage`) | **是**(都写 MEMORY.md / SKILL.md) | 同动作不同触发方式,纵深防御 |
| post-turn vs session-end | **否** | **否**(写完全不同的目标) | **完全不同的清理流程** |

#### session-end 实际触发了什么?源码扒一遍

看 `run_agent.py:3461-3478` 和 `cli.py:709-712` 的注释:

> "Shut down memory provider (on_session_end + shutdown_all) at actual session boundary — NOT per-turn inside run_conversation()."

session-end 触发链涉及**至少 5 个完全不同的目标**:

| # | 触发的方法 | 写入/操作的目标 | 跟 memory 工具有关系吗 |
|---|-----------|---------------|-------------------|
| 1 | `MemoryManager.on_session_end()` | 外部 provider 刷缓冲到云端 + **关连接** | 无 |
| 2 | `MemoryManager.shutdown_all()` | 关线程池、断网络连接 | 无 |
| 3 | `ContextCompressor.on_session_end()` | 把整段对话压缩成 summary 写进 SQLite 的 `sessions` 表 | 无 |
| 4 | `save_trajectory()`(`agent/trajectory.py:30`) | 把对话历史以 ShareGPT 格式 append 到 `trajectory_samples.jsonl` | 无 |
| 5 | 插件钩子 `on_session_end`(`hermes_cli/plugins.py:62`) | 各插件自定义清理 | 无 |

**注意**:session-end **根本不再调 `memory` 或 `skill_manage` 工具**。它做的是"会话级别的资源回收 + 外部服务收尾 + 长期归档",跟"沉淀经验"是两个层次的事。

#### 为什么两档性质完全不同?

因为**它们解决的问题根本不在一个层级**:

```
┌─────────────────────────────────────────┐
│  turn / post-turn:                       │  解决"经验沉淀"
│  - 模型/review agent 调工具              │  → 写 markdown 文件
│  - 写 MEMORY.md / SKILL.md              │  → 同一动作两种触发
└─────────────────────────────────────────┘
              ↑ 高频,每轮发生

              ↓ 低频,只在会话真正结束时触发一次

┌─────────────────────────────────────────┐
│  session-end:                            │  解决"资源清理 + 长期归档"
│  - 刷外部服务缓冲区                      │  → 不同的目标
│  - 关网络连接                            │  → 不同的方法
│  - 把对话写 trajectory.jsonl            │  → 不同的存储格式
│  - 把 summary 写 SQLite                 │
└─────────────────────────────────────────┘
```

session-end 不是"在更晚的时候做一遍同样的事",而是**回答一个完全不同的问题**:会话结束了,那些"还没来得及落地的东西"和"必须释放的资源"怎么办?

#### 一个更精确的类比

如果 turn 和 post-turn 是"程序员随手记 commit 和 reviewer 复盘 PR"(同一份 changelog 的两种触发),

那 session-end 就是**"项目结束时的归档动作"** —— 把代码打 tag、关掉 CI 流水线、把日志归档到 S3、把临时分支删掉。**根本不再写 changelog 了**,做的是完全不同的事。

#### 结论

**只有 turn 和 post-turn 是"同方法不同时机"**,因为它们都是"经验沉淀"这件事的不同触发方式。

**session-end 跟它们俩都不是同类**,它是"会话生命周期收尾",目标、方法、数据全部不同。讲稿第二章把它们一起放进"四档变速箱",是从**触发频率**这个维度看的统一性,但从**做的事情**来看,session-end 是独立的一档。

---

## 💭 第七章:我的思考(20:30 – 21:30)

(语速放慢,进入总结段)

把五个问题讲完,再往上抽一层,聊三个我觉得更本质的事。

### 思考一: Agent 的「进化」本质上是外化认知

LLM 本身是无状态的(stateless)。每次 API 调用,它都是一个全新的大脑。所谓 Agent 的"进化",本质是在模型之外维护一个状态机 + 知识库,然后每次调用前把相关状态注入上下文。

Hermes 的贡献是:把这个外化过程做得非常工程化、非常分层、非常有章法。三类记忆、四个时间尺度、多层过滤网、对外开放 hook —— 这套架构可以原样照搬到任何 LLM 之上,换 GPT、换 Claude、换 Qwen 都不影响。

**这是一个重要信号:Agent 竞争的下半场,拼的不是模型,而是记忆工程。**

### 思考二: 「文档型进化」的天花板在哪?

Hermes 当前所有进化产物都是 Markdown。这有优点 —— 人类可读、可审计、可手改、可 git 管理。但也有明显的天花板:

- **无类型** —— 没办法做结构化检索
- **无 embedding** —— 只能按目录 + 描述召回,精度有限
- **无版本图** —— 一个 skill 被 patch 多次,分叉、合并、回滚都不支持
- **无效用指标** —— 前面讲过的问题

下一代 Agent 记忆系统,我认为会走向:

> **Markdown(人类可读层) + Typed AST(机器可推理层) + Vector Index(召回层) + Usage Graph(效用层)**

四层合一,同一份知识四种表达,按场景取用。

### 思考三: 从「个体进化」到「群体进化」

Skills Hub 打开了一个激动人心的可能性 —— Agent 可以彼此学习。

但 Hub 目前是人工策展的。真正的跃迁应该是:

- Agent A 发现 skill X 有效,自动在本地打分
- 打分够高、使用够多,自动推送到 hub(带匿名化)
- 其他 Agent 拉取时,按本地上下文相似度 + hub 效用分混合排序
- Agent B 拉取后二次改进,再回流

这就是演化论层面的物种演化 —— **变异(个体 patch)+ 选择(效用评分)+ 遗传(hub 同步)**。

一旦这条链路跑通,每一个独立部署的 Agent 实例,都是物种进化的一个基因样本。整个 Hermes 生态的知识,会以天为单位累积增长,而不是等下一代模型 release。

**这才是我认为真正意义上的"自我进化"。不是某个 Agent 变聪明,而是整个 Agent 物种变聪明。**

---

## 🎬 结尾(21:30 – 22:00)

(收束)

Hermes 不是一个完美的项目,它有耦合、有粗糙、有缺口。但它认真对待了一件事 —— Agent 的经验到底应该怎么沉淀、怎么流动、怎么复用。

这一点,比市面上 90% 的"炫酷多 Agent 框架"都更有价值。

如果你在做 Agent,强烈推荐把 Hermes 的 `agent/skill_evolution.py`、`agent/memory_manager.py`、`agent/prompt_builder.py`、`tools/skill_manager_tool.py` 四个文件精读三遍。里面每一段提示词、每一个 guard 条件、每一个计数器,都是被反复打磨过的工程智慧。

今天就到这里。下期我把我自己实现的完整复用框架也开源出来,带大家一行一行撸。我是 XXX,关注我,不迷路。

---

## 📋 讲稿使用补充

- **章节切分点**:每章结束可以留 1 秒停顿 + 章节字幕,方便观众断点观看
- **代码镜头**:每当引用原文提示词时,同步切屏显示英文原文 + 中文译文双列
- **架构图**:第二章「四档变速箱」、第三章「时间尺度轴」、第五章「Layer 1-5 分层」建议提前画好 SVG 动画
- **高亮句(字幕放大 + 标色)**:
  - 「Agent 竞争的下半场,拼的不是模型,而是记忆工程」
  - 「同一份知识四种表达」
  - 「真正意义上的自我进化,是整个 Agent 物种变聪明」
- **时长控制**:按口播正常语速约 21 分钟,可压缩到 18 分钟(加速第三章代码部分)或扩展到 25 分钟(第七章加案例)
