# Hermes Agent Skill 自我优化机制解析

本文档说明 Hermes Agent 如何在工作过程中总结经验，并把经验沉淀为可复用的 Skill。这里的“自我优化”不是训练模型参数，而是把成功流程、踩坑经验、用户偏好中的程序性知识写入 Skill 文档，后续任务再加载这些 Skill 来改变 Agent 的行为。

## 1. 核心结论

Hermes 的 Skill 自我优化是一套“文档型程序记忆”机制：

```text
一次复杂任务
    |
    v
Agent 使用工具解决问题
    |
    v
发现可复用流程、坑点、用户纠正或非平凡经验
    |
    v
调用 skill_manage 创建或修补 Skill
    |
    v
写入 ~/.hermes/skills/<skill-name>/SKILL.md
    |
    v
下次构建系统提示词时扫描 Skill 索引
    |
    v
遇到相关任务时必须 skill_view 加载 Skill
    |
    v
按照 Skill 中沉淀的流程执行
```

它的本质不是“模型学会了”，而是“Agent 把经验写成了可检索、可编辑、可复用的操作手册”。

## 2. 参与模块

Skill 自我优化主要由以下文件协作完成：

```text
run_agent.py
    负责触发后台 review、统计复杂任务、调用工具循环

tools/skill_manager_tool.py
    负责真正创建、修改、删除、写入 Skill 文件

agent/prompt_builder.py
    负责把 Skill 索引和 Skill 维护规则注入系统提示词

tools/skills_tool.py
    负责 skills_list 和 skill_view，也就是列出和读取 Skill

agent/skill_commands.py
    负责把 Skill 暴露成 /skill-name 形式的 slash command

toolsets.py
    负责把 skills_list、skill_view、skill_manage 放进工具集
```

整体关系如下：

```text
                       +---------------------------+
                       |      agent/prompt_builder |
                       |  构建 Skill 索引和维护规则 |
                       +-------------+-------------+
                                     |
                                     v
+------------+       +---------------+---------------+       +--------------------------+
| 用户任务    | ----> | run_agent.py / AIAgent loop    | ----> | tools/skill_manager_tool |
|            |       | 决定何时总结、何时调用工具       |       | 写入或修补 Skill 文件      |
+------------+       +---------------+---------------+       +-------------+------------+
                                     |                                     |
                                     v                                     v
                       +-------------+-------------+       +---------------+-------------+
                       | tools/skills_tool          |       | ~/.hermes/skills/...        |
                       | skills_list / skill_view   |       | SKILL.md + references 等    |
                       +---------------------------+       +-----------------------------+
```

## 3. Skill 工具默认可用

Skill 能力默认属于 Hermes 核心工具集。

关键位置：`toolsets.py`

```python
_HERMES_CORE_TOOLS = [
    ...
    "skills_list", "skill_view", "skill_manage",
    ...
]
```

独立工具集定义：

```python
"skills": {
    "description": "Access, create, edit, and manage skill documents with specialized instructions and knowledge",
    "tools": ["skills_list", "skill_view", "skill_manage"],
    "includes": []
}
```

这意味着只要没有禁用 `skills` 工具集，Agent 就能读取 Skill，也能创建或更新 Skill。

## 4. 自我优化的两条路径

Hermes 有两种 Skill 自我优化路径。

### 4.1 任务中即时优化

如果 Agent 在使用某个 Skill 时发现它过时、不完整、命令错误或缺少坑点，系统提示词会要求它立刻修补。

关键位置：`agent/prompt_builder.py`

```python
SKILLS_GUIDANCE = (
    "After completing a complex task (5+ tool calls), fixing a tricky error, "
    "or discovering a non-trivial workflow, save the approach as a "
    "skill with skill_manage so you can reuse it next time.\n"
    "When using a skill and finding it outdated, incomplete, or wrong, "
    "patch it immediately with skill_manage(action='patch') — don't wait to be asked. "
    "Skills that aren't maintained become liabilities."
)
```

这段内容会被 `run_agent.py` 注入系统提示词：

```python
if "skill_manage" in self.valid_tool_names:
    tool_guidance.append(SKILLS_GUIDANCE)
```

逻辑含义：

```text
Agent 正在执行任务
    |
    v
加载了某个 Skill
    |
    v
发现 Skill 中的步骤、命令或坑点不准确
    |
    v
不要等用户要求
    |
    v
直接调用 skill_manage(action="patch")
```

### 4.2 任务后后台优化

如果主任务过程中没有主动写 Skill，`run_agent.py` 还会在任务结束后启动后台 review。

后台 review 的目标是：回看刚刚的完整对话，判断是否有值得沉淀的经验。

核心提示词位置：`run_agent.py`

```python
_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and consider saving or updating a skill if appropriate.\n\n"
    "Focus on: was a non-trivial approach used to complete a task that required trial "
    "and error, or changing course due to experiential findings along the way, or did "
    "the user expect or desire a different method or outcome?\n\n"
    "If a relevant skill already exists, update it with what you learned. "
    "Otherwise, create a new skill if the approach is reusable.\n"
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)
```

这段提示词让后台 Agent 判断：

- 是否完成了非平凡任务。
- 是否发生过试错。
- 是否中途改变方案并获得经验。
- 用户是否纠正过预期方法或结果。
- 是否已有相关 Skill 需要更新。
- 是否应该创建新的可复用 Skill。

## 5. 后台 Review 的触发条件

初始化时，`AIAgent` 会读取 Skill 创建提醒间隔。

关键位置：`run_agent.py`

```python
self._skill_nudge_interval = 10
try:
    skills_config = _agent_cfg.get("skills", {})
    self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
except Exception:
    pass
```

默认值是 `10`。也就是累计一定数量的工具调用迭代后，Agent 会考虑是否要总结 Skill。

配置形式可以是：

```yaml
skills:
  creation_nudge_interval: 10
```

每轮模型调用前，Agent 会累计 skill nudge 计数：

```python
if (self._skill_nudge_interval > 0
        and "skill_manage" in self.valid_tool_names):
    self._iters_since_skill += 1
```

当模型实际调用了 `skill_manage`，说明已经主动维护 Skill，计数器会重置：

```python
if function_name == "memory":
    self._turns_since_memory = 0
elif function_name == "skill_manage":
    self._iters_since_skill = 0
```

一轮对话结束时，检查是否达到阈值：

```python
_should_review_skills = False
if (self._skill_nudge_interval > 0
        and self._iters_since_skill >= self._skill_nudge_interval
        and "skill_manage" in self.valid_tool_names):
    _should_review_skills = True
    self._iters_since_skill = 0
```

如果满足条件，就启动后台 review：

```python
if final_response and not interrupted and (_should_review_memory or _should_review_skills):
    self._spawn_background_review(
        messages_snapshot=list(messages),
        review_memory=_should_review_memory,
        review_skills=_should_review_skills,
    )
```

可以把这段看成一个经验沉淀节流器：

```text
             +-------------------------+
             |  本轮任务产生工具调用？  |
             +------------+------------+
                          |
                          v
             +------------+------------+
             | _iters_since_skill += 1 |
             +------------+------------+
                          |
                          v
        +-----------------+-----------------+
        | 是否达到 creation_nudge_interval? |
        +-------------+---------------------+
                      |
           no         | yes
            |         v
            |   +-----+-----------------------+
            |   | 启动后台 Skill review Agent |
            |   +-------------+---------------+
            |                 |
            v                 v
      不做处理        判断是否 create / patch Skill
```

## 6. 后台 Review Agent 的运行方式

后台 review 入口是 `run_agent.py` 中的 `_spawn_background_review()`。

它会创建一个新的静默 Agent：

```python
review_agent = AIAgent(
    model=self.model,
    max_iterations=8,
    quiet_mode=True,
    platform=self.platform,
    provider=self.provider,
)
```

然后把刚才主对话的完整消息列表作为历史，把 review prompt 作为新的用户输入：

```python
review_agent.run_conversation(
    user_message=prompt,
    conversation_history=messages_snapshot,
)
```

这里的关键点是：

- 后台 Agent 使用同一个模型。
- 后台 Agent 静默运行，不打扰主任务响应。
- 后台 Agent 能看到刚才完整上下文。
- 后台 Agent 也有 `skill_manage` 工具。
- 如果它判断有可复用经验，就会调用 `skill_manage`。

抽象流程：

```text
主 Agent 完成任务
    |
    v
复制 messages_snapshot
    |
    v
启动 quiet review_agent
    |
    v
review_agent 读取完整任务历史
    |
    v
review_agent 收到 _SKILL_REVIEW_PROMPT
    |
    v
判断是否有可复用经验
    |
    +-------------------------+
    |                         |
    v                         v
Nothing to save        skill_manage(...)
                              |
                              v
                   写入 ~/.hermes/skills
```

## 7. Skill 如何落盘

真正写文件的是 `tools/skill_manager_tool.py`。

核心入口：

```python
def skill_manage(
    action: str,
    name: str,
    content: str = None,
    category: str = None,
    file_path: str = None,
    file_content: str = None,
    old_string: str = None,
    new_string: str = None,
    replace_all: bool = False,
) -> str:
```

它支持这些动作：

```text
create
    创建新的 Skill 目录和 SKILL.md

edit
    完整重写已有 Skill 的 SKILL.md

patch
    对 SKILL.md 或 supporting file 做局部替换

delete
    删除本地 Skill

write_file
    写入 references、templates、scripts、assets 下的辅助文件

remove_file
    删除辅助文件
```

调度逻辑：

```python
if action == "create":
    result = _create_skill(name, content, category)
elif action == "edit":
    result = _edit_skill(name, content)
elif action == "patch":
    result = _patch_skill(name, old_string, new_string, file_path, replace_all)
elif action == "delete":
    result = _delete_skill(name)
elif action == "write_file":
    result = _write_file(name, file_path, file_content)
elif action == "remove_file":
    result = _remove_file(name, file_path)
else:
    result = {"success": False, "error": "..."}
```

## 8. 创建 Skill 的核心逻辑

创建新 Skill 的函数是 `_create_skill()`。

逻辑如下：

```text
skill_manage(action="create")
    |
    v
校验 name 是否合法
    |
    v
校验 category 是否合法
    |
    v
校验 SKILL.md frontmatter
    |
    v
校验内容大小
    |
    v
检查是否已有同名 Skill
    |
    v
创建 ~/.hermes/skills/<category>/<name>/
    |
    v
原子写入 SKILL.md
    |
    v
安全扫描
    |
    +---------------------+
    |                     |
    v                     v
扫描通过              扫描失败
    |                     |
    v                     v
返回 success          删除目录并返回 error
```

核心代码：

```python
err = _validate_name(name)
if err:
    return {"success": False, "error": err}

err = _validate_frontmatter(content)
if err:
    return {"success": False, "error": err}

existing = _find_skill(name)
if existing:
    return {
        "success": False,
        "error": f"A skill named '{name}' already exists at {existing['path']}."
    }

skill_dir = _resolve_skill_dir(name, category)
skill_dir.mkdir(parents=True, exist_ok=True)

skill_md = skill_dir / "SKILL.md"
_atomic_write_text(skill_md, content)
```

Skill 写入位置是：

```text
~/.hermes/skills/<skill-name>/SKILL.md
```

如果带 category：

```text
~/.hermes/skills/<category>/<skill-name>/SKILL.md
```

## 9. Skill 文件格式校验

Hermes 要求 `SKILL.md` 必须有 YAML frontmatter。

格式类似：

```markdown
---
name: example-skill
description: Short description of when to use this skill.
---

# Example Skill

Use this skill when...
```

校验函数是 `_validate_frontmatter()`。

它会检查：

- 内容不能为空。
- 文件必须以 `---` 开始。
- YAML frontmatter 必须闭合。
- frontmatter 必须是 key-value mapping。
- 必须包含 `name`。
- 必须包含 `description`。
- description 不能超过长度限制。
- frontmatter 后必须有正文内容。

这保证 Agent 不能随便写一个无结构文档当 Skill。

## 10. 修改 Skill 的核心逻辑

### 10.1 完整重写

完整重写使用 `action="edit"`，内部函数是 `_edit_skill()`。

它适合大改，但风险更高，所以工具描述里更推荐 patch。

逻辑：

```text
读取新 content
    |
    v
校验 frontmatter
    |
    v
校验大小
    |
    v
找到已有 Skill
    |
    v
确认不是 external skill
    |
    v
备份原内容
    |
    v
原子写入新 SKILL.md
    |
    v
安全扫描
    |
    +-----------------------+
    |                       |
    v                       v
扫描通过                扫描失败
    |                       |
    v                       v
返回 updated           回滚原内容
```

### 10.2 局部修补

局部修补使用 `action="patch"`，内部函数是 `_patch_skill()`。

这是自我优化里最关键的动作，因为 Agent 可以把一次任务中发现的小坑直接写回已有 Skill。

核心代码：

```python
from tools.fuzzy_match import fuzzy_find_and_replace

new_content, match_count, _strategy, match_error = fuzzy_find_and_replace(
    content, old_string, new_string, replace_all
)
```

这里使用 fuzzy matching，不要求完全逐字符匹配。它可以容忍缩进、空格、换行等小差异，降低 Agent patch 失败率。

如果 patch 的是 `SKILL.md`，还会重新校验 frontmatter：

```python
if not file_path:
    err = _validate_frontmatter(new_content)
    if err:
        return {
            "success": False,
            "error": f"Patch would break SKILL.md structure: {err}",
        }
```

局部修补流程：

```text
skill_manage(action="patch")
    |
    v
找到已有 Skill
    |
    v
确认 Skill 在本地可修改目录
    |
    v
读取目标文件
    |
    v
fuzzy_find_and_replace(old_string, new_string)
    |
    +------------------------+
    |                        |
    v                        v
匹配失败                 匹配成功
    |                        |
    v                        v
返回 file_preview       校验结果大小
                             |
                             v
                    如果是 SKILL.md，校验 frontmatter
                             |
                             v
                         原子写入
                             |
                             v
                         安全扫描
                             |
                             v
                         返回 patched
```

## 11. Supporting Files 的作用

Skill 不一定只包含 `SKILL.md`。复杂知识可以拆到 supporting files。

允许写入的子目录：

```python
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}
```

典型结构：

```text
~/.hermes/skills/my-skill/
    SKILL.md
    references/
        api-guide.md
        pitfalls.md
    templates/
        report-template.md
    scripts/
        helper.py
    assets/
        example.json
```

设计意图：

- `SKILL.md` 保持短小，说明触发条件和核心流程。
- `references/` 存放长文档、API 说明、坑点列表。
- `templates/` 存放输出模板。
- `scripts/` 存放可复用脚本。
- `assets/` 存放示例配置、图片、结构化数据等。

Agent 后续可以用：

```python
skill_view(name="my-skill", file_path="references/api-guide.md")
```

按需加载辅助文件，避免每次都把全部资料塞进上下文。

## 12. 安全与边界控制

Skill 自我优化有多层保护。

### 12.1 名称限制

Skill 名称必须符合安全字符规则：

```python
VALID_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')
```

含义：

- 必须以小写字母或数字开头。
- 只能包含小写字母、数字、点、下划线、短横线。
- 长度不能超过限制。

### 12.2 路径限制

Supporting file 只能写入指定子目录：

```text
references/
templates/
scripts/
assets/
```

同时会检查路径穿越：

```python
if has_traversal_component(file_path):
    return "Path traversal ('..') is not allowed."
```

这可以防止模型试图写入 Skill 目录外的文件。

### 12.3 External Skill 只读

`skills.external_dirs` 中的 Skill 可以被读取，但不能被 Agent 修改。

相关判断：

```python
if not _is_local_skill(existing["path"]):
    return {"success": False, "error": "... external directory and cannot be modified ..."}
```

这让团队共享 Skill 或插件 Skill 保持只读，避免某个 Agent 随意改共享知识库。

### 12.4 安全扫描

Agent 写入或修改 Skill 后，会调用安全扫描：

```python
scan_error = _security_scan_skill(skill_dir)
if scan_error:
    shutil.rmtree(skill_dir, ignore_errors=True)
    return {"success": False, "error": scan_error}
```

修改已有 Skill 时，如果扫描失败，会回滚原内容。

### 12.5 原子写入

文件写入使用 `_atomic_write_text()`。

核心策略：

```text
写临时文件
    |
    v
flush 完成
    |
    v
os.replace(temp_path, target_path)
```

这样可以避免进程崩溃或中断时留下半个 `SKILL.md`。

## 13. 新 Skill 如何在下一次生效

Skill 生效不是靠改模型，而是靠下一次系统提示词扫描 Skill 索引。

核心函数：`agent/prompt_builder.py` 中的 `build_skills_system_prompt()`。

它会扫描：

```text
~/.hermes/skills/
skills.external_dirs
```

然后生成一个 Skill 索引，注入系统提示词。

关键提示内容：

```python
result = (
    "## Skills (mandatory)\n"
    "Before replying, scan the skills below. If a skill matches or is even partially relevant "
    "to your task, you MUST load it with skill_view(name) and follow its instructions. "
    ...
    "If a skill has issues, fix it with skill_manage(action='patch').\n"
    "After difficult/iterative tasks, offer to save as a skill. "
    "If a skill you loaded was missing steps, had wrong commands, or needed "
    "pitfalls you discovered, update it before finishing.\n"
)
```

也就是说，系统提示词会强制 Agent 做三件事：

- 先扫描可用 Skill。
- 如果相关，必须 `skill_view(name)` 加载。
- 如果 Skill 有问题，必须修补。

生效路径：

```text
skill_manage 成功写入文件
    |
    v
clear_skills_system_prompt_cache()
    |
    v
下一次构建 system prompt
    |
    v
build_skills_system_prompt 扫描 ~/.hermes/skills
    |
    v
系统提示词出现新 Skill 名称和 description
    |
    v
相关任务触发 skill_view
    |
    v
Skill 内容进入上下文
```

## 14. Skill 的读取方式

列出 Skill 的工具是 `skills_list()`：

```python
def skills_list(category: str = None, task_id: str = None) -> str:
```

它只返回轻量元数据：

```text
name
description
category
```

读取完整 Skill 的工具是 `skill_view()`：

```python
def skill_view(name: str, file_path: str = None, task_id: str = None) -> str:
```

它负责：

- 根据名称查找 `SKILL.md`。
- 支持 `category/skill-name` 形式。
- 支持插件 Skill。
- 支持 external skill dirs。
- 校验平台兼容。
- 检查是否被用户禁用。
- 防止 supporting file 路径穿越。
- 返回完整内容或指定 supporting file 内容。

渐进式披露设计如下：

```text
系统提示词中只放 Skill 索引
    |
    v
模型判断相关性
    |
    v
skill_view(name)
    |
    v
加载 SKILL.md
    |
    v
如需更多资料
    |
    v
skill_view(name, file_path="references/xxx.md")
```

这种设计避免系统提示词过大，同时又能在需要时加载完整专业知识。

## 15. Skill 也会变成 Slash Command

`agent/skill_commands.py` 会扫描本地 Skill，把它们暴露成 `/skill-name`。

扫描函数：

```python
def scan_skill_commands() -> Dict[str, Dict[str, Any]]:
```

它会读取每个 `SKILL.md` 的 frontmatter：

```text
name
description
platforms
```

然后生成命令：

```text
/github-code-review
/systematic-debugging
/my-custom-skill
```

用户显式调用 Skill 时，会构造一条特殊用户消息：

```python
activation_note = (
    f'[SYSTEM: The user has invoked the "{skill_name}" skill, indicating they want '
    "you to follow its instructions. The full skill content is loaded below.]"
)
```

注意：这里文本里写了 `[SYSTEM: ...]`，但实现上是作为用户消息注入，不是真正修改 system prompt。这样可以保持 prompt caching 稳定。

## 16. 完整闭环 ASCII 图

```text
 .---------------------------------------------------------------------.
 |                      Hermes Skill Self-Optimization                  |
 '---------------------------------------------------------------------'

        USER TASK
           |
           v
   +------------------+
   |  AIAgent Loop    |
   |  run_agent.py    |
   +--------+---------+
            |
            v
   +------------------+       no        +------------------------+
   | Complex / useful | --------------> | Continue normal answer |
   | workflow found?  |                 +------------------------+
   +--------+---------+
            |
           yes
            |
            v
   +---------------------------+
   | Direct skill_manage call? |
   +-----------+---------------+
               |
       yes     |      no
        |      |       |
        v      |       v
+---------------+  +-----------------------------+
| create/patch  |  | Count _iters_since_skill    |
| during task   |  | and maybe spawn review      |
+-------+-------+  +--------------+--------------+
        |                         |
        |                         v
        |              +--------------------------+
        |              | Quiet Review AIAgent     |
        |              | sees conversation history|
        |              +------------+-------------+
        |                           |
        |                           v
        |              +--------------------------+
        |              | Worth saving/updating?   |
        |              +------+-------------------+
        |                     |
        |             no      | yes
        |              |      v
        |              |  +-----------------------+
        |              |  | skill_manage(...)     |
        |              |  +-----------+-----------+
        |              |              |
        v              v              v
   +-------------------------------------------+
   | ~/.hermes/skills/<name>/SKILL.md          |
   | references/ templates/ scripts/ assets/   |
   +----------------------+--------------------+
                          |
                          v
   +-------------------------------------------+
   | clear_skills_system_prompt_cache()        |
   +----------------------+--------------------+
                          |
                          v
   +-------------------------------------------+
   | Next turn / next session builds index     |
   | build_skills_system_prompt()              |
   +----------------------+--------------------+
                          |
                          v
   +-------------------------------------------+
   | Relevant task => MUST skill_view(name)    |
   +----------------------+--------------------+
                          |
                          v
                    BETTER BEHAVIOR
```

## 17. 为什么说这是“进化”

它满足一个工程意义上的进化闭环：

```text
经验产生
    |
    v
经验选择
    |
    v
经验编码
    |
    v
经验存储
    |
    v
经验检索
    |
    v
经验复用
    |
    v
经验修正
```

对应到代码：

```text
经验产生
    run_agent.py 主任务执行过程

经验选择
    _SKILL_REVIEW_PROMPT + 模型判断

经验编码
    模型生成 SKILL.md 内容或 patch 内容

经验存储
    tools/skill_manager_tool.py 写入 ~/.hermes/skills

经验检索
    build_skills_system_prompt + skills_list + skill_view

经验复用
    系统提示词强制相关任务加载 Skill

经验修正
    skill_manage(action="patch")
```

所以，这里的“进化”是可审计、可回滚、可复制的知识库进化，而不是不可见的模型参数变化。

## 18. 这套机制的优点

- 可审计：所有结果都是 `SKILL.md` 和 supporting files。
- 可编辑：开发者可以直接打开 Skill 文档修改。
- 可迁移：复制 `~/.hermes/skills` 即可迁移经验。
- 可共享：通过 `skills.external_dirs` 可以读取团队共享 Skill。
- 可控：有名称校验、路径校验、frontmatter 校验、安全扫描、大小限制。
- 成本低：不需要微调模型或维护训练任务。
- 渐进加载：系统提示词只放索引，完整内容按需 `skill_view`。

## 19. 这套机制的限制

- 它依赖模型判断什么经验值得保存，不是完全确定性规则。
- `Confirm with user before creating/deleting` 主要是工具 schema 中的自然语言约束，代码层没有强制弹窗审批。
- 当前会话为了 prompt caching 通常不会频繁重建 system prompt，新 Skill 更自然地影响后续会话或后续重新构建提示词的场景。
- Skill 质量取决于模型生成的 `SKILL.md` 是否具体、可触发、可验证。
- 如果 Skill 写得太泛，会污染后续决策；如果写得太窄，复用价值有限。

## 20. 阅读代码建议顺序

建议按以下顺序阅读源码：

```text
1. toolsets.py
   确认 skills_list、skill_view、skill_manage 是核心工具

2. agent/prompt_builder.py
   理解系统提示词如何要求 Agent 使用和维护 Skill

3. run_agent.py
   理解主循环如何统计复杂任务、触发后台 review

4. tools/skill_manager_tool.py
   理解 Skill 如何创建、修补、落盘和安全校验

5. tools/skills_tool.py
   理解 Skill 如何被列出和加载

6. agent/skill_commands.py
   理解 Skill 如何变成 /skill-name 命令
```

## 21. 一句话总结

Hermes Agent 的自我优化机制是：在完成复杂任务后，由主 Agent 或后台 review Agent 把可复用经验写成 Skill；Skill 被保存为本地 Markdown 文件；后续系统提示词会强制 Agent 发现相关 Skill、加载 Skill、遵循 Skill，并在发现问题时继续修补 Skill。这是一种基于可维护文档的程序性记忆进化机制。
