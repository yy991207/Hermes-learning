"""Skill evolution orchestration for Hermes Agent.

This module owns the self-improvement loop around skills:

- Track how long the agent has worked without maintaining a skill.
- Decide when a post-turn skill review should be spawned.
- Run the quiet background review agent that can call ``skill_manage``.

The actual skill file operations still live in ``tools.skill_manager_tool``.
This class deliberately stays at the orchestration layer.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SkillEvolutionManager:
    """Manage the Skill self-optimization lifecycle for one agent session."""

    DEFAULT_CREATION_NUDGE_INTERVAL = 10

    MEMORY_REVIEW_PROMPT = (
        "Review the conversation above and consider saving to memory if appropriate.\n\n"
        "Focus on:\n"
        "1. Has the user revealed things about themselves -- their persona, desires, "
        "preferences, or personal details worth remembering?\n"
        "2. Has the user expressed expectations about how you should behave, their work "
        "style, or ways they want you to operate?\n\n"
        "If something stands out, save it using the memory tool. "
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    SKILL_REVIEW_PROMPT = (
        "Review the conversation above and consider saving or updating a skill if appropriate.\n\n"
        "Focus on: was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome?\n\n"
        "If a relevant skill already exists, update it with what you learned. "
        "Otherwise, create a new skill if the approach is reusable.\n"
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    COMBINED_REVIEW_PROMPT = (
        "Review the conversation above and consider two things:\n\n"
        "**Memory**: Has the user revealed things about themselves -- their persona, "
        "desires, preferences, or personal details? Has the user expressed expectations "
        "about how you should behave, their work style, or ways they want you to operate? "
        "If so, save using the memory tool.\n\n"
        "**Skills**: Was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome? If a relevant skill "
        "already exists, update it. Otherwise, create a new one if the approach is reusable.\n\n"
        "Only act if there's something genuinely worth saving. "
        "If nothing stands out, just say 'Nothing to save.' and stop."
    )

    def __init__(self, creation_nudge_interval: int = DEFAULT_CREATION_NUDGE_INTERVAL):
        self.creation_nudge_interval = self._coerce_interval(creation_nudge_interval)
        self.iters_since_skill = 0

    @classmethod
    def from_config(cls, config: Dict[str, Any] | None) -> "SkillEvolutionManager":
        """Create a manager from the Hermes config dict."""
        skills_config = (config or {}).get("skills", {})
        if not isinstance(skills_config, dict):
            skills_config = {}
        interval = skills_config.get(
            "creation_nudge_interval",
            cls.DEFAULT_CREATION_NUDGE_INTERVAL,
        )
        return cls(interval)

    @staticmethod
    def _coerce_interval(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return SkillEvolutionManager.DEFAULT_CREATION_NUDGE_INTERVAL

    def on_agent_iteration(self, valid_tool_names: set[str]) -> None:
        """Count one LLM/tool-loop iteration toward the skill review trigger."""
        if self.creation_nudge_interval > 0 and "skill_manage" in valid_tool_names:
            self.iters_since_skill += 1

    def on_tool_invoked(self, function_name: str) -> None:
        """Reset skill-review cadence after the agent actually maintains a skill."""
        if function_name == "skill_manage":
            self.iters_since_skill = 0

    def should_review_after_turn(self, valid_tool_names: set[str]) -> bool:
        """Return True when a completed turn should spawn a skill review."""
        if (
            self.creation_nudge_interval > 0
            and self.iters_since_skill >= self.creation_nudge_interval
            and "skill_manage" in valid_tool_names
        ):
            self.iters_since_skill = 0
            return True
        return False

    def _select_review_prompt(self, review_memory: bool, review_skills: bool) -> str:
        if review_memory and review_skills:
            return self.COMBINED_REVIEW_PROMPT
        if review_memory:
            return self.MEMORY_REVIEW_PROMPT
        return self.SKILL_REVIEW_PROMPT

    def spawn_background_review(
        self,
        parent_agent: Any,
        messages_snapshot: List[Dict],
        *,
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """Spawn a quiet background agent to review memory and/or skill updates.

        The parent agent is passed in explicitly to avoid importing ``run_agent``
        from this module. That keeps the orchestration class independent while
        preserving the existing background-review behavior.
        """
        prompt = self._select_review_prompt(review_memory, review_skills)

        def _run_review() -> None:
            review_agent = None
            try:
                with open(os.devnull, "w", encoding="utf-8") as devnull, \
                     contextlib.redirect_stdout(devnull), \
                     contextlib.redirect_stderr(devnull):
                    review_agent = parent_agent.__class__(
                        model=parent_agent.model,
                        max_iterations=8,
                        quiet_mode=True,
                        platform=parent_agent.platform,
                        provider=parent_agent.provider,
                    )
                    review_agent._memory_store = parent_agent._memory_store
                    review_agent._memory_enabled = parent_agent._memory_enabled
                    review_agent._user_profile_enabled = parent_agent._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0

                    review_agent.run_conversation(
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                self._emit_review_summary(parent_agent, review_agent)
            except Exception as exc:
                logger.debug("Background memory/skill review failed: %s", exc)
            finally:
                if review_agent is not None:
                    try:
                        review_agent.close()
                    except Exception:
                        pass

        thread = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        thread.start()

    @staticmethod
    def _emit_review_summary(parent_agent: Any, review_agent: Any) -> None:
        """Display a compact summary for successful background review writes."""
        actions: list[str] = []
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
            lowered = message.lower()
            if "created" in lowered:
                actions.append(message)
            elif "updated" in lowered:
                actions.append(message)
            elif "added" in lowered or (target and "add" in lowered):
                actions.append(f"{SkillEvolutionManager._target_label(target)} updated")
            elif "Entry added" in message:
                actions.append(f"{SkillEvolutionManager._target_label(target)} updated")
            elif "removed" in lowered or "replaced" in lowered:
                actions.append(f"{SkillEvolutionManager._target_label(target)} updated")

        if not actions:
            return

        summary = " | ".join(dict.fromkeys(actions))
        parent_agent._safe_print(f"  {summary}")
        callback = parent_agent.background_review_callback
        if callback:
            try:
                callback(summary)
            except Exception:
                pass

    @staticmethod
    def _target_label(target: str) -> str:
        if target == "memory":
            return "Memory"
        if target == "user":
            return "User profile"
        return target
