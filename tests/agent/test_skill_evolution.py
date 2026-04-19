import json

from agent.skill_evolution import SkillEvolutionManager


def test_from_config_uses_creation_nudge_interval():
    manager = SkillEvolutionManager.from_config({
        "skills": {"creation_nudge_interval": 3},
    })

    assert manager.creation_nudge_interval == 3


def test_iteration_counter_only_counts_when_skill_manage_available():
    manager = SkillEvolutionManager(creation_nudge_interval=2)

    manager.on_agent_iteration({"web_search"})
    assert manager.iters_since_skill == 0

    manager.on_agent_iteration({"skill_manage"})
    assert manager.iters_since_skill == 1


def test_should_review_resets_after_threshold():
    manager = SkillEvolutionManager(creation_nudge_interval=2)

    manager.on_agent_iteration({"skill_manage"})
    assert manager.should_review_after_turn({"skill_manage"}) is False

    manager.on_agent_iteration({"skill_manage"})
    assert manager.should_review_after_turn({"skill_manage"}) is True
    assert manager.iters_since_skill == 0


def test_skill_manage_invocation_resets_counter():
    manager = SkillEvolutionManager(creation_nudge_interval=10)
    manager.iters_since_skill = 7

    manager.on_tool_invoked("web_search")
    assert manager.iters_since_skill == 7

    manager.on_tool_invoked("skill_manage")
    assert manager.iters_since_skill == 0


def test_emit_review_summary_collects_skill_and_memory_actions():
    class Parent:
        background_review_callback = None

        def __init__(self):
            self.printed = []

        def _safe_print(self, message):
            self.printed.append(message)

    class Review:
        _session_messages = [
            {
                "role": "tool",
                "content": json.dumps({
                    "success": True,
                    "message": "Skill 'debugging-flow' created.",
                }),
            },
            {
                "role": "tool",
                "content": json.dumps({
                    "success": True,
                    "message": "Entry added",
                    "target": "memory",
                }),
            },
            {
                "role": "tool",
                "content": json.dumps({
                    "success": False,
                    "message": "Ignored failure",
                }),
            },
        ]

    parent = Parent()
    SkillEvolutionManager._emit_review_summary(parent, Review())

    assert parent.printed == [
        "  Skill 'debugging-flow' created. | Memory updated",
    ]
