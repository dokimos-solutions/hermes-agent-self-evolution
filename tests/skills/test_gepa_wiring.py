"""GEPA wiring tests (offline, stub LMs, no network).

These fail on upstream 0a929e3, where evolve():
  * builds dspy.GEPA(metric=..., max_steps=...) -> TypeError on DSPy 3.x,
  * catches it and silently runs MIPROv2 instead,
  * never passes --optimizer-model to any LM,
  * keeps the skill text outside the predictor instruction, so it never changes.
"""

import json
import socket

import dspy
import pytest

from evolution.core import fitness  # attribute access: module imports on upstream too
from evolution.skills import evolve_skill
from evolution.skills.skill_module import SkillModule
from tests.stub_lm import StubLM, chat_fields, system_text

REFLECTION_MODEL = "openai/reflection-sentinel"
EVAL_MODEL = "openai/eval-sentinel"
BASELINE_BODY = "# Demo\n\nDo the task."
IMPROVED = "IMPROVED: answer in one line"


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("test tried to open a network connection")
    monkeypatch.setattr(socket.socket, "connect", refuse)


def _task_reply(messages):
    good = IMPROVED in system_text(messages)
    return chat_fields(reasoning="stub", output="answer in one line" if good else "unrelated words")


def _reflection_reply(messages):
    # dspy's built-in proposer is a Signature with output field `new_instruction`,
    # parsed with JSONAdapter on DSPy 3.4.
    return json.dumps({"new_instruction": IMPROVED})


@pytest.fixture
def fixture_repo(tmp_path):
    skill_dir = tmp_path / "hermes-agent" / "skills" / "demo" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: demo-skill\ndescription: toy skill\n---\n\n{BASELINE_BODY}\n")
    ds = tmp_path / "ds"
    ds.mkdir()
    rows = [{"task_input": f"task {i}", "expected_behavior": "answer in one line"} for i in range(8)]
    (ds / "golden.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


class TestSkillModule:
    def test_skill_text_is_the_predictor_instruction(self):
        mod = SkillModule(BASELINE_BODY)
        (name, pred), = mod.named_predictors()
        assert pred.signature.instructions == BASELINE_BODY
        assert mod.skill_text == BASELINE_BODY

    def test_changing_the_instruction_changes_skill_text(self):
        mod = SkillModule(BASELINE_BODY)
        (_, pred), = mod.named_predictors()
        pred.signature = pred.signature.with_instructions("new body")
        assert mod.skill_text == "new body"


class TestFeedbackMetric:
    def test_accepts_gepa_five_args_and_returns_score_and_feedback(self):
        gold = dspy.Example(task_input="t", expected_behavior="answer in one line")
        res = fitness.skill_feedback_metric(gold, dspy.Prediction(output="answer"), None, "predictor.predict", None)
        assert 0.0 <= res.score <= 1.0
        assert res.score == fitness.skill_fitness_metric(gold, dspy.Prediction(output="answer"))
        assert "line" in res.feedback

    def test_empty_output_gets_zero_and_feedback(self):
        gold = dspy.Example(task_input="t", expected_behavior="x")
        res = fitness.skill_feedback_metric(gold, dspy.Prediction(output=""), None, None, None)
        assert res.score == 0.0 and "empty" in res.feedback


class TestBuildOptimizer:
    def test_gepa_gets_reflection_lm_and_budget(self):
        lm = StubLM(_reflection_reply)
        opt = evolve_skill.build_optimizer(
            "gepa", metric=fitness.skill_feedback_metric, reflection_lm=lm, max_metric_calls=7)
        assert isinstance(opt, dspy.GEPA)
        assert opt.reflection_lm is lm
        assert opt.max_metric_calls == 7

    def test_gepa_without_reflection_lm_fails_loudly(self):
        with pytest.raises(ValueError, match="reflection"):
            evolve_skill.build_optimizer("gepa", metric=fitness.skill_feedback_metric, max_metric_calls=5)

    def test_gepa_error_is_not_swallowed(self, monkeypatch):
        def boom(**kw):
            raise TypeError("GEPA construction failed")
        monkeypatch.setattr(dspy, "GEPA", boom)
        monkeypatch.setattr(dspy, "MIPROv2", lambda **kw: pytest.fail("silent MIPROv2 fallback"))
        with pytest.raises(TypeError, match="GEPA construction failed"):
            evolve_skill.build_optimizer(
                "gepa", metric=fitness.skill_feedback_metric, reflection_lm=object(), max_metric_calls=5)

    def test_unknown_optimizer_rejected(self):
        with pytest.raises(ValueError):
            evolve_skill.build_optimizer("simba", metric=fitness.skill_fitness_metric)


class TestEvolveEndToEnd:
    def test_gepa_runs_with_optimizer_model_and_changes_the_skill(self, fixture_repo, monkeypatch, no_network):
        task_lm = StubLM(_task_reply, "task")
        reflection_lm = StubLM(_reflection_reply, "reflection")
        built = []

        def fake_lm(model, *a, **k):
            built.append(model)
            return reflection_lm if model == REFLECTION_MODEL else task_lm

        monkeypatch.setattr(dspy, "LM", fake_lm)
        monkeypatch.setattr(dspy, "MIPROv2", lambda **kw: pytest.fail("silent MIPROv2 fallback"))
        monkeypatch.chdir(fixture_repo)

        evolve_skill.evolve(
            skill_name="demo-skill",
            iterations=2,
            eval_source="golden",
            dataset_path=str(fixture_repo / "ds"),
            optimizer_model=REFLECTION_MODEL,
            eval_model=EVAL_MODEL,
            hermes_repo=str(fixture_repo / "hermes-agent"),
        )

        assert REFLECTION_MODEL in built, "--optimizer-model must build the reflection LM"
        assert reflection_lm.calls, "GEPA must call the reflection LM"
        evolved = list((fixture_repo / "output" / "demo-skill").rglob("evolved_skill.md"))
        assert evolved, "constraints must pass and the evolved skill must be saved"
        text = evolved[0].read_text()
        assert IMPROVED in text and text.startswith("---\nname: demo-skill")
        metrics = json.loads(evolved[0].with_name("metrics.json").read_text())
        assert metrics["optimizer"] == "gepa"
        assert metrics["evolved_score"] > metrics["baseline_score"]
