"""--metric module:function (offline)."""

import json

import dspy
import pytest
from click.testing import CliRunner

from evolution.core.metrics import as_float_metric, as_gepa_metric, load_metric, score_of
from evolution.skills import evolve_skill
from tests.skills.test_gepa_wiring import (  # noqa: F401  (fixtures)
    EVAL_MODEL, IMPROVED, REFLECTION_MODEL, _reflection_reply, _task_reply, fixture_repo, no_network,
)
from tests.stub_lm import StubLM

METRIC_FILE = '''
import dspy
CALLS = []
def one_line(gold, pred, trace=None, pred_name=None, pred_trace=None):
    CALLS.append(pred_name)
    ok = pred.output.strip() == "answer in one line"
    return dspy.Prediction(score=1.0 if ok else 0.0,
                           feedback="good" if ok else "Say exactly: answer in one line")
'''


@pytest.fixture
def metric_file(tmp_path):
    f = tmp_path / "my_metric.py"
    f.write_text(METRIC_FILE)
    return f


def test_load_from_module():
    fn = load_metric("evolution.core.fitness:skill_feedback_metric")
    assert fn.__name__ == "skill_feedback_metric"


def test_load_from_file(metric_file):
    fn = load_metric(f"{metric_file}:one_line")
    gold = dspy.Example(task_input="t", expected_behavior="x")
    assert fn(gold, dspy.Prediction(output="answer in one line")).score == 1.0


@pytest.mark.parametrize("spec", ["no_colon", "evolution.core.fitness:nope"])
def test_bad_spec_raises(spec):
    with pytest.raises((ValueError, AttributeError)):
        load_metric(spec)


def test_adapters_wrap_three_arg_float_metric():
    def three(gold, pred, trace=None):
        return 0.25
    g = as_gepa_metric(three)(None, None, None, "p", None)
    assert g.score == 0.25 and g.feedback
    assert as_float_metric(three)(None, None) == 0.25


def test_adapters_keep_five_arg_metric_and_extract_score():
    def five(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return {"score": 0.5}
    assert as_gepa_metric(five) is five
    assert as_float_metric(five)(None, None) == 0.5
    assert score_of(dspy.Prediction(score=0.75)) == 0.75


def test_cli_rejects_bad_metric_before_any_work(tmp_path):
    res = CliRunner().invoke(evolve_skill.main, ["--skill-path", str(tmp_path), "--metric", "x:y"])
    assert res.exit_code != 0


def test_gepa_uses_the_custom_metric(fixture_repo, metric_file, monkeypatch, no_network):
    task_lm, refl = StubLM(_task_reply), StubLM(_reflection_reply)
    monkeypatch.setattr(dspy, "LM", lambda m, *a, **k: refl if m == REFLECTION_MODEL else task_lm)
    monkeypatch.chdir(fixture_repo)
    evolve_skill.evolve(
        skill_name="demo-skill", iterations=2, eval_source="golden",
        dataset_path=str(fixture_repo / "ds"), optimizer_model=REFLECTION_MODEL, eval_model=EVAL_MODEL,
        hermes_repo=str(fixture_repo / "hermes-agent"), metric=f"{metric_file}:one_line",
    )
    metrics_json = next((fixture_repo / "output" / "demo-skill").rglob("metrics.json"))
    m = json.loads(metrics_json.read_text())
    # exact-match metric: baseline 0.0, evolved 1.0
    assert m["baseline_score"] == 0.0 and m["evolved_score"] == 1.0
