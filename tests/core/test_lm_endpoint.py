"""OpenAI-compatible endpoint option (offline)."""

import dspy
from click.testing import CliRunner

from evolution.core import lm as lm_mod
from evolution.skills import evolve_skill


def test_make_lm_sets_endpoint_and_placeholder_key():
    lm = lm_mod.make_lm("openai/some-model", api_base="http://127.0.0.1:9/v1")
    assert isinstance(lm, dspy.LM)
    assert lm.kwargs["api_base"] == "http://127.0.0.1:9/v1"
    assert lm.kwargs["api_key"] == lm_mod.PLACEHOLDER_API_KEY


def test_make_lm_keeps_explicit_key_and_extra_kwargs():
    lm = lm_mod.make_lm("openai/m", api_base="http://127.0.0.1:9/v1", api_key="k", temperature=1.0)
    assert lm.kwargs["api_key"] == "k"
    assert lm.kwargs["temperature"] == 1.0


def test_make_lm_without_endpoint_adds_nothing():
    lm = lm_mod.make_lm("openai/m")
    assert "api_base" not in lm.kwargs and "api_key" not in lm.kwargs


def test_cli_passes_endpoints_and_reads_env(monkeypatch):
    seen = {}
    monkeypatch.setattr(evolve_skill, "evolve", lambda **kw: seen.update(kw))
    res = CliRunner().invoke(
        evolve_skill.main,
        ["--skill", "x", "--optimizer-api-base", "http://127.0.0.1:9/v1"],
        env={"EVOLUTION_EVAL_API_BASE": "http://127.0.0.1:8/v1", "EVOLUTION_EVAL_API_KEY": "e"},
    )
    assert res.exit_code == 0, res.output
    assert seen["optimizer_api_base"] == "http://127.0.0.1:9/v1"
    assert seen["optimizer_api_key"] is None
    assert seen["eval_api_base"] == "http://127.0.0.1:8/v1"
    assert seen["eval_api_key"] == "e"
