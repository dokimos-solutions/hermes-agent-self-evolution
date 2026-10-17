"""Skill-set optimisation: toy two-skill set, deterministic runner, stub reflection LM.

The runner reads the candidate SKILL.md files from disk (so it sees exactly
what forward() wrote) and simulates a producer -> consumer handoff:

  producer body has INCLUDE_FIELDS -> handoff carries "files:"   (+0.5)
  producer body has BAD            -> handoff is empty           (breaks the consumer)
  consumer body has READ_FILES and handoff has "files:" -> done (+0.5)

The consumer can only score after the producer is fixed, so only the SET
score shows the handoff failure.
"""

import json
import socket

import dspy
import pytest

from evolution.skills import evolve_skill
from evolution.skills.skill_set import (
    SkillSetModule, load_skill_set, make_skill_set_metric, predictor_name, sha256_text,
)
from tests.stub_lm import StubLM

PRODUCER = "PRODUCER: write a handoff for the next step."
CONSUMER = "CONSUMER: finish the work from the handoff."


def write_skill(root, name, body):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: toy {name}\n---\n\n{body}\n")
    return d


@pytest.fixture
def toy_skills(tmp_path):
    return load_skill_set([write_skill(tmp_path / "src", "producer", PRODUCER),
                           write_skill(tmp_path / "src", "consumer", CONSUMER)])


def toy_runner(cand_dir, files, inputs):
    prod = files["producer"].read_text()
    cons = files["consumer"].read_text()
    if "BAD" in prod:
        handoff = ""
    elif "INCLUDE_FIELDS" in prod:
        handoff = f"summary: {inputs['task_input']}\nfiles: a.py"
    else:
        handoff = f"summary: {inputs['task_input']}"
    done = "READ_FILES" in cons and "files:" in handoff
    return {"output": "done using files" if done else "stuck",
            "per_skill": {"producer": handoff, "consumer": "done using files" if done else "stuck"}}


def toy_score(gold, pred):
    handoff = pred.per_skill["producer"]
    p_ok = "files:" in handoff
    c_ok = pred.output == "done using files"
    per = {
        "producer": "Handoff ok." if p_ok else "The handoff lacks the 'files:' field the next step needs.",
        "consumer": "Consumer ok." if c_ok else "The consumer did not use the files from the handoff.",
    }
    return 0.5 * p_ok + 0.5 * c_ok, f"set score {0.5 * p_ok + 0.5 * c_ok:.1f}", per


toy_metric = make_skill_set_metric(toy_score)


def examples(n):
    return [dspy.Example(task_input=f"t{i}").with_inputs("task_input") for i in range(n)]


class TestModule:
    def test_one_predictor_per_skill_with_body_as_instruction(self, toy_skills):
        mod = SkillSetModule(toy_skills, toy_runner)
        names = dict(mod.named_predictors())
        assert set(names) == {"skill_producer", "skill_consumer"}
        assert names["skill_producer"].signature.instructions == PRODUCER
        assert mod.skill_texts() == {"producer": PRODUCER, "consumer": CONSUMER}

    def test_forward_writes_current_instructions_and_hashes(self, toy_skills, tmp_path):
        mod = SkillSetModule(toy_skills, toy_runner, candidate_root=tmp_path / "cands")
        p = mod.predictor_for("producer")
        p.signature = p.signature.with_instructions(PRODUCER + " INCLUDE_FIELDS")
        pred = mod(task_input="x")
        written = (tmp_path / "cands").glob("cand-*/producer/SKILL.md")
        text = next(written).read_text()
        assert text.startswith("---\nname: producer") and "INCLUDE_FIELDS" in text
        assert pred.skill_hashes["producer"] == sha256_text(text)
        assert "files:" in pred.per_skill["producer"]

    def test_forward_records_one_trace_entry_per_skill(self, toy_skills):
        mod = SkillSetModule(toy_skills, toy_runner)
        with dspy.context(trace=[]):
            mod(task_input="x")
            trace = dspy.settings.trace
        assert [t[0] for t in trace] == [mod.predictor_for("producer"), mod.predictor_for("consumer")]
        assert trace[0][2].output.startswith("summary: x")

    def test_metric_gives_per_component_feedback(self, toy_skills):
        pred = SkillSetModule(toy_skills, toy_runner)(task_input="x")
        gold = examples(1)[0]
        assert "files:" in toy_metric(gold, pred, None, predictor_name("producer"), None).feedback
        assert "consumer did not" in toy_metric(gold, pred, None, predictor_name("consumer"), None).feedback
        assert toy_metric(gold, pred).score == 0.0

    def test_signatures_distinct_even_with_equal_bodies(self, tmp_path):
        skills = load_skill_set([write_skill(tmp_path, "a", "same"), write_skill(tmp_path, "b", "same")])
        mod = SkillSetModule(skills, toy_runner)
        assert not mod.skill_a.signature.equals(mod.skill_b.signature)


class TestGEPAOnSkillSet:
    def test_both_components_change_and_set_score_drives_selection(self, toy_skills, tmp_path, monkeypatch):
        monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("network"))
        proposals = []

        def reflect(messages):
            prompt = "\n".join(str(m.get("content", "")) for m in messages)
            if "PRODUCER:" in prompt:
                n = sum(1 for p in proposals if p.startswith("PRODUCER"))
                new = PRODUCER + (" BAD" if n == 0 else " INCLUDE_FIELDS")
            else:
                new = CONSUMER + " READ_FILES"
            proposals.append(new)
            return json.dumps({"new_instruction": new})

        reflection_lm = StubLM(reflect)
        student = SkillSetModule(toy_skills, toy_runner, candidate_root=tmp_path / "cands")
        opt = evolve_skill.build_optimizer(
            "gepa", metric=toy_metric, reflection_lm=reflection_lm, max_metric_calls=80,
            track_stats=True, use_merge=False, reflection_minibatch_size=2,
        )
        with dspy.context(lm=StubLM(lambda m: pytest.fail("task LM must not be called"))):
            best = opt.compile(student, trainset=examples(4), valset=examples(3))

        # the bad producer edit was proposed, but the set score rejected it
        assert any("BAD" in p for p in proposals)
        texts = best.skill_texts()
        assert "BAD" not in texts["producer"]
        # both components changed
        assert texts["producer"] == PRODUCER + " INCLUDE_FIELDS"
        assert texts["consumer"] == CONSUMER + " READ_FILES"
        # selection follows the set score
        res = best.detailed_results
        assert max(res.val_aggregate_scores) == 1.0
        assert res.best_idx == res.val_aggregate_scores.index(1.0)
        assert res.val_aggregate_scores[0] == 0.0  # baseline
        # the student is untouched
        assert student.skill_texts() == {"producer": PRODUCER, "consumer": CONSUMER}


def test_runner_base_exception_stops_gepa(toy_skills, tmp_path):
    """A budget stop raised by the runner as BaseException must end compile(), not be scored as failure."""
    class Stop(BaseException):
        pass

    calls = []

    def capped_runner(cand_dir, files, inputs):
        calls.append(1)
        if len(calls) > 4:
            raise Stop("token cap")
        return toy_runner(cand_dir, files, inputs)

    def reflect(messages):
        return json.dumps({"new_instruction": PRODUCER + " INCLUDE_FIELDS"})

    student = SkillSetModule(toy_skills, capped_runner, candidate_root=tmp_path / "c")
    opt = evolve_skill.build_optimizer("gepa", metric=toy_metric, reflection_lm=StubLM(reflect),
                                       max_metric_calls=200, use_merge=False)
    with pytest.raises(Stop):
        opt.compile(student, trainset=examples(4), valset=examples(3))
    # stopped at the first refused call (an already-started parallel call may also hit the stop)
    assert 5 <= len(calls) <= 7


def test_evolve_skill_set_end_to_end_writes_evolved_set(tmp_path, monkeypatch):
    from evolution.skills.evolve_skill_set import evolve_skill_set
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("network"))
    src = tmp_path / "src"
    paths = [write_skill(src, "producer", PRODUCER), write_skill(src, "consumer", CONSUMER)]
    ds = tmp_path / "ds"
    ds.mkdir()
    for split, n in (("train", 4), ("val", 3), ("holdout", 2)):
        (ds / f"{split}.jsonl").write_text("".join(json.dumps({"task_input": f"{split}{i}"}) + "\n"
                                                   for i in range(n)))

    def reflect(messages):
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        new = PRODUCER + " INCLUDE_FIELDS" if "PRODUCER:" in prompt else CONSUMER + " READ_FILES"
        return json.dumps({"new_instruction": new})

    m = evolve_skill_set(
        [str(p) for p in paths], "tests.skills.test_skill_set:toy_runner",
        "tests.skills.test_skill_set:toy_metric", str(ds), max_metric_calls=60,
        optimizer_model="openai/unused", output_dir=str(tmp_path / "out"), reflection_lm=StubLM(reflect),
        gepa_extra={"use_merge": False, "reflection_minibatch_size": 2},
    )
    assert m["changed"] == {"producer": True, "consumer": True}
    assert m["holdout_baseline"] == 0.0 and m["holdout_evolved"] == 1.0
    evolved = next((tmp_path / "out").glob("*/evolved/producer/SKILL.md")).read_text()
    assert "INCLUDE_FIELDS" in evolved and sha256_text(evolved) == m["evolved_sha256"]["producer"]
    assert (src / "producer" / "SKILL.md").read_text().count("INCLUDE_FIELDS") == 0  # source untouched
