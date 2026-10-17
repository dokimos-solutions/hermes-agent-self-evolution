"""Optimise a SET of skills together.

Skills often fail at their handoffs (one skill's output is the next one's
input), so scoring them one by one misses the failure. ``SkillSetModule``
holds one ``dspy.Predict`` per skill; each predictor's *instruction* is that
skill's body, so GEPA can rewrite every skill. On ``forward()`` the module
writes the current instructions to candidate SKILL.md files and hands the
directory to a user-supplied *runner*, which executes the real task (for
example a producer agent, then a consumer agent) and returns the result.

The runner does not call the predictors. To give GEPA a per-skill trace to
reflect on, ``forward()`` records one trace entry per skill:
``(predictor, {"task_input": ...}, Prediction(output=<what that skill's step
produced>))``. A set metric can then return per-component feedback chosen by
``pred_name`` (see ``make_skill_set_metric``).
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import dspy

from evolution.skills.skill_module import load_skill, reassemble_skill

# runner(candidate_dir, skill_files, inputs) -> dict with:
#   "output":    final result text (str)                    [required]
#   "per_skill": {skill_name: text produced by that step}   [optional]
#   any other keys are passed through on the Prediction (e.g. "checks").
Runner = Callable[[Path, dict, dict], dict]

PREFIX = "skill_"


def predictor_name(skill_name: str) -> str:
    """Stable attribute / pred_name for a skill (``skill_<identifier>``)."""
    return PREFIX + re.sub(r"\W", "_", skill_name)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SkillSetModule(dspy.Module):
    """One Predict component per skill; ``forward()`` runs the real task."""

    def __init__(
        self,
        skills: list[dict],
        runner: Runner,
        candidate_root: Optional[Path] = None,
        input_field: str = "task_input",
    ):
        super().__init__()
        if not skills:
            raise ValueError("SkillSetModule needs at least one skill")
        self._skills = {}  # skill name -> {"frontmatter", "dirname"}
        self._order = []
        for s in skills:
            name = s["name"] or Path(s["path"]).parent.name
            pname = predictor_name(name)
            if pname in self.__dict__:
                raise ValueError(f"duplicate skill name {name!r}")
            sig = dspy.Signature(
                {
                    input_field: dspy.InputField(desc="The task for this skill set"),
                    "output": dspy.OutputField(desc=f"What the '{name}' step produced"),
                },
                s["body"],
            )
            setattr(self, pname, dspy.Predict(sig))
            self._skills[name] = {
                "frontmatter": s["frontmatter"],
                "dirname": Path(s["path"]).parent.name if s.get("path") else name,
            }
            self._order.append(name)
        self._runner = runner
        self._candidate_root = Path(candidate_root) if candidate_root else None
        self._input_field = input_field

    # ── skill text access ──────────────────────────────────────────────
    @property
    def skill_names(self) -> list[str]:
        return list(self._order)

    def predictor_for(self, skill_name: str) -> dspy.Predict:
        return getattr(self, predictor_name(skill_name))

    def skill_texts(self) -> dict[str, str]:
        """Current (possibly optimised) body of every skill."""
        return {n: self.predictor_for(n).signature.instructions for n in self._order}

    def skill_files(self) -> dict[str, str]:
        """Full SKILL.md text (original frontmatter + current body) per skill."""
        return {n: reassemble_skill(self._skills[n]["frontmatter"], body)
                for n, body in self.skill_texts().items()}

    def write_skills(self, root: Path) -> dict[str, dict]:
        """Write every skill to ``root/<dirname>/SKILL.md``; return paths and hashes."""
        root = Path(root)
        out = {}
        for name, text in self.skill_files().items():
            d = root / self._skills[name]["dirname"]
            d.mkdir(parents=True, exist_ok=True)
            p = d / "SKILL.md"
            p.write_text(text)
            out[name] = {"path": p, "sha256": sha256_text(text)}
        return out

    # ── execution ─────────────────────────────────────────────────────
    def forward(self, **inputs: Any) -> dspy.Prediction:
        if self._candidate_root:
            self._candidate_root.mkdir(parents=True, exist_ok=True)
        cand_dir = Path(tempfile.mkdtemp(prefix="cand-", dir=self._candidate_root))  # unique per call
        written = self.write_skills(cand_dir)
        result = self._runner(cand_dir, {n: w["path"] for n, w in written.items()}, dict(inputs))
        if not isinstance(result, dict) or "output" not in result:
            raise TypeError("runner must return a dict with an 'output' key")

        per_skill = result.get("per_skill") or {}
        trace = dspy.settings.trace
        if trace is not None:
            task = inputs.get(self._input_field, "")
            for name in self._order:
                step_out = per_skill.get(name, result["output"])
                trace.append((self.predictor_for(name), {self._input_field: task},
                              dspy.Prediction(output=str(step_out))))

        extra = {k: v for k, v in result.items() if k not in ("output", "per_skill")}
        return dspy.Prediction(
            output=result["output"],
            per_skill=per_skill,
            skill_hashes={n: w["sha256"] for n, w in written.items()},
            candidate_dir=str(cand_dir),
            **extra,
        )


def load_skill_set(paths: list[Path]) -> list[dict]:
    """Load several SKILL.md files (or skill directories)."""
    skills = []
    for p in paths:
        p = Path(p)
        skills.append(load_skill(p / "SKILL.md" if p.is_dir() else p))
    return skills


def make_skill_set_metric(
    score_fn: Callable[[dspy.Example, dspy.Prediction], tuple[float, str, dict[str, str]]],
) -> Callable:
    """Build a GEPA metric from ``score_fn(gold, pred) -> (score, feedback, {skill: feedback})``.

    The set score is the same for every component. When GEPA asks for a
    specific component (``pred_name``), the feedback is that skill's own
    feedback plus the overall feedback, so the reflection model sees what its
    skill did wrong at the handoff.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        score, overall, per_skill = score_fn(gold, pred)
        feedback = overall
        if pred_name:
            for skill, fb in (per_skill or {}).items():
                if predictor_name(skill) == pred_name and fb:
                    feedback = f"{fb}\nOverall: {overall}"
                    break
        return dspy.Prediction(score=float(score), feedback=feedback)

    return metric
