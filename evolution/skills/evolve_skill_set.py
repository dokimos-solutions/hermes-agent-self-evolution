"""Evolve a SET of skills together with GEPA.

Usage:
    python -m evolution.skills.evolve_skill_set \\
        --skill-path skills/producer --skill-path skills/consumer \\
        --runner my_runner.py:run --metric my_metric.py:score \\
        --dataset-path datasets/pair/ --max-metric-calls 20 \\
        --optimizer-model openai/some-model --optimizer-api-base http://127.0.0.1:8000/v1

The dataset directory holds frozen splits: train.jsonl, val.jsonl and
(optional) holdout.jsonl. Each row is a JSON object with at least
``task_input``; other keys are passed to the runner and metric unchanged.

The runner is ``run(candidate_dir, skill_files, inputs) -> dict`` (see
evolution.skills.skill_set). The metric may be a GEPA 5-arg metric or a
plain ``(gold, pred, trace) -> float``.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console

from evolution.core.lm import make_lm
from evolution.core.metrics import as_float_metric, as_gepa_metric, load_metric
from evolution.skills.evolve_skill import build_optimizer
from evolution.skills.skill_set import SkillSetModule, load_skill_set

console = Console()


def load_split(path: Path) -> list[dspy.Example]:
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        if "task_input" not in r:
            raise ValueError(f"{path}: every row needs 'task_input'")
    return [dspy.Example(**r).with_inputs("task_input") for r in rows]


def evolve_skill_set(
    skill_paths: list[str],
    runner: str,
    metric: str,
    dataset_path: str,
    max_metric_calls: int,
    optimizer_model: str,
    optimizer_api_base: Optional[str] = None,
    optimizer_api_key: Optional[str] = None,
    output_dir: str = "output/skill_sets",
    candidate_root: Optional[str] = None,
    reflection_lm=None,
    gepa_extra: Optional[dict] = None,
) -> dict:
    """Run GEPA over the skill set; write evolved skills and metrics; return metrics."""
    skills = load_skill_set([Path(p) for p in skill_paths])
    run_fn = load_metric(runner)
    user_metric = load_metric(metric)
    gepa_metric, float_metric = as_gepa_metric(user_metric), as_float_metric(user_metric)

    ds = Path(dataset_path)
    trainset, valset, holdout = (load_split(ds / f"{s}.jsonl") for s in ("train", "val", "holdout"))
    if not trainset or not valset:
        raise click.UsageError(f"{ds} needs non-empty train.jsonl and val.jsonl")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / ts
    cand_root = Path(candidate_root) if candidate_root else out / "candidates"
    student = SkillSetModule(skills, run_fn, candidate_root=cand_root)
    console.print(f"Skill set: {', '.join(student.skill_names)}")
    console.print(f"Split: {len(trainset)} train / {len(valset)} val / {len(holdout)} holdout")

    if reflection_lm is None:
        reflection_lm = make_lm(optimizer_model, api_base=optimizer_api_base, api_key=optimizer_api_key,
                                temperature=1.0, max_tokens=32000)
    opt = build_optimizer("gepa", metric=gepa_metric, reflection_lm=reflection_lm,
                          max_metric_calls=max_metric_calls, track_stats=True, **(gepa_extra or {}))
    start = time.time()
    best = opt.compile(student, trainset=trainset, valset=valset)
    elapsed = time.time() - start

    def mean_score(module, data):
        scores = [float_metric(ex, module(**ex.inputs())) for ex in data]
        return sum(scores) / len(scores) if scores else None

    baseline_hold = mean_score(student, holdout) if holdout else None
    evolved_hold = mean_score(best, holdout) if holdout else None

    written = best.write_skills(out / "evolved")
    student.write_skills(out / "baseline")
    res = getattr(best, "detailed_results", None)
    metrics = {
        "skills": student.skill_names,
        "optimizer": "gepa",
        "optimizer_model": optimizer_model,
        "max_metric_calls": max_metric_calls,
        "elapsed_seconds": elapsed,
        "val_aggregate_scores": list(res.val_aggregate_scores) if res else None,
        "best_idx": res.best_idx if res else None,
        "changed": {n: best.skill_texts()[n] != student.skill_texts()[n] for n in student.skill_names},
        "evolved_sha256": {n: w["sha256"] for n, w in written.items()},
        "holdout_baseline": baseline_hold,
        "holdout_evolved": evolved_hold,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    console.print(f"Output: {out}")
    return metrics


@click.command()
@click.option("--skill-path", "skill_paths", multiple=True, required=True, type=click.Path(exists=True),
              help="SKILL.md or skill directory; repeat once per skill in the set")
@click.option("--runner", required=True, help="Runner as module:function or file.py:function")
@click.option("--metric", required=True, help="Metric as module:function or file.py:function")
@click.option("--dataset-path", required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory with train.jsonl, val.jsonl[, holdout.jsonl]")
@click.option("--max-metric-calls", required=True, type=int, help="GEPA budget (total metric calls)")
@click.option("--optimizer-model", required=True, help="Reflection model (LiteLLM name)")
@click.option("--optimizer-api-base", default=None, envvar="EVOLUTION_OPTIMIZER_API_BASE")
@click.option("--optimizer-api-key", default=None, envvar="EVOLUTION_OPTIMIZER_API_KEY")
@click.option("--output-dir", default="output/skill_sets")
@click.option("--candidate-root", default=None, help="Where candidate skill dirs are written")
def main(skill_paths, runner, metric, dataset_path, max_metric_calls, optimizer_model,
         optimizer_api_base, optimizer_api_key, output_dir, candidate_root):
    """Evolve several skills together, scored by a real task runner."""
    if len(skill_paths) < 2:
        console.print("[yellow]Only one skill given; a set usually has two or more.[/yellow]")
    load_metric(runner), load_metric(metric)  # fail fast
    evolve_skill_set(list(skill_paths), runner, metric, dataset_path, max_metric_calls, optimizer_model,
                     optimizer_api_base, optimizer_api_key, output_dir, candidate_root)


if __name__ == "__main__":
    main()
