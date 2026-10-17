"""Pluggable metrics: ``--metric module:function`` or ``path/to/file.py:function``."""

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Callable

import dspy


def load_metric(spec: str) -> Callable:
    """Import a metric from ``package.module:function`` or ``file.py:function``."""
    if ":" not in spec:
        raise ValueError(f"--metric must look like module:function or file.py:function, got {spec!r}")
    target, func_name = spec.rsplit(":", 1)
    if target.endswith(".py") or "/" in target:
        path = Path(target).expanduser().resolve()
        mod_spec = importlib.util.spec_from_file_location(f"_evolution_metric_{path.stem}", path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot load metric file {path}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise AttributeError(f"{target} has no callable {func_name!r}")
    return fn


def score_of(result) -> float:
    """Extract a float score from a float, dict or dspy.Prediction metric result."""
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        return float(result["score"])
    return float(getattr(result, "score"))


def _accepts_five(fn: Callable) -> bool:
    try:
        inspect.signature(fn).bind(None, None, None, None, None)
        return True
    except TypeError:
        return False


def as_gepa_metric(fn: Callable) -> Callable:
    """Adapt any metric to GEPA's five-argument interface.

    A 2/3-argument metric is wrapped; its float result becomes the score and
    a short generic feedback string is attached.
    """
    if _accepts_five(fn):
        return fn

    def wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
        s = score_of(fn(gold, pred, trace))
        return dspy.Prediction(score=s, feedback=f"Score {s:.2f}.")

    wrapped.__name__ = getattr(fn, "__name__", "metric")
    return wrapped


def as_float_metric(fn: Callable) -> Callable:
    """Adapt any metric to ``(gold, pred, trace=None) -> float`` (MIPROv2, holdout)."""
    five = _accepts_five(fn)

    def wrapped(gold, pred, trace=None):
        r = fn(gold, pred, trace, None, None) if five else fn(gold, pred, trace)
        return score_of(r)

    wrapped.__name__ = getattr(fn, "__name__", "metric")
    return wrapped
