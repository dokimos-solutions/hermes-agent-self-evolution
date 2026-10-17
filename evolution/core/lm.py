"""Language-model construction with optional OpenAI-compatible endpoints."""

from typing import Optional

import dspy

# LiteLLM refuses an empty key even when the endpoint does not check it
# (local proxies, vLLM, `hermes proxy`). Use an obvious placeholder.
PLACEHOLDER_API_KEY = "not-needed"


def make_lm(
    model: str,
    *,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> dspy.LM:
    """Build a dspy.LM, optionally pointed at an OpenAI-compatible endpoint.

    With ``api_base`` set, the model should use LiteLLM's ``openai/<name>``
    prefix and ``api_base`` should end in ``/v1``. A missing key becomes a
    non-empty placeholder.
    """
    if api_base:
        kwargs["api_base"] = api_base
        kwargs["api_key"] = api_key or PLACEHOLDER_API_KEY
    elif api_key:
        kwargs["api_key"] = api_key
    return dspy.LM(model, **kwargs)
