"""Offline stub language model for tests (no network, no provider)."""

from typing import Callable

from dspy.clients.engines.dummy_engine import AsyncDummyEngine, DummyEngine
from dspy.lm15 import Message, Response, TextPart, Usage
from dspy.utils.dummies import DummyLM


class _ScriptedEngine(DummyEngine):
    def _complete_messages(self, messages):
        text = self.owner.respond(messages)
        self.owner.calls.append(messages)
        return Response(id=None, model="stub", message=Message.assistant([TextPart(text)]),
                        finish_reason="stop", usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))


class StubLM(DummyLM):
    """DummyLM whose reply is computed by ``respond(messages) -> str``."""

    def __init__(self, respond: Callable[[list], str], name: str = "stub"):
        super().__init__(answers=[])
        self.respond = respond
        self.stub_name = name
        self.calls: list = []
        self._engine_spec = _ScriptedEngine(self)
        self._async_engine_spec = AsyncDummyEngine(self._engine_spec)

    def copy(self, **kwargs):  # GEPA/DSPy may copy LMs; keep the script and call log
        return self


def chat_fields(**fields) -> str:
    """Format output fields the way dspy.ChatAdapter parses them."""
    out = "".join(f"[[ ## {k} ## ]]\n{v}\n\n" for k, v in fields.items())
    return out + "[[ ## completed ## ]]\n"


def system_text(messages) -> str:
    return "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")


def all_text(messages) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)
