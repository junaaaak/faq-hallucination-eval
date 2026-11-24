"""FAQ bot: sends a question plus the fact sheet to a local Ollama model."""

from __future__ import annotations

import time
from typing import Any, Protocol

from pydantic import BaseModel, Field

from faq_eval.dataset import load_fact_sheet
from faq_eval.prompts import PromptName, build_messages

DEFAULT_MODEL = "llama3.2:3b"
# The fact sheet plus system prompt is ~1.5k tokens; 4096 leaves room for the answer
# and keeps 3B models fully on a 4GB GPU.
DEFAULT_NUM_CTX = 4096
DEFAULT_MAX_TOKENS = 256


class ChatClient(Protocol):
    def chat(self, *, model: str, messages: list[dict[str, str]], options: dict[str, Any], **kwargs: Any) -> Any: ...


class BotConfig(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: PromptName = PromptName.STRICT
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = 0
    num_ctx: int = DEFAULT_NUM_CTX
    max_tokens: int = DEFAULT_MAX_TOKENS

    def options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.max_tokens,
        }
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts


class BotResponse(BaseModel):
    answer: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool = False


def _field(resp: Any, name: str) -> Any:
    """Read a field from an Ollama response object or plain dict."""
    if isinstance(resp, dict):
        return resp.get(name)
    return getattr(resp, name, None)


class FAQBot:
    def __init__(self, config: BotConfig | None = None, client: ChatClient | None = None, fact_sheet: str | None = None):
        self.config = config or BotConfig()
        if client is None:
            import ollama

            client = ollama.Client()
        self.client = client
        self.fact_sheet = fact_sheet if fact_sheet is not None else load_fact_sheet()

    def ask(self, question: str) -> BotResponse:
        messages = build_messages(self.config.prompt, self.fact_sheet, question)
        start = time.perf_counter()
        resp = self.client.chat(model=self.config.model, messages=messages, options=self.config.options())
        latency = time.perf_counter() - start

        message = _field(resp, "message")
        answer = (_field(message, "content") or "").strip()
        return BotResponse(
            answer=answer,
            latency_s=round(latency, 3),
            prompt_tokens=_field(resp, "prompt_eval_count"),
            completion_tokens=_field(resp, "eval_count"),
            truncated=_field(resp, "done_reason") == "length",
        )
