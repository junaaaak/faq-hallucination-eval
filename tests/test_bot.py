import pytest

from faq_eval.bot import BotConfig, FAQBot
from faq_eval.prompts import REFUSAL_PHRASE, PromptName, build_messages, build_system_prompt

SHEET = "- Library closes at 9:00 PM."


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, *, model, messages, options):
        self.calls.append({"model": model, "messages": messages, "options": options})
        return self.response


def test_strict_prompt_contains_refusal_and_sheet():
    prompt = build_system_prompt(PromptName.STRICT, SHEET)
    assert REFUSAL_PHRASE in prompt
    assert SHEET in prompt


def test_lenient_prompt_has_no_refusal_instruction():
    prompt = build_system_prompt("lenient", SHEET)
    assert REFUSAL_PHRASE not in prompt
    assert SHEET in prompt


def test_unknown_prompt_rejected():
    with pytest.raises(ValueError):
        build_system_prompt("nope", SHEET)


def test_messages_shape():
    msgs = build_messages("strict", SHEET, "  When does the library close?  ")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "When does the library close?"


def test_options_include_seed_and_temperature():
    opts = BotConfig(temperature=0.7, seed=3).options()
    assert opts["temperature"] == 0.7
    assert opts["seed"] == 3
    assert "seed" not in BotConfig(seed=None).options()


def test_ask_parses_response():
    client = FakeClient(
        {
            "message": {"role": "assistant", "content": "  It closes at 9:00 PM.  "},
            "prompt_eval_count": 120,
            "eval_count": 8,
            "done_reason": "stop",
        }
    )
    bot = FAQBot(BotConfig(model="m"), client=client, fact_sheet=SHEET)
    resp = bot.ask("When does the library close?")

    assert resp.answer == "It closes at 9:00 PM."
    assert resp.prompt_tokens == 120
    assert resp.completion_tokens == 8
    assert not resp.truncated
    assert client.calls[0]["model"] == "m"
    assert SHEET in client.calls[0]["messages"][0]["content"]


def test_ask_flags_truncation():
    client = FakeClient({"message": {"content": "partial"}, "done_reason": "length"})
    resp = FAQBot(client=client, fact_sheet=SHEET).ask("q")
    assert resp.truncated
