"""LLM judge for answers the rules can't decide.

A small judge model is unreliable when asked for a verdict directly: it tends to judge whether the
*question* is answerable rather than whether the *answer* says anything unsupported. So the judge only
fills in narrow fields, and the verdict is computed from them in code:

- claims:            each factual statement the answer makes, marked supported / not supported by the sheet
- declines:          the answer says it can't answer, the information isn't available, or points elsewhere
- gives_reference:   (answerable questions) the answer conveys the reference answer

Verdict:
- any unsupported claim                       -> unsupported (wrong / hallucination)
- answerable and gives_reference              -> correct
- answerable and declines                     -> refusal (false refusal)
- answerable otherwise                        -> unsupported (wrong)
- unanswerable with no unsupported claims     -> refusal (correct refusal)

The judge runs as a separate pass over saved answers so the judge model and the bot model don't
compete for GPU memory.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

from faq_eval.bot import ChatClient
from faq_eval.dataset import Category, Question
from faq_eval.grader import Grade, Label, grade

DEFAULT_JUDGE_MODEL = "qwen2.5:7b"


class Verdict(str, Enum):
    CORRECT = "correct"
    REFUSAL = "refusal"
    UNSUPPORTED = "unsupported"


class Claim(BaseModel):
    claim: str
    supported: bool


class JudgeOutput(BaseModel):
    claims: list[Claim]
    declines: bool
    gives_reference: bool = False

    @property
    def unsupported_claims(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]

    def verdict(self, answerable: bool) -> Verdict:
        if self.unsupported_claims:
            return Verdict.UNSUPPORTED
        if not answerable:
            return Verdict.REFUSAL
        if self.gives_reference:
            return Verdict.CORRECT
        return Verdict.REFUSAL if self.declines else Verdict.UNSUPPORTED

    def summary(self, answerable: bool) -> str:
        bad = self.unsupported_claims
        if bad:
            return "unsupported: " + "; ".join(bad)
        if answerable:
            return "gives the reference answer" if self.gives_reference else ("declines" if self.declines else "misses the reference answer")
        return "no unsupported claims"


JUDGE_SCHEMA = JudgeOutput.model_json_schema()

_SYSTEM = """You check a college FAQ bot's answer against a fact sheet. Use ONLY the fact sheet below.
Ignore your own knowledge: a statement that is true in the real world but not in the fact sheet is NOT supported.

<fact_sheet>
{fact_sheet}
</fact_sheet>

Return JSON with these fields:
- "claims": every factual statement the answer makes about the college or the world, each with
  "supported": true only if the fact sheet states it or it follows directly from the fact sheet
  (simple arithmetic on fact-sheet numbers counts as supported). Leave out statements that are only about
  what the bot knows or doesn't know ("I don't have that information", "the fact sheet doesn't mention it")
  and leave out advice on where to look ("check the student portal", "contact the office").
  Include statements that are hinted at, such as presenting a guess as fact.
- "declines": true if the answer says it cannot answer, the information is not available, or only suggests where to look.
- "gives_reference": true if the answer conveys the reference answer given below (same meaning; wording may differ).
  false if no reference answer is given."""

_USER_ANSWERABLE = """Question: {question}
Reference answer: {facts}

Bot answer:
<answer>
{answer}
</answer>"""

_USER_UNANSWERABLE = """Question: {question}
Reference answer: none. The fact sheet does not answer this question{premise_note}.

Bot answer:
<answer>
{answer}
</answer>"""


def build_judge_messages(question: Question, answer: str, fact_sheet: str) -> list[dict[str, str]]:
    if question.answerable:
        facts = "; ".join(" or ".join(group) for group in question.required_facts)
        user = _USER_ANSWERABLE.format(question=question.question, facts=facts, answer=answer)
    else:
        premise_note = (
            " because its assumption is false (a statement correcting the assumption from the fact sheet is supported)"
            if question.category == Category.FALSE_PREMISE
            else ""
        )
        user = _USER_UNANSWERABLE.format(question=question.question, premise_note=premise_note, answer=answer)
    return [
        {"role": "system", "content": _SYSTEM.format(fact_sheet=fact_sheet.strip())},
        {"role": "user", "content": user},
    ]


def verdict_to_label(question: Question, verdict: Verdict) -> Label:
    if question.answerable:
        return {Verdict.CORRECT: Label.CORRECT, Verdict.REFUSAL: Label.FALSE_REFUSAL, Verdict.UNSUPPORTED: Label.WRONG}[verdict]
    return Label.HALLUCINATION if verdict == Verdict.UNSUPPORTED else Label.CORRECT_REFUSAL


def needs_judge(question: Question, rule: Grade) -> bool:
    """Uncertain answers always go to the judge. Rule 'hallucination' labels do too, because a reply that
    corrects a near-miss question ("only the second instalment has a late charge") looks like an answer."""
    return rule.label == Label.UNCERTAIN or (not question.answerable and rule.label == Label.HALLUCINATION)


class Judge:
    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, client: ChatClient | None = None, num_ctx: int = 4096):
        if client is None:
            import ollama

            client = ollama.Client()
        self.client = client
        self.model = model
        self.num_ctx = num_ctx

    def judge(self, question: Question, answer: str, fact_sheet: str) -> JudgeOutput:
        messages = build_judge_messages(question, answer, fact_sheet)
        options: dict[str, Any] = {"temperature": 0.0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": 500}
        last_error: Exception | None = None
        for _ in range(2):  # one retry on malformed JSON
            resp = self.client.chat(model=self.model, messages=messages, options=options, format=JUDGE_SCHEMA)
            content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
            try:
                return JudgeOutput.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
        raise ValueError(f"judge returned invalid output for {question.id}: {last_error}")


def final_grade(question: Question, answer: str, fact_sheet: str, judge: Judge | None) -> Grade:
    """Rule grade, overridden by the judge where the rules aren't confident."""
    rule = grade(question, answer)
    if judge is None or not needs_judge(question, rule):
        return rule
    out = judge.judge(question, answer, fact_sheet)
    return rule.model_copy(
        update={
            "label": verdict_to_label(question, out.verdict(question.answerable)),
            "reason": f"judge: {out.summary(question.answerable)}",
            "method": "judge",
        }
    )
