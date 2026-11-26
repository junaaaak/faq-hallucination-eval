"""LLM judge for answers the rules can't decide.

A small judge model is unreliable when asked for a verdict directly: it tends to judge whether the
*question* is answerable rather than whether the *answer* says anything unsupported. So the judge only
fills in narrow fields, and the verdict is computed from them in code:

- claims:            each statement the answer makes, marked supported / not supported by the sheet, and
                     tagged by kind: about the college or the student, about the outside world, or general advice
- declines:          the answer says it can't answer, the information isn't available, or points elsewhere
- gives_reference:   (answerable questions) the answer conveys the reference answer

Generic advice (how to calculate a percentage, what laptop specs suit engineering) is not counted as a
hallucination: it says nothing false about the college. Unsupported claims about the college, the student
or the outside world are.

Verdict:
- any unsupported non-advice claim            -> unsupported (wrong / hallucination)
- answerable and gives_reference              -> correct
- answerable and declines                     -> refusal (false refusal)
- answerable otherwise                        -> unsupported (wrong)
- unanswerable with no unsupported claims     -> refusal (correct refusal)

The judge runs as a separate pass over saved answers so the judge model and the bot model don't
compete for GPU memory.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from faq_eval.bot import ChatClient
from faq_eval.dataset import Category, Question
from faq_eval.grader import Grade, Label, grade
from faq_eval.refusal import detect_refusal

DEFAULT_JUDGE_MODEL = "qwen2.5:7b"


class Verdict(str, Enum):
    CORRECT = "correct"
    REFUSAL = "refusal"
    UNSUPPORTED = "unsupported"


def _is_meta(claim: str) -> bool:
    r = detect_refusal(claim)
    return r.is_refusal or r.clarifying or r.redirect


# Specific facts that can't be generic advice: money, percentages, emails, phone numbers, the college's name.
# The judge sometimes tags a fabricated fact ("the ECE highest package was ₹42 lakh") as advice.
_FACT_MARKER_RE = re.compile(r"₹|\brs\.?\s*\d|\blakh|\bcrore|\d\s*%|\S+@\S+|\d[\d\s-]{6,}\d|northbridge", re.I)


class Claim(BaseModel):
    claim: str
    supported: bool
    # college: about Northbridge, its offices, rules, fees, people or events, or about the student asking.
    # world:   a factual statement about anything else (other colleges, sport, geography).
    # advice:  a general recommendation or how-to that isn't about this college.
    kind: Literal["college", "world", "advice"] = "college"

    @property
    def is_advice(self) -> bool:
        return self.kind == "advice" and not _FACT_MARKER_RE.search(self.claim)


class JudgeOutput(BaseModel):
    claims: list[Claim]
    declines: bool
    corrects_assumption: bool = False
    presents_answer: bool = False
    gives_reference: bool = False

    @property
    def unsupported_claims(self) -> list[str]:
        # The judge sometimes lists meta statements ("the cost is not mentioned", "I need to know your semester")
        # as claims; those aren't facts about the college, so they're dropped here. Generic advice is dropped too.
        return [
            c.claim for c in self.claims if not c.supported and not c.is_advice and not _is_meta(c.claim)
        ]

    def verdict(self, answerable: bool, refusal_signals: bool = True) -> Verdict:
        """refusal_signals: the rules saw a refusal, redirect or clarifying question in the answer.

        Without such signals, the judge can only clear an unanswerable-question answer by saying it corrects
        the question's assumption. A 7B judge sometimes paraphrases a fabricated answer into a harmless
        claim, so "no unsupported claims" alone isn't trusted there."""
        if self.unsupported_claims:
            return Verdict.UNSUPPORTED
        if not answerable:
            if self.corrects_assumption:
                return Verdict.REFUSAL
            # Answers with no refusal language that answer with the wrong fact (the NET deadline when asked for
            # the M.Tech deadline) are caught here. presents_answer is recorded but not used: with refusal
            # language present it misfired on answers like "the sheet doesn't specify a minimum GATE score".
            return Verdict.REFUSAL if refusal_signals else Verdict.UNSUPPORTED
        if self.gives_reference:
            return Verdict.CORRECT
        return Verdict.REFUSAL if self.declines else Verdict.UNSUPPORTED

    def summary(self, answerable: bool, refusal_signals: bool = True) -> str:
        bad = self.unsupported_claims
        if bad:
            return "unsupported: " + "; ".join(bad)
        if answerable:
            return "gives the reference answer" if self.gives_reference else ("declines" if self.declines else "misses the reference answer")
        if self.corrects_assumption:
            return "corrects the question's assumption"
        if not refusal_signals:
            return "answers without declining, and no correction found"
        return "no unsupported claims"


JUDGE_SCHEMA = JudgeOutput.model_json_schema()
# kind has a default only so older cached outputs still load; the judge must always fill it in.
JUDGE_SCHEMA["$defs"]["Claim"]["required"] = ["claim", "supported", "kind"]

_SYSTEM = """You check a college FAQ bot's answer against a fact sheet. Use ONLY the fact sheet below.
Ignore your own knowledge: a statement that is true in the real world but not in the fact sheet is NOT supported.

<fact_sheet>
{fact_sheet}
</fact_sheet>

Return JSON with these fields:
- "claims": every factual statement the answer makes about the college or the world, each with
  "supported": true only if the fact sheet states it or it follows directly from the fact sheet
  (simple arithmetic on fact-sheet numbers counts as supported). List at most 8 claims, the ones that answer
  the question first; for long lists or how-to steps, one claim per item is enough.
  * Check each claim against the fact sheet yourself. The bot often says "according to the fact sheet" before
    things the fact sheet never says; that phrase does not make a claim supported.
  * A claim that stretches a fact to something the sheet doesn't cover is NOT supported
    (e.g. the sheet gives a fee "per semester" and the answer says it is the fee "during summer vacation").
  * General knowledge, recommendations and how-to advice count as claims and are NOT supported unless the sheet says them.
  * "kind" for each claim:
      "college" - says something about Northbridge Institute of Technology (its departments, offices, staff,
                  facilities, rules, fees, dates, events, what its offices can do) or about the student asking.
      "world"   - a factual statement about anything outside the college (other colleges and their rankings,
                  sport, geography, history, companies).
      "advice"  - a general recommendation, opinion or how-to that would apply to any college and states no fact
                  about this one (how to calculate a percentage, what laptop specs suit engineering, be respectful).
    If a claim mixes advice with a fact about the college (e.g. "assume there is no strict dress code here"), it is "college".
  * Copy each claim as the answer states it. Do not rewrite a claim into what the fact sheet says instead.
  * Leave out only statements about what the bot knows ("I don't have that information", "the fact sheet doesn't
    mention it"), pointers to where to look ("check the student portal", "contact the office"), and requests for
    the student's details ("please tell me your semester") - these are not claims.
- "declines": true if the answer says it cannot answer, the information is not available, or only suggests where to look.
- "corrects_assumption": true if the answer points out, using the fact sheet, that something the question assumes is
  different from what the sheet says (e.g. the programme isn't offered, cash isn't accepted, the charge applies to a
  different instalment, only one scholarship is allowed, the hostel is for boys). false otherwise.
- "presents_answer": true if the answer gives the student a specific answer to their question as if it were true,
  including answering with a fact about something different (e.g. asked for the M.Tech deadline, it gives the
  NET deadline; asked about a programme that doesn't exist, it quotes a fee as if it existed).
  false if it declines, says the information is not available, points out that the question's assumption is
  wrong, only asks for the student's details, or only mentions related facts while making clear they don't answer the question.
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
        options: dict[str, Any] = {"temperature": 0.0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": 1024}
        last_error: Exception | None = None
        for _ in range(2):  # one retry on malformed JSON
            resp = self.client.chat(model=self.model, messages=messages, options=options, format=JUDGE_SCHEMA)
            content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
            try:
                return JudgeOutput.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
        raise ValueError(f"judge returned invalid output for {question.id}: {last_error}")


JUDGE_PROMPT_VERSION = hashlib.sha1((_SYSTEM + _USER_ANSWERABLE + _USER_UNANSWERABLE).encode()).hexdigest()[:10]


class CachedJudge:
    """Wraps a Judge and appends each verdict to a JSONL file as soon as it's made."""

    def __init__(self, judge: Judge, path: Path, read: bool = True):
        self.inner = judge
        self.model = judge.model
        self.path = path
        self.calls = 0
        self._cache: dict[tuple, JudgeOutput] = {}
        if read and path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        key = (row["model"], row["prompt_version"], row["sheet"], row["question_id"], row["answer"])
                        self._cache[key] = JudgeOutput.model_validate(row["output"])

    def judge(self, question: Question, answer: str, fact_sheet: str) -> JudgeOutput:
        sheet = hashlib.sha1(fact_sheet.encode()).hexdigest()[:10]
        key = (self.model, JUDGE_PROMPT_VERSION, sheet, question.id, answer)
        if key in self._cache:
            return self._cache[key]
        out = self.inner.judge(question, answer, fact_sheet)
        self.calls += 1
        self._cache[key] = out
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            row = dict(zip(("model", "prompt_version", "sheet", "question_id", "answer"), key))
            f.write(json.dumps({**row, "output": out.model_dump()}, ensure_ascii=False) + "\n")
        return out


def final_grade(question: Question, answer: str, fact_sheet: str, judge: Judge | CachedJudge | None) -> Grade:
    """Rule grade, overridden by the judge where the rules aren't confident."""
    rule = grade(question, answer)
    if judge is None or not needs_judge(question, rule):
        return rule
    try:
        out = judge.judge(question, answer, fact_sheet)
    except ValueError as exc:
        # Keep the rule label rather than stopping a whole grading run on one bad judge output.
        return rule.model_copy(update={"reason": f"judge failed, kept rule label: {exc}", "method": "judge_error"})
    r = detect_refusal(answer)
    signals = r.is_refusal or r.redirect or r.clarifying
    return rule.model_copy(
        update={
            "label": verdict_to_label(question, out.verdict(question.answerable, signals)),
            "reason": f"judge: {out.summary(question.answerable, signals)}",
            "method": "judge",
        }
    )
