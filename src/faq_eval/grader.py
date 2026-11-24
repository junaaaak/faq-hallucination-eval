"""Rule-based grading of bot answers.

Labels
------
Answerable questions:   correct | wrong | false_refusal
Unanswerable questions: correct_refusal | hallucination
Either:                 uncertain  (rules can't decide; the LLM judge settles these in phase 5)
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from faq_eval.dataset import Category, Question
from faq_eval.matcher import contains_any, is_structured_group, match_facts
from faq_eval.refusal import detect_refusal


class Label(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    FALSE_REFUSAL = "false_refusal"
    CORRECT_REFUSAL = "correct_refusal"
    HALLUCINATION = "hallucination"
    UNCERTAIN = "uncertain"


class Grade(BaseModel):
    label: Label
    reason: str
    method: str = "rule"
    refusal: bool = False
    hedged: bool = False
    matched_facts: list[list[str]] = []
    missing_facts: list[list[str]] = []


def grade(question: Question, answer: str) -> Grade:
    refusal = detect_refusal(answer)
    base = {"refusal": refusal.is_refusal, "hedged": refusal.hedged}

    if question.answerable:
        facts = match_facts(question.required_facts, answer)
        base |= {"matched_facts": facts.matched, "missing_facts": facts.missing}
        if facts.all_matched:
            return Grade(label=Label.CORRECT, reason="all required facts present", **base)
        if facts.any_matched:
            return Grade(label=Label.UNCERTAIN, reason="some required facts present", **base)
        if refusal.is_refusal:
            return Grade(label=Label.FALSE_REFUSAL, reason="refused an answerable question", **base)
        if all(is_structured_group(g) for g in facts.missing):
            return Grade(label=Label.WRONG, reason="expected number/time/date not found", **base)
        return Grade(label=Label.UNCERTAIN, reason="expected wording not found", **base)

    if question.category == Category.FALSE_PREMISE:
        hit = contains_any(question.premise_corrections, answer)
        if hit:
            return Grade(label=Label.CORRECT_REFUSAL, reason=f"corrected the premise ({hit!r})", **base)

    if refusal.exact:
        return Grade(label=Label.CORRECT_REFUSAL, reason="exact refusal phrase", **base)
    if refusal.is_refusal and not refusal.hedged:
        return Grade(label=Label.CORRECT_REFUSAL, reason="refused", **base)
    if refusal.is_refusal:
        return Grade(label=Label.UNCERTAIN, reason="refusal followed by extra content", **base)
    if refusal.clarifying:
        return Grade(label=Label.UNCERTAIN, reason="asked the student for more details", **base)
    if refusal.redirect:
        return Grade(label=Label.UNCERTAIN, reason="redirected the student elsewhere", **base)
    return Grade(label=Label.HALLUCINATION, reason="answered a question the sheet cannot answer", **base)
