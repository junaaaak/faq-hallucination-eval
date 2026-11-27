"""Helpers behind the Streamlit demo (kept separate from the UI so they can be tested)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
from pydantic import BaseModel

from faq_eval import runner
from faq_eval.bot import BotConfig, BotResponse, ChatClient, FAQBot
from faq_eval.dataset import Question
from faq_eval.grader import Grade, Label
from faq_eval.judge import Judge, final_grade
from faq_eval.metrics import setting_metrics
from faq_eval.prompts import PromptName
from faq_eval.refusal import detect_refusal

LABEL_TEXT = {
    Label.CORRECT: "Correct",
    Label.WRONG: "Wrong",
    Label.FALSE_REFUSAL: "False refusal",
    Label.CORRECT_REFUSAL: "Correct refusal",
    Label.HALLUCINATION: "Hallucination",
    Label.UNCERTAIN: "Uncertain",
}
GOOD_LABELS = {Label.CORRECT, Label.CORRECT_REFUSAL}


def available_runs() -> list[str]:
    """Names of runs that have graded results, main first."""
    names = sorted(p.stem for p in runner.GRADED_DIR.glob("*.jsonl")) if runner.GRADED_DIR.exists() else []
    return sorted(names, key=lambda n: (n != "main", n))


def records_frame(records: Iterable[runner.GradedRecord]) -> pd.DataFrame:
    cols = ["model", "prompt", "temperature", "sample", "question_id", "category", "answerable", "question",
            "answer", "label", "method", "reason"]
    rows = [{c: getattr(r, c) for c in cols} for r in records]
    df = pd.DataFrame(rows, columns=cols)
    df["label"] = df["label"].map(lambda v: Label(v).value)
    return df


def metrics_frame(records: Iterable[runner.GradedRecord]) -> pd.DataFrame:
    rows = []
    for m in setting_metrics(records):
        rows.append({
            "model": m.model, "prompt": m.prompt, "temperature": m.temperature, "samples": m.samples,
            "accuracy": m.accuracy.p, "accuracy 95% CI": _ci(m.accuracy),
            "hallucination": m.hallucination.p, "hallucination 95% CI": _ci(m.hallucination),
            "false refusal": m.false_refusal.p, "balanced": m.balanced,
        })
    return pd.DataFrame(rows)


def _ci(rate) -> str:
    lo, hi = rate.ci
    return f"{lo:.0%}–{hi:.0%}" if rate.n else "n/a"


def filter_frame(df: pd.DataFrame, **filters: list[Any] | None) -> pd.DataFrame:
    """Keep rows whose column value is in the given list; an empty or None list means no filter."""
    for col, allowed in filters.items():
        if allowed:
            df = df[df[col].isin(allowed)]
    return df


def recorded_answers(df: pd.DataFrame, question_id: str) -> pd.DataFrame:
    """How a question was answered in a recorded run, one row per setting with label counts."""
    sub = df[df["question_id"] == question_id]
    if sub.empty:
        return sub
    out = (sub.groupby(["model", "prompt", "temperature"])["label"]
           .agg(lambda s: ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in s.value_counts().items()))
           .reset_index())
    return out


class LiveResult(BaseModel):
    model: str
    prompt: str
    temperature: float
    response: BotResponse
    grade: Grade | None = None  # None for a free-text question (no ground truth)
    refusal: bool
    hedged: bool


def ask(question: Question | str, model: str, prompt: str, temperature: float, *, judge_model: str | None = None,
        client: ChatClient | None = None, judge_client: ChatClient | None = None,
        fact_sheet: str | None = None) -> LiveResult:
    """Ask the bot one question. Questions from the set are graded (rules, then the judge if given);
    free-text questions only get refusal detection, since there is no reference answer."""
    seed = 0 if temperature == 0 else None
    bot = FAQBot(BotConfig(model=model, prompt=PromptName(prompt), temperature=temperature, seed=seed),
                 client=client, fact_sheet=fact_sheet)
    text = question.question if isinstance(question, Question) else question
    resp = bot.ask(text)
    r = detect_refusal(resp.answer)
    g = None
    if isinstance(question, Question):
        judge = Judge(model=judge_model, client=judge_client) if judge_model else None
        g = final_grade(question, resp.answer, bot.fact_sheet, judge)
    return LiveResult(model=model, prompt=prompt, temperature=temperature, response=resp, grade=g,
                      refusal=r.is_refusal, hedged=r.hedged)
