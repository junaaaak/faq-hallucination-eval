"""Run the bot over the question set and grade the saved answers.

Answers are appended to results/runs/<name>.jsonl one line at a time, so an interrupted run resumes
where it stopped. Grading reads that file and writes results/graded/<name>.jsonl; judge verdicts
are reused when the answer hasn't changed.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from faq_eval.bot import BotConfig, FAQBot
from faq_eval.dataset import Question, load_fact_sheet, load_questions
from faq_eval.grader import Label, grade
from faq_eval.judge import Judge, final_grade

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RUNS_DIR = RESULTS_DIR / "runs"
GRADED_DIR = RESULTS_DIR / "graded"


class RunRecord(BaseModel):
    model: str
    prompt: str
    temperature: float
    sample: int
    seed: int
    question_id: str
    category: str
    answerable: bool
    question: str
    answer: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool = False
    timestamp: str

    def key(self) -> tuple:
        return (self.model, self.prompt, self.temperature, self.sample, self.question_id)


class GradedRecord(RunRecord):
    rule_label: Label
    label: Label
    method: str
    reason: str


def read_jsonl(path: Path, model: type[BaseModel]) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


def run_path(name: str) -> Path:
    return RUNS_DIR / f"{name}.jsonl"


def graded_path(name: str) -> Path:
    return GRADED_DIR / f"{name}.jsonl"


def run(
    name: str,
    models: Iterable[str],
    prompts: Iterable[str],
    temperatures: Iterable[float] = (0.0,),
    samples: int = 1,
    questions: list[Question] | None = None,
    bot_factory: Callable[[BotConfig], FAQBot] = FAQBot,
    on_progress: Callable[[int, int, RunRecord], None] | None = None,
) -> Path:
    """Ask every (model, prompt, temperature, sample, question) combination not already in the run file."""
    questions = questions or load_questions()
    path = run_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r.key() for r in read_jsonl(path, RunRecord)}

    jobs = [
        (m, p, t, s, q)
        for m, p, t, s in itertools.product(models, prompts, temperatures, range(samples))
        for q in questions
        if (m, p, float(t), s, q.id) not in done
    ]
    # Group by model so each model is loaded onto the GPU once.
    bots: dict[tuple, FAQBot] = {}
    with path.open("a", encoding="utf-8") as f:
        for i, (m, p, t, s, q) in enumerate(jobs, start=1):
            cfg_key = (m, p, float(t), s)
            if cfg_key not in bots:
                bots.clear()
                bots[cfg_key] = bot_factory(BotConfig(model=m, prompt=p, temperature=t, seed=s))
            resp = bots[cfg_key].ask(q.question)
            rec = RunRecord(
                model=m, prompt=p, temperature=float(t), sample=s, seed=s,
                question_id=q.id, category=q.category.value, answerable=q.answerable, question=q.question,
                answer=resp.answer, latency_s=resp.latency_s, prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens, truncated=resp.truncated,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            f.write(rec.model_dump_json() + "\n")
            f.flush()
            if on_progress:
                on_progress(i, len(jobs), rec)
    return path


def grade_run(
    name: str,
    judge: Judge | None = None,
    on_progress: Callable[[int, int, GradedRecord], None] | None = None,
    rejudge: bool = False,
) -> Path:
    """Grade every answer in a run, reusing earlier judge verdicts for unchanged answers unless rejudge is set."""
    questions = {q.id: q for q in load_questions()}
    fact_sheet = load_fact_sheet()
    records: list[RunRecord] = read_jsonl(run_path(name), RunRecord)
    if not records:
        raise FileNotFoundError(f"no answers in {run_path(name)}")

    out_path = graded_path(name)
    previous = {(g.key(), g.answer): g for g in read_jsonl(out_path, GradedRecord) if g.method == "judge"}

    graded: list[GradedRecord] = []
    for i, rec in enumerate(records, start=1):
        q = questions[rec.question_id]
        cached = previous.get((rec.key(), rec.answer)) if judge and not rejudge else None
        if cached:
            g = cached
        else:
            rule = grade(q, rec.answer)
            final = final_grade(q, rec.answer, fact_sheet, judge=judge) if judge else rule
            g = GradedRecord(
                **rec.model_dump(), rule_label=rule.label, label=final.label, method=final.method, reason=final.reason
            )
        graded.append(g)
        if on_progress:
            on_progress(i, len(records), g)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for g in graded:
            f.write(g.model_dump_json() + "\n")
    return out_path


def load_graded(name: str) -> list[GradedRecord]:
    return read_jsonl(graded_path(name), GradedRecord)
