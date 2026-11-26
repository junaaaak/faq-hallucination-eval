"""Hand-labelling sheet and grader-vs-human agreement.

`make_label_sheet` writes a CSV of answers (without the grader's labels, so labelling is blind) for a
person to fill in the `human_label` column. `agreement` compares those labels with the rule-only and
rule+judge labels.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from pydantic import BaseModel
from sklearn.metrics import cohen_kappa_score

from faq_eval.dataset import load_questions
from faq_eval.grader import Label
from faq_eval.refusal import detect_refusal
from faq_eval.runner import RESULTS_DIR, GradedRecord, load_graded

LABELS_DIR = RESULTS_DIR / "labels"
HUMAN_LABELS = [l.value for l in Label if l != Label.UNCERTAIN]
ERROR_LABELS = {Label.WRONG.value, Label.HALLUCINATION.value}

COLUMNS = [
    "item", "model", "prompt", "temperature", "sample", "question_id", "category", "answerable",
    "question", "reference", "answer", "human_label", "notes",
]


def label_sheet_path(name: str) -> Path:
    return LABELS_DIR / f"{name}.csv"


def _reference(qid: str) -> str:
    q = {q.id: q for q in load_questions()}[qid]
    if q.answerable:
        return "MUST INCLUDE: " + "; ".join(" or ".join(g) for g in q.required_facts)
    if q.premise_corrections:
        return "NOT ANSWERABLE (false premise; correcting it is fine): " + " / ".join(q.premise_corrections)
    return "NOT ANSWERABLE from the fact sheet"


def make_label_sheet(name: str, n: int = 60, seed: int = 0, overwrite: bool = False) -> Path:
    """Pick n answers: half from those the judge handled (the hard ones), the rest spread over rule labels."""
    path = label_sheet_path(name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; it may contain your labels. Pass overwrite to replace it.")
    graded = load_graded(name)
    rng = random.Random(seed)

    judged = [g for g in graded if g.method == "judge"]
    rest = [g for g in graded if g.method != "judge"]
    picked = rng.sample(judged, min(len(judged), n // 2))

    # Round-robin over rule labels so rare labels (false_refusal, wrong) are represented.
    by_label: dict[str, list[GradedRecord]] = {}
    for g in rng.sample(rest, len(rest)):
        by_label.setdefault(g.label.value, []).append(g)
    while len(picked) < n and any(by_label.values()):
        for bucket in by_label.values():
            if bucket and len(picked) < n:
                picked.append(bucket.pop())
    rng.shuffle(picked)

    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel shows the rupee sign correctly.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, g in enumerate(picked, start=1):
            w.writerow({
                "item": i, "model": g.model, "prompt": g.prompt, "temperature": g.temperature, "sample": g.sample,
                "question_id": g.question_id, "category": g.category, "answerable": g.answerable,
                "question": g.question, "reference": _reference(g.question_id), "answer": g.answer,
                "human_label": "", "notes": "",
            })
    return path


class Agreement(BaseModel):
    n: int
    rule_accuracy: float  # rule-only label matches human (uncertain counts as a miss)
    rule_undecided: int
    final_accuracy: float  # rule+judge label matches human
    final_kappa: float
    error_accuracy: float  # agreement on the binary question "is this a wrong answer / hallucination?"
    confusion: dict[str, dict[str, int]]  # human label -> final label -> count
    disagreements: list[dict]


PARTIAL_AS = ("error", "ok")


def resolve_human_label(raw: str, answerable: bool, answer: str, partial_as: str = "error") -> str:
    """Map a hand label onto the grader's label set.

    Besides the grader's own labels, a simpler scheme is accepted:
      correct / incorrect / partial            (resolved using whether the question is answerable)
      safe_unanswerable / unsafe_unanswerable
    "incorrect" on an answerable question is a false refusal when the answer is exactly the strict prompt's
    refusal sentence, otherwise wrong. "partial" has no grader equivalent; partial_as decides whether it
    counts as an error or as acceptable.
    """
    label = raw.strip().lower()
    # "correct" is handled below rather than here: on an unanswerable question it means a correct refusal.
    if label in HUMAN_LABELS and label != Label.CORRECT.value:
        return label
    if label == "safe_unanswerable":
        return Label.CORRECT_REFUSAL.value
    if label == "unsafe_unanswerable":
        return Label.HALLUCINATION.value
    if label == "partial":
        if partial_as not in PARTIAL_AS:
            raise ValueError(f"partial_as must be one of {PARTIAL_AS}")
        label = "incorrect" if partial_as == "error" else "correct"
    if label == "correct":
        return Label.CORRECT.value if answerable else Label.CORRECT_REFUSAL.value
    if label == "incorrect":
        if not answerable:
            return Label.HALLUCINATION.value
        return Label.FALSE_REFUSAL.value if detect_refusal(answer).exact else Label.WRONG.value
    raise ValueError(
        f"unknown label {raw!r}; use one of {HUMAN_LABELS} or correct/incorrect/partial/safe_unanswerable/unsafe_unanswerable"
    )


def agreement(name: str, partial_as: str = "error") -> Agreement:
    path = label_sheet_path(name)
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human_label"].strip()]
    if not rows:
        raise ValueError(f"no human labels filled in {path}")

    graded = {(g.model, g.prompt, g.temperature, g.sample, g.question_id): g for g in load_graded(name)}
    human, rule, final, disagreements = [], [], [], []
    for r in rows:
        g = graded[(r["model"], r["prompt"], float(r["temperature"]), int(r["sample"]), r["question_id"])]
        h = resolve_human_label(r["human_label"], g.answerable, g.answer, partial_as)
        human.append(h)
        rule.append(g.rule_label.value)
        final.append(g.label.value)
        if g.label.value != h:
            disagreements.append({
                "item": r["item"], "question_id": g.question_id, "raw": r["human_label"].strip(), "human": h,
                "grader": g.label.value, "method": g.method, "reason": g.reason, "answer": g.answer,
            })

    confusion: dict[str, dict[str, int]] = {}
    for h, f_ in zip(human, final):
        confusion.setdefault(h, Counter())[f_] += 1
    n = len(human)
    return Agreement(
        n=n,
        rule_accuracy=sum(h == r for h, r in zip(human, rule)) / n,
        rule_undecided=sum(r == Label.UNCERTAIN.value for r in rule),
        final_accuracy=sum(h == f_ for h, f_ in zip(human, final)) / n,
        final_kappa=float(cohen_kappa_score(human, final)) if len(set(human) | set(final)) > 1 else 1.0,
        error_accuracy=sum((h in ERROR_LABELS) == (f_ in ERROR_LABELS) for h, f_ in zip(human, final)) / n,
        confusion={h: dict(c) for h, c in confusion.items()},
        disagreements=disagreements,
    )
