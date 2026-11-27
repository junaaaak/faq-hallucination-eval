"""Demo helpers, with fake Ollama clients."""

import pandas as pd

from faq_eval import demo, runner
from faq_eval.dataset import load_questions
from faq_eval.grader import Label
from faq_eval.prompts import REFUSAL_PHRASE

Q = {q.id: q for q in load_questions()}


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.options = None

    def chat(self, *, model, messages, options, **kwargs):
        self.options = options
        return {"message": {"content": self.answer}, "eval_count": 5}


def test_ask_grades_questions_from_the_set():
    res = demo.ask(Q["a001"], "m", "strict", 0.0, client=FakeClient("It closes at 9:00 PM."), fact_sheet="x")
    assert res.grade.label == Label.CORRECT and not res.refusal


def test_ask_refusal_on_unanswerable():
    client = FakeClient(REFUSAL_PHRASE)
    res = demo.ask(Q["u013"], "m", "lenient", 0.7, client=client, fact_sheet="x")
    assert res.grade.label == Label.CORRECT_REFUSAL and res.refusal
    assert "seed" not in client.options  # sampling at temperature > 0 isn't pinned


def test_ask_free_text_is_not_graded():
    res = demo.ask("Who is the dean?", "m", "strict", 0.0, client=FakeClient(REFUSAL_PHRASE), fact_sheet="x")
    assert res.grade is None and res.refusal


def rec(model, prompt, qid, answerable, label, temperature=0.0, sample=0):
    return runner.GradedRecord(
        model=model, prompt=prompt, temperature=temperature, sample=sample, seed=0, question_id=qid,
        category="direct" if answerable else "near_miss", answerable=answerable, question="q", answer="a",
        latency_s=0.0, timestamp="t", rule_label=label, label=label, method="rule", reason="",
    )


RECORDS = [
    rec("m", "strict", "a1", True, Label.CORRECT),
    rec("m", "strict", "u1", False, Label.HALLUCINATION),
    rec("m", "lenient", "u1", False, Label.HALLUCINATION, temperature=1.0, sample=0),
    rec("m", "lenient", "u1", False, Label.HALLUCINATION, temperature=1.0, sample=1),
    rec("m", "lenient", "u1", False, Label.CORRECT_REFUSAL, temperature=1.0, sample=2),
]


def test_frames_and_filters():
    df = demo.records_frame(RECORDS)
    assert set(df["label"]) == {"correct", "hallucination", "correct_refusal"}
    assert len(demo.filter_frame(df, label=["hallucination"], prompt=None)) == 3
    assert len(demo.filter_frame(df, label=["hallucination"], prompt=["strict"])) == 1
    m = demo.metrics_frame(RECORDS)
    strict = m[m["prompt"] == "strict"].iloc[0]
    assert strict["accuracy"] == 1.0 and strict["hallucination"] == 1.0


def test_recorded_answers_summarise_samples():
    out = demo.recorded_answers(demo.records_frame(RECORDS), "u1")
    row = out[out["prompt"] == "lenient"].iloc[0]
    assert row["label"] == "hallucination ×2, correct_refusal"
    assert demo.recorded_answers(demo.records_frame(RECORDS), "nope").empty


def test_available_runs_puts_main_first(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "GRADED_DIR", tmp_path)
    for n in ("sweep", "main", "alpha"):
        (tmp_path / f"{n}.jsonl").write_text("", encoding="utf-8")
    assert demo.available_runs() == ["main", "alpha", "sweep"]
