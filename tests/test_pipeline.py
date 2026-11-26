"""Judge, runner and labelling tests with fake model clients (no Ollama needed)."""

import csv
import json

import pytest

from faq_eval import labels, runner
from faq_eval.bot import BotResponse
from faq_eval.dataset import load_questions
from faq_eval.grader import Label, grade
from faq_eval.judge import (
    Claim,
    Judge,
    JudgeOutput,
    Verdict,
    build_judge_messages,
    final_grade,
    needs_judge,
    verdict_to_label,
)
from faq_eval.prompts import REFUSAL_PHRASE

Q = {q.id: q for q in load_questions()}
SHEET = "- Library closes at 9:00 PM."


class FakeJudgeClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    def chat(self, *, model, messages, options, format=None):
        self.calls += 1
        return {"message": {"content": self.contents.pop(0)}}


def judge_json(claims=(), declines=False, corrects_assumption=False, gives_reference=False) -> str:
    return json.dumps({
        "claims": [{"claim": c, "supported": s} for c, s in claims],
        "declines": declines,
        "corrects_assumption": corrects_assumption,
        "gives_reference": gives_reference,
    })


def test_judge_prompt_states_ground_truth():
    ans = build_judge_messages(Q["a001"], "9 PM", SHEET)[1]["content"]
    una = build_judge_messages(Q["u037"], "x", SHEET)[1]["content"]
    assert "Reference answer: 9:00 PM" in ans
    assert "does not answer this question" in una and "assumption is false" in una


@pytest.mark.parametrize(
    "answerable, out, verdict",
    [
        (True, JudgeOutput(claims=[Claim(claim="x", supported=True)], declines=False, gives_reference=True), Verdict.CORRECT),
        (True, JudgeOutput(claims=[Claim(claim="x", supported=False)], declines=False, gives_reference=True), Verdict.UNSUPPORTED),
        (True, JudgeOutput(claims=[], declines=True), Verdict.REFUSAL),
        (True, JudgeOutput(claims=[Claim(claim="x", supported=True)], declines=False), Verdict.UNSUPPORTED),
        (False, JudgeOutput(claims=[Claim(claim="x", supported=True)], declines=False), Verdict.REFUSAL),
        (False, JudgeOutput(claims=[], declines=False), Verdict.REFUSAL),
        (False, JudgeOutput(claims=[Claim(claim="x", supported=False)], declines=True), Verdict.UNSUPPORTED),
    ],
)
def test_verdict_from_fields(answerable, out, verdict):
    assert out.verdict(answerable) == verdict


def test_unanswerable_without_refusal_signals_needs_a_correction():
    plain = JudgeOutput(claims=[Claim(claim="x", supported=True)], declines=True)
    corrected = JudgeOutput(claims=[Claim(claim="x", supported=True)], declines=False, corrects_assumption=True)
    assert plain.verdict(False, refusal_signals=False) == Verdict.UNSUPPORTED
    assert corrected.verdict(False, refusal_signals=False) == Verdict.REFUSAL


def test_meta_claims_are_ignored():
    out = JudgeOutput(
        claims=[
            Claim(claim="The cost of the swimming pool is **not mentioned**.", supported=False),
            Claim(claim="I need to know which semester you are in.", supported=False),
        ],
        declines=True,
    )
    assert out.unsupported_claims == []


@pytest.mark.parametrize(
    "qid, verdict, label",
    [
        ("a001", Verdict.CORRECT, Label.CORRECT),
        ("a001", Verdict.REFUSAL, Label.FALSE_REFUSAL),
        ("a001", Verdict.UNSUPPORTED, Label.WRONG),
        ("u013", Verdict.CORRECT, Label.CORRECT_REFUSAL),
        ("u013", Verdict.REFUSAL, Label.CORRECT_REFUSAL),
        ("u013", Verdict.UNSUPPORTED, Label.HALLUCINATION),
    ],
)
def test_verdict_mapping(qid, verdict, label):
    assert verdict_to_label(Q[qid], verdict) == label


def test_only_unsure_or_hallucination_go_to_judge():
    assert not needs_judge(Q["a001"], grade(Q["a001"], "It closes at 9 PM."))
    assert not needs_judge(Q["u013"], grade(Q["u013"], REFUSAL_PHRASE))
    assert needs_judge(Q["u014"], grade(Q["u014"], "Only the second instalment has a late charge of ₹2,000."))


def test_final_grade_uses_judge_and_retries_bad_json():
    client = FakeJudgeClient(
        ["not json", judge_json(claims=[("only the second instalment has a late charge", True)], corrects_assumption=True)]
    )
    g = final_grade(Q["u014"], "Only the second instalment has a late charge.", SHEET, Judge(model="j", client=client))
    assert g.label == Label.CORRECT_REFUSAL
    assert g.method == "judge"
    assert client.calls == 2


def test_judge_gives_up_after_retry():
    client = FakeJudgeClient(["{}", "{}"])
    with pytest.raises(ValueError):
        Judge(model="j", client=client).judge(Q["u014"], "x", SHEET)


def test_final_grade_keeps_rule_label_when_judge_fails():
    client = FakeJudgeClient(["{", "{"])
    g = final_grade(Q["u001"], "Argentina won.", SHEET, Judge(model="j", client=client))
    assert g.label == Label.HALLUCINATION
    assert g.method == "judge_error"


class FakeBot:
    asked = 0

    def __init__(self, config):
        self.config = config

    def ask(self, question):
        FakeBot.asked += 1
        return BotResponse(answer=REFUSAL_PHRASE, latency_s=0.0)


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "GRADED_DIR", tmp_path / "graded")
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(labels, "LABELS_DIR", tmp_path / "labels")
    FakeBot.asked = 0
    return tmp_path


def test_run_resumes_without_repeating(tmp_results):
    qs = [Q["a001"], Q["u013"]]
    runner.run("t", ["m"], ["strict", "lenient"], [0.0], 1, questions=qs, bot_factory=FakeBot)
    assert FakeBot.asked == 4
    runner.run("t", ["m"], ["strict", "lenient"], [0.0, 0.7], 1, questions=qs, bot_factory=FakeBot)
    assert FakeBot.asked == 8  # only the new temperature was asked
    assert len(runner.read_jsonl(runner.run_path("t"), runner.RunRecord)) == 8


class HallucinatingBot(FakeBot):
    def ask(self, question):
        FakeBot.asked += 1
        return BotResponse(answer="Exam-week hours are 7 AM to midnight.", latency_s=0.0)


def test_judge_verdicts_are_cached_and_resumable(tmp_results):
    runner.run("t", ["m"], ["lenient"], [0.0], 1, questions=[Q["u013"], Q["u021"]], bot_factory=HallucinatingBot)
    verdict = judge_json(claims=[("Exam-week hours are 7 AM to midnight.", False)])

    client = FakeJudgeClient([verdict, verdict])
    runner.grade_run("t", judge=Judge(model="j", client=client))
    assert client.calls == 2
    assert runner.judge_cache_path("t").exists()

    again = FakeJudgeClient([])
    runner.grade_run("t", judge=Judge(model="j", client=again))
    assert again.calls == 0  # all verdicts came from the cache

    other_model = FakeJudgeClient([verdict, verdict])
    runner.grade_run("t", judge=Judge(model="other", client=other_model))
    assert other_model.calls == 2  # cache is per judge model

    forced = FakeJudgeClient([verdict, verdict])
    runner.grade_run("t", judge=Judge(model="j", client=forced), rejudge=True)
    assert forced.calls == 2
    assert {g.label for g in runner.load_graded("t")} == {Label.HALLUCINATION}


def test_grade_run_and_agreement(tmp_results):
    qs = [Q["a001"], Q["u013"], Q["u049"]]
    runner.run("t", ["m"], ["strict"], [0.0], 1, questions=qs, bot_factory=FakeBot)
    runner.grade_run("t", judge=None)
    graded = {g.question_id: g.label for g in runner.load_graded("t")}
    assert graded == {"a001": Label.FALSE_REFUSAL, "u013": Label.CORRECT_REFUSAL, "u049": Label.CORRECT_REFUSAL}

    path = labels.make_label_sheet("t", n=3)
    with pytest.raises(FileExistsError):
        labels.make_label_sheet("t", n=3)
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert all(r["human_label"] == "" for r in rows)  # blind: grader labels are not in the sheet
    for r in rows:
        r["human_label"] = "correct_refusal"  # human disagrees on a001
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=labels.COLUMNS)
        w.writeheader()
        w.writerows(rows)

    a = labels.agreement("t")
    assert a.n == 3
    assert a.final_accuracy == pytest.approx(2 / 3)
    assert [d["question_id"] for d in a.disagreements] == ["a001"]
