import pytest
from statsmodels.stats.proportion import proportion_confint

from faq_eval.grader import Label
from faq_eval.metrics import Rate, category_metrics, paired_test, setting_metrics, wilson
from faq_eval.runner import GradedRecord


@pytest.mark.parametrize("k, n", [(0, 60), (3, 60), (30, 60), (57, 60), (60, 60), (1, 1)])
def test_wilson_matches_statsmodels(k, n):
    lo, hi = wilson(k, n)
    ref_lo, ref_hi = proportion_confint(k, n, method="wilson")
    assert lo == pytest.approx(ref_lo) and hi == pytest.approx(ref_hi)


def test_wilson_edges():
    lo, hi = wilson(0, 60)
    assert lo == 0.0 and 0 < hi < 0.07
    assert Rate(k=0, n=0).fmt() == "n/a"
    assert Rate(k=30, n=60).fmt() == "50% (38%–62%)"


def rec(model, prompt, qid, answerable, label, category="direct", sample=0, temperature=0.0):
    return GradedRecord(
        model=model, prompt=prompt, temperature=temperature, sample=sample, seed=sample, question_id=qid,
        category=category, answerable=answerable, question="q", answer="a", latency_s=0.0, timestamp="t",
        rule_label=label, label=label, method="rule", reason="",
    )


RECORDS = [
    rec("m", "strict", "a1", True, Label.CORRECT),
    rec("m", "strict", "a2", True, Label.FALSE_REFUSAL),
    rec("m", "strict", "u1", False, Label.CORRECT_REFUSAL, "near_miss"),
    rec("m", "strict", "u2", False, Label.CORRECT_REFUSAL, "personal"),
    rec("m", "lenient", "a1", True, Label.CORRECT),
    rec("m", "lenient", "a2", True, Label.CORRECT),
    rec("m", "lenient", "u1", False, Label.HALLUCINATION, "near_miss"),
    rec("m", "lenient", "u2", False, Label.HALLUCINATION, "personal"),
]


def test_setting_metrics():
    strict, lenient = sorted(setting_metrics(RECORDS), key=lambda m: m.prompt != "strict")
    assert (strict.accuracy.k, strict.accuracy.n) == (1, 2)
    assert strict.false_refusal.k == 1 and strict.refusal.p == 1.0 and strict.hallucination.k == 0
    assert strict.balanced == pytest.approx(0.75)
    assert lenient.accuracy.p == 1.0 and lenient.hallucination.p == 1.0 and lenient.balanced == pytest.approx(0.5)


def test_category_metrics_use_the_right_measure():
    cats = {(c.prompt, c.category): c for c in category_metrics(RECORDS)}
    assert cats[("strict", "direct")].rate.p == 0.5  # accuracy
    assert cats[("lenient", "near_miss")].rate.p == 1.0  # hallucination rate


def test_paired_test_counts_discordant_questions():
    t = paired_test(RECORDS, ("m", "lenient", 0.0), ("m", "strict", 0.0), Label.HALLUCINATION)
    assert (t.only_a, t.only_b) == (2, 0)
    assert t.p_value == pytest.approx(0.5)  # exact binomial, 2 vs 0
    same = paired_test(RECORDS, ("m", "strict", 0.0), ("m", "strict", 0.0), Label.HALLUCINATION)
    assert same.p_value == 1.0


def test_samples_are_pooled():
    rs = [rec("m", "strict", "a1", True, Label.CORRECT, sample=s, temperature=0.7) for s in range(3)]
    rs.append(rec("m", "strict", "a2", True, Label.WRONG, sample=0, temperature=0.7))
    (m,) = setting_metrics(rs)
    assert m.samples == 3 and (m.accuracy.k, m.accuracy.n) == (3, 4)


def test_build_report_writes_markdown_and_plots(tmp_path, monkeypatch):
    from faq_eval import labels, report, runner

    monkeypatch.setattr(runner, "GRADED_DIR", tmp_path / "graded")
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(labels, "LABELS_DIR", tmp_path / "labels")
    runner.GRADED_DIR.mkdir()
    with (runner.GRADED_DIR / "t.jsonl").open("w", encoding="utf-8") as f:
        for r in RECORDS:
            f.write(r.model_dump_json() + "\n")

    path = report.build_report("t")
    text = path.read_text(encoding="utf-8")
    assert "| m | strict | 50% (9%–91%)" in text
    assert "Strict vs lenient" in text and "Example hallucinations" in text
    assert (tmp_path / "reports" / "t" / "tradeoff.png").stat().st_size > 0
    assert (tmp_path / "reports" / "t" / "categories.png").stat().st_size > 0


def test_consistency_metrics():
    rs = [
        rec("m", "lenient", "u1", False, Label.HALLUCINATION, "near_miss", sample=0, temperature=1.0),
        rec("m", "lenient", "u1", False, Label.CORRECT_REFUSAL, "near_miss", sample=1, temperature=1.0),
        rec("m", "lenient", "u2", False, Label.CORRECT_REFUSAL, "near_miss", sample=0, temperature=1.0),
        rec("m", "lenient", "u2", False, Label.CORRECT_REFUSAL, "near_miss", sample=1, temperature=1.0),
        rec("m", "lenient", "a1", True, Label.CORRECT, sample=0, temperature=1.0),
        rec("m", "lenient", "a1", True, Label.WRONG, sample=1, temperature=1.0),
        rec("m", "lenient", "a1", True, Label.CORRECT, temperature=0.0),  # one sample: skipped
    ]
    from faq_eval.metrics import consistency_metrics

    (c,) = consistency_metrics(rs)
    assert c.temperature == 1.0 and c.samples == 2
    assert (c.consistent.k, c.consistent.n) == (1, 3)
    assert (c.any_hallucination.k, c.any_hallucination.n) == (1, 2)
    assert (c.any_error_answerable.k, c.any_error_answerable.n) == (1, 1)


def test_build_report_with_temperatures(tmp_path, monkeypatch):
    from faq_eval import labels, report, runner

    monkeypatch.setattr(runner, "GRADED_DIR", tmp_path / "graded")
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(labels, "LABELS_DIR", tmp_path / "labels")
    runner.GRADED_DIR.mkdir()
    hot = [r.model_copy(update={"temperature": 1.0, "sample": s}) for r in RECORDS for s in range(3)]
    with (runner.GRADED_DIR / "t.jsonl").open("w", encoding="utf-8") as f:
        for r in RECORDS + hot:
            f.write(r.model_dump_json() + "\n")

    text = report.build_report("t").read_text(encoding="utf-8")
    assert "## Headline (temperature 0)" in text and "## Temperature" in text
    assert "| t=0 | t=1 |" in text
    assert "hallucinated in at least one sample" in text
    assert (tmp_path / "reports" / "t" / "temperature.png").stat().st_size > 0
