"""Rates with Wilson 95% confidence intervals, per setting and per question category.

A setting is (model, prompt, temperature). When a setting has several samples per question, rates
are pooled over all samples; the intervals then treat samples as independent, which understates
the uncertainty a little (answers to the same question are correlated).

Rates:
- accuracy        correct / answerable
- false_refusal   false_refusal / answerable
- wrong           wrong / answerable
- refusal         correct_refusal / unanswerable
- hallucination   hallucination / unanswerable
- balanced        mean of accuracy and refusal (a bot that always refuses, or never does, scores ~50%)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from pydantic import BaseModel
from statsmodels.stats.contingency_tables import mcnemar

from faq_eval.grader import Label
from faq_eval.runner import GradedRecord

Z95 = 1.959963984540054


class Rate(BaseModel):
    k: int
    n: int

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.k, self.n)

    def fmt(self, digits: int = 0) -> str:
        if not self.n:
            return "n/a"
        lo, hi = self.ci
        return f"{self.p:.{digits}%} ({lo:.{digits}%}–{hi:.{digits}%})"


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n. Behaves sensibly at 0 and n, unlike the normal one."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(records: Sequence[GradedRecord], label: Label) -> Rate:
    return Rate(k=sum(r.label == label for r in records), n=len(records))


class SettingMetrics(BaseModel):
    model: str
    prompt: str
    temperature: float
    samples: int
    accuracy: Rate
    false_refusal: Rate
    wrong: Rate
    refusal: Rate
    hallucination: Rate

    @property
    def balanced(self) -> float:
        return (self.accuracy.p + self.refusal.p) / 2

    @property
    def key(self) -> tuple[str, str, float]:
        return (self.model, self.prompt, self.temperature)


def setting_metrics(records: Iterable[GradedRecord]) -> list[SettingMetrics]:
    groups: dict[tuple, list[GradedRecord]] = {}
    for r in records:
        groups.setdefault((r.model, r.prompt, r.temperature), []).append(r)
    out = []
    for (model, prompt, temp), rs in groups.items():
        ans = [r for r in rs if r.answerable]
        una = [r for r in rs if not r.answerable]
        out.append(SettingMetrics(
            model=model, prompt=prompt, temperature=temp, samples=len({r.sample for r in rs}),
            accuracy=_rate(ans, Label.CORRECT), false_refusal=_rate(ans, Label.FALSE_REFUSAL),
            wrong=_rate(ans, Label.WRONG), refusal=_rate(una, Label.CORRECT_REFUSAL),
            hallucination=_rate(una, Label.HALLUCINATION),
        ))
    return sorted(out, key=lambda m: m.key)


class CategoryRate(BaseModel):
    model: str
    prompt: str
    temperature: float
    category: str
    answerable: bool
    rate: Rate  # accuracy for answerable categories, hallucination rate for unanswerable ones


def category_metrics(records: Iterable[GradedRecord]) -> list[CategoryRate]:
    groups: dict[tuple, list[GradedRecord]] = {}
    for r in records:
        groups.setdefault((r.model, r.prompt, r.temperature, r.category, r.answerable), []).append(r)
    out = []
    for (model, prompt, temp, cat, answerable), rs in groups.items():
        label = Label.CORRECT if answerable else Label.HALLUCINATION
        out.append(CategoryRate(model=model, prompt=prompt, temperature=temp, category=cat,
                                answerable=answerable, rate=_rate(rs, label)))
    return sorted(out, key=lambda c: (c.model, c.prompt, c.temperature, not c.answerable, c.category))


class PairedTest(BaseModel):
    """McNemar's exact test on the same questions answered under two settings."""
    a: tuple[str, str, float]
    b: tuple[str, str, float]
    measure: str
    only_a: int  # questions where the event happened under a but not b
    only_b: int
    p_value: float


def paired_test(records: Iterable[GradedRecord], a: tuple, b: tuple, label: Label) -> PairedTest:
    """Compare how often `label` occurs under settings a and b, pairing by (question, sample)."""
    by_key: dict[tuple, dict[tuple, bool]] = {a: {}, b: {}}
    for r in records:
        s = (r.model, r.prompt, r.temperature)
        if s in by_key:
            by_key[s][(r.question_id, r.sample)] = r.label == label
    common = by_key[a].keys() & by_key[b].keys()
    only_a = sum(by_key[a][k] and not by_key[b][k] for k in common)
    only_b = sum(by_key[b][k] and not by_key[a][k] for k in common)
    both = sum(by_key[a][k] and by_key[b][k] for k in common)
    neither = len(common) - only_a - only_b - both
    p = 1.0 if only_a + only_b == 0 else float(mcnemar([[both, only_a], [only_b, neither]], exact=True).pvalue)
    return PairedTest(a=a, b=b, measure=label.value, only_a=only_a, only_b=only_b, p_value=p)
