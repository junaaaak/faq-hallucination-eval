"""Question schema and loaders for the fact sheet and question set."""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FACT_SHEET_PATH = DATA_DIR / "fact_sheet.md"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"


class Category(str, Enum):
    # answerable
    DIRECT = "direct"
    PARAPHRASED = "paraphrased"
    MULTI_FACT = "multi_fact"
    # unanswerable
    OUT_OF_SCOPE = "out_of_scope"
    NEAR_MISS = "near_miss"
    FALSE_PREMISE = "false_premise"
    PERSONAL = "personal"


ANSWERABLE_CATEGORIES = {Category.DIRECT, Category.PARAPHRASED, Category.MULTI_FACT}


class Question(BaseModel):
    id: str
    question: str
    answerable: bool
    category: Category
    # All groups must be present in the answer; any alternative within a group counts.
    required_facts: list[list[str]] = Field(default_factory=list)
    # For false-premise questions: phrases showing the bot corrected the premise.
    premise_corrections: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "Question":
        if self.answerable != (self.category in ANSWERABLE_CATEGORIES):
            raise ValueError(f"{self.id}: category {self.category.value} does not match answerable={self.answerable}")
        if self.answerable and not self.required_facts:
            raise ValueError(f"{self.id}: answerable question needs required_facts")
        if self.answerable and any(not group for group in self.required_facts):
            raise ValueError(f"{self.id}: empty alternative group in required_facts")
        if not self.answerable and self.required_facts:
            raise ValueError(f"{self.id}: unanswerable question must not have required_facts")
        if self.category == Category.FALSE_PREMISE and not self.premise_corrections:
            raise ValueError(f"{self.id}: false-premise question needs premise_corrections")
        if self.category != Category.FALSE_PREMISE and self.premise_corrections:
            raise ValueError(f"{self.id}: premise_corrections only apply to false-premise questions")
        return self


def load_fact_sheet(path: Path = FACT_SHEET_PATH) -> str:
    return path.read_text(encoding="utf-8")


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    questions: list[Question] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(Question.model_validate(json.loads(line)))
            except Exception as exc:  # re-raise with line context
                raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
    duplicates = [qid for qid, n in Counter(q.id for q in questions).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate question ids: {duplicates}")
    return questions


def summarise(questions: list[Question]) -> dict[str, int]:
    """Count of questions per category, in enum order."""
    counts = Counter(q.category for q in questions)
    return {c.value: counts.get(c, 0) for c in Category}
