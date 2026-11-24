import pytest

from faq_eval.dataset import Category, Question, load_fact_sheet, load_questions, summarise


@pytest.fixture(scope="module")
def questions() -> list[Question]:
    return load_questions()


@pytest.fixture(scope="module")
def fact_sheet() -> str:
    return load_fact_sheet()


def test_set_is_balanced(questions):
    answerable = [q for q in questions if q.answerable]
    assert len(questions) == 120
    assert len(answerable) == 60


def test_hard_unanswerable_share(questions):
    unanswerable = [q for q in questions if not q.answerable]
    hard = [q for q in unanswerable if q.category in {Category.NEAR_MISS, Category.FALSE_PREMISE}]
    assert len(hard) / len(unanswerable) >= 0.6


def test_every_category_present(questions):
    assert all(n > 0 for n in summarise(questions).values())


def test_ids_match_answerability(questions):
    for q in questions:
        assert q.id.startswith("a" if q.answerable else "u"), q.id


@pytest.mark.parametrize("category", [Category.DIRECT, Category.PARAPHRASED])
def test_lookup_facts_appear_in_sheet(questions, fact_sheet, category):
    """Every required-fact group for a lookup question has at least one alternative quoted in the sheet."""
    sheet = fact_sheet.lower()
    for q in (q for q in questions if q.category == category):
        for group in q.required_facts:
            assert any(alt.lower() in sheet for alt in group), f"{q.id}: none of {group} in fact sheet"


def test_rejects_inconsistent_question():
    with pytest.raises(ValueError):
        Question(id="x", question="?", answerable=True, category=Category.NEAR_MISS, required_facts=[["a"]])
    with pytest.raises(ValueError):
        Question(id="x", question="?", answerable=False, category=Category.FALSE_PREMISE)
