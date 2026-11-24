"""Grader tests. Most answers are real llama3.2:3b outputs from the phase 3 smoke run."""

import pytest

from faq_eval.dataset import load_questions
from faq_eval.grader import Label, grade
from faq_eval.prompts import REFUSAL_PHRASE
from faq_eval.refusal import detect_refusal

Q = {q.id: q for q in load_questions()}

LENIENT_AC = (
    "The hostel fee for an AC room is not explicitly mentioned in the provided fact sheet. "
    "However, it is mentioned that the hostel fee (non-AC, twin sharing) is ₹45,000 per semester."
)
LENIENT_ATTENDANCE = (
    "I don't have any information about your attendance percentage. As a student at Northbridge Institute "
    "of Technology, you would need to check your student portal or speak with your department's attendance "
    "officer to find out your current attendance percentage."
)
LENIENT_WORLD_CUP = (
    "I'm not aware of any information about the Northbridge Institute of Technology or its involvement in the "
    "FIFA World Cup. My training data only includes information up to 2023, and I don't have real-time access "
    "to current events. However, I can tell you that the 2022 FIFA World Cup was won by Argentina, who defeated "
    "France 4-2 in a penalty shootout after the match ended 3-3 after extra time."
)
LENIENT_EXAM_WEEK = (
    "During exam week, the library hours are as follows:\n- Library hours: 8:00 AM to 9:00 PM, Monday to Friday; "
    "9:00 AM to 5:00 PM on Saturday.\n- However, during exam week, the library hours are extended to:\n"
    "  - 8:00 AM to 10:00 PM, Monday to Friday.\n  - 9:00 AM to 6:00 PM on Saturday."
)
LENIENT_FIRST_SEM = (
    "Total cost for the first semester: ₹25,000 + ₹1,20,000 + ₹45,000 + ₹22,000 + ₹10,000 = ₹2,22,000\n\n"
    "So, a new B.Tech hosteller pays ₹2,22,000 in the first semester."
)


@pytest.mark.parametrize(
    "qid, answer, expected",
    [
        # answerable
        ("a001", "The library is open from 8:00 AM to 9:00 PM, Monday to Friday.", Label.CORRECT),
        ("a048", LENIENT_FIRST_SEM, Label.CORRECT),
        ("a048", REFUSAL_PHRASE, Label.FALSE_REFUSAL),
        ("a001", "The library closes at 8:00 PM on weekdays.", Label.WRONG),
        ("a021", "The pool is open from 6:00 AM to 8:00 AM.", Label.UNCERTAIN),
        ("a029", "No. All fees must be paid online.", Label.UNCERTAIN),
        # unanswerable
        ("u013", REFUSAL_PHRASE, Label.CORRECT_REFUSAL),
        ("u013", LENIENT_EXAM_WEEK, Label.HALLUCINATION),
        ("u037", REFUSAL_PHRASE, Label.CORRECT_REFUSAL),
        ("u037", LENIENT_AC, Label.CORRECT_REFUSAL),
        ("u049", LENIENT_ATTENDANCE, Label.CORRECT_REFUSAL),
        ("u001", REFUSAL_PHRASE, Label.CORRECT_REFUSAL),
        ("u001", LENIENT_WORLD_CUP, Label.UNCERTAIN),
        ("u001", "Argentina won the 2022 FIFA World Cup.", Label.HALLUCINATION),
        ("u021", "The head of the CSE department is Dr. Rajesh Iyer.", Label.HALLUCINATION),
        ("u041", "The library is closed on Sundays.", Label.CORRECT_REFUSAL),
        # from the full llama3.2:3b lenient run
        (
            "u019",
            "I don't have the specific email address of the Placement Cell in the provided fact sheet. However, I "
            "can suggest that you check the student portal or contact the Admissions office for the correct email.",
            Label.CORRECT_REFUSAL,
        ),
        (
            "u022",
            "I don't have the exact information on the number of faculty members in the ECE department.",
            Label.CORRECT_REFUSAL,
        ),
        (
            "u051",
            "To check if your fee payment has been received, I recommend checking the following:\n1. Student "
            "Portal: Log in to your student portal account and check the Payment History section.",
            Label.UNCERTAIN,
        ),
        (
            "u059",
            "To check if you've been placed, I'll need to know a few details. Can you please tell me: 1. What is "
            "your program (B.Tech or M.Tech)?",
            Label.UNCERTAIN,
        ),
        ("u040", "According to the fact sheet, a single-occupancy hostel room costs ₹45,000 per semester.", Label.HALLUCINATION),
        (
            "u027",
            "No, the NET application fee is not waived for SC/ST candidates. The application fee is ₹1,500.",
            Label.HALLUCINATION,
        ),
    ],
)
def test_grade(qid, answer, expected):
    assert grade(Q[qid], answer).label == expected


@pytest.mark.parametrize(
    "answer",
    [
        REFUSAL_PHRASE,
        "I do not know.",
        "That information is not mentioned in the fact sheet.",
        "The fact sheet does not specify the exam-week hours.",
        "Sorry, I can't find that in the information provided.",
        "That is outside the scope of what I can answer.",
        "I don't have access to your personal records.",
        "The exam-week timings aren't specified anywhere.",
        "The sheet doesn't mention a dress code.",
    ],
)
def test_detects_refusals(answer):
    assert detect_refusal(answer).is_refusal


@pytest.mark.parametrize(
    "answer",
    [
        "The library is closed on Sundays.",
        "No, cash is not accepted; pay through the student portal.",
        "The Mathematics and Humanities department does not offer its own degree.",
        "A student can receive only one institute scholarship at a time.",
        "Wi-Fi is available in all hostel rooms.",
        "The fee is provided in two instalments and does include the exam fee.",
    ],
)
def test_ordinary_negatives_are_not_refusals(answer):
    assert not detect_refusal(answer).is_refusal


def test_exact_and_hedged_flags():
    assert detect_refusal(REFUSAL_PHRASE).exact
    assert detect_refusal(LENIENT_WORLD_CUP).hedged
    assert not detect_refusal(LENIENT_ATTENDANCE).hedged
