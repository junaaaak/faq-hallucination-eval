import pytest

from faq_eval.matcher import extract_numbers, extract_times, match_alternative, match_facts


@pytest.mark.parametrize(
    "alt, answer",
    [
        ("9:00 PM", "The library closes at 9 pm on weekdays."),
        ("9:00 PM", "It is open until 21:00."),
        ("9:00 PM", "Open 8:00 AM to 9:00 PM."),
        ("10:30 PM", "Curfew is 10:30 p.m."),
        ("4 August 2026", "Classes begin on August 4, 2026."),
        ("4 August 2026", "The year starts 4th August 2026."),
        ("1,20,000", "Tuition is ₹120,000 per semester."),
        ("1,20,000", "It costs Rs. 1.2 lakh each semester."),
        ("7.2 lakh", "The median package was ₹7,20,000 per year."),
        ("7.2 lakh", "Median was 7.2 LPA."),
        ("75%", "You need 75 percent attendance."),
        ("₹5", "The fine is Rs.5 per book per day."),
        ("5.0", "The minimum CGPA to graduate is 5."),
        ("1800-200-4455", "Call 1800 200 4455."),
        ("exams@northbridge.example.edu", "Email exams@northbridge.example.edu."),
        ("Meera Kulkarni", "The director is Dr. Meera Kulkerni."),
        ("closed", "The library is CLOSED on Sundays."),
        ("4", "Undergraduates can borrow up to 4 books."),
    ],
)
def test_matches(alt, answer):
    assert match_alternative(alt, answer)


@pytest.mark.parametrize(
    "alt, answer",
    [
        ("9:00 PM", "The library closes at 9:00 AM."),
        ("9:00 PM", "The bus pass costs 9,000."),  # number, not a time
        ("4", "You can borrow books for 14 days."),  # no partial-digit matches
        ("4", "The campus is 42 acres."),
        ("1,20,000", "Tuition is ₹1,02,000."),
        ("180", "There are 1800 seats."),
        ("7.5", "You need a CGPA of 7.0."),
        ("4 August 2026", "Classes begin on 14 August 2026."),
        ("Kaveri", "Girls stay in Nilgiri."),
        ("closed", "The library is open on Sundays."),
        ("ATM", "There is a bank near the gate."),  # whole-word, not inside other words
    ],
)
def test_non_matches(alt, answer):
    assert not match_alternative(alt, answer)


def test_times_are_not_numbers():
    times, _ = extract_times("8:00 AM to 9:00 PM")
    assert times == {8 * 60, 21 * 60}
    assert extract_numbers("8:00 AM to 9:00 PM, 13 hours") == {13.0}


def test_indian_grouping_and_lakh():
    assert extract_numbers("₹2,22,000 or 2.22 lakh") == {222000.0}


def test_match_facts_needs_every_group():
    required = [["6:00 AM"], ["8:00 AM"], ["5:00 PM"], ["7:00 PM"]]
    full = match_facts(required, "Open 6-8 AM? No: 6:00 AM to 8:00 AM and 5:00 PM to 7:00 PM.")
    partial = match_facts(required, "The pool opens at 6:00 AM and closes at 8:00 AM.")
    assert full.all_matched
    assert not partial.all_matched and partial.any_matched
    assert partial.missing == [["5:00 PM"], ["7:00 PM"]]
