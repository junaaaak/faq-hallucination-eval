"""Check whether an answer contains the expected facts.

Each expected fact is a group of alternatives; a group is satisfied if any alternative matches.
Alternatives are typed so formatting differences don't cause misses:

- time:   "9:00 PM" matches "9 PM", "9 pm" and "21:00"
- date:   "4 August 2026" matches "August 4, 2026" and "4th August 2026"
- number: "1,20,000" matches "120,000", "₹1.2 lakh"; "75%" matches "75 percent"
- phone:  "1800-200-4455" matches "1800 200 4455"
- text:   case-insensitive whole-word match, with fuzzy matching for multi-word names
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz

MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"), ("december", "dec"),
        ],
        start=1,
    )
    for m in names
}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))

_TIME_12H = re.compile(r"(?<!\d)(\d{1,2})(?::([0-5]\d))?\s*([ap])\.?\s?m\b\.?", re.I)
_TIME_24H = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_DATE_DMY = re.compile(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})", re.I)
_DATE_MDY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", re.I)
_NUMBER = re.compile(
    r"(?<!\d)(?<!\d\.)(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?(?:\s*(lakhs?|lacs?|lpa|crores?)\b)?",
    re.I,
)
_PHONE_RUN = re.compile(r"\+?\d[\d\s\-()]{5,}\d")
_CURRENCY = r"(?:₹|rs\.?|inr)"
_NUMBER_ALT = re.compile(
    rf"^\s*{_CURRENCY}?\s*[\d.,]+\s*(?:%|percent|per cent|lakhs?|lacs?|lpa|crores?|rupees)?\s*$",
    re.I,
)
_MULTIPLIERS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "lpa": 1e5, "crore": 1e7, "crores": 1e7}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[*_`#]+", "", text)  # markdown emphasis, e.g. "is **not specified**"
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_times(text: str) -> tuple[set[int], str]:
    """Return times as minutes after midnight, plus the text with times masked out."""
    found: set[int] = set()

    def _12h(m: re.Match) -> str:
        hour, minute, half = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
        if 1 <= hour <= 12:
            hour = hour % 12 + (12 if half == "p" else 0)
            found.add(hour * 60 + minute)
        return " "

    def _24h(m: re.Match) -> str:
        found.add(int(m.group(1)) * 60 + int(m.group(2)))
        return " "

    masked = _TIME_12H.sub(_12h, text)
    masked = _TIME_24H.sub(_24h, masked)
    return found, masked


def extract_dates(text: str) -> set[tuple[int, int, int]]:
    found = set()
    for m in _DATE_DMY.finditer(text):
        found.add((int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1))))
    for m in _DATE_MDY.finditer(text):
        found.add((int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))))
    return found


def extract_numbers(text: str) -> set[float]:
    """Numbers in the text, with Indian/Western grouping removed and lakh/crore expanded. Times are ignored."""
    _, masked = extract_times(text)
    found = set()
    for m in _NUMBER.finditer(masked):
        value = float(m.group(1).replace(",", "") + ("." + m.group(2) if m.group(2) else ""))
        if m.group(3):
            value *= _MULTIPLIERS[m.group(3).lower()]
        found.add(round(value, 6))
    return found


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _alt_kind(alt: str) -> str:
    if "@" in alt:
        return "text"
    if re.fullmatch(r"\+?[\d\s\-()]+", alt) and len(_digits(alt)) >= 7:
        return "phone"
    if _TIME_12H.fullmatch(alt.strip()) or _TIME_24H.fullmatch(alt.strip()):
        return "time"
    if _DATE_DMY.fullmatch(alt.strip()) or _DATE_MDY.fullmatch(alt.strip()):
        return "date"
    if _NUMBER_ALT.match(alt):
        return "number"
    return "text"


def match_alternative(alt: str, answer: str) -> bool:
    kind = _alt_kind(alt)
    if kind == "time":
        return bool(extract_times(alt)[0] & extract_times(answer)[0])
    if kind == "date":
        return bool(extract_dates(alt) & extract_dates(answer))
    if kind == "number":
        alt_values = extract_numbers(re.sub(r"(?i)rupees|percent|per cent|%", "", alt))
        return bool(alt_values & extract_numbers(answer))
    if kind == "phone":
        target = _digits(alt)
        return any(_digits(run).endswith(target) for run in _PHONE_RUN.findall(answer))

    alt_n, ans_n = normalise(alt), normalise(answer)
    if re.search(rf"(?<!\w){re.escape(alt_n)}(?!\w)", ans_n):
        return True
    # Fuzzy match for multi-word names, e.g. "Meera Kulkarni" vs "Meera Kulkerni".
    words = alt_n.split()
    if len(words) >= 2 and len(alt_n) >= 8 and all(w.isalpha() for w in words):
        return fuzz.partial_ratio(alt_n, ans_n) >= 90
    return False


@dataclass
class FactMatch:
    matched: list[list[str]] = field(default_factory=list)
    missing: list[list[str]] = field(default_factory=list)

    @property
    def all_matched(self) -> bool:
        return not self.missing

    @property
    def any_matched(self) -> bool:
        return bool(self.matched)


def match_facts(required: list[list[str]], answer: str) -> FactMatch:
    result = FactMatch()
    for group in required:
        (result.matched if any(match_alternative(a, answer) for a in group) else result.missing).append(group)
    return result


def is_structured_group(group: list[str]) -> bool:
    """True if every alternative is a number, time, date or phone, so a miss is a confident miss."""
    return all(_alt_kind(a) in {"number", "time", "date", "phone"} for a in group)


def contains_any(phrases: list[str], answer: str) -> str | None:
    """First phrase found in the answer (case-insensitive, whole words), if any."""
    ans_n = normalise(answer)
    for p in phrases:
        if re.search(rf"(?<!\w){re.escape(normalise(p))}(?!\w)", ans_n):
            return p
    return None
