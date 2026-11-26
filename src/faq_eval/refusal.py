"""Detect when the bot declines to answer.

The strict prompt asks for one fixed sentence, but models (and the lenient prompt) phrase refusals
many ways, so a phrase list is used as well. A refusal followed by "however ..." plus more content
is flagged as hedged: it may be harmless context or a hallucinated answer, so it goes to the judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from faq_eval.matcher import normalise
from faq_eval.prompts import REFUSAL_PHRASE

_NOT = r"(?:not|n't|never)"
_INFO = r"(?:information|info|details?|data|records?|access|email(?: address)?|numbers?|figures?|list|answer)"

REFUSAL_PATTERNS = [
    r"\bi (?:do not|don't|dont) know\b",
    # "I don't have (the|any) (specific|exact ...) information/email/number ..." with up to 3 words between
    rf"\b(?:i|we) (?:do|did|does)?\s?{_NOT}\s+have (?:[\w'-]+ ){{0,3}}?{_INFO}\b",
    rf"\b(?:i|we) (?:have|has) no {_INFO}\b",
    rf"\bno (?:specific |such |relevant )?{_INFO} (?:is |was )?(?:available|provided|given|about|on|regarding|in)\b",
    rf"\b(?:is|are|was|were)(?:n't| not)\s+(?:explicitly |specifically |clearly )?(?:mentioned|specified|stated|provided|listed|included|given|available|covered)\b",
    rf"\b(?:does|do|did)(?:n't| not)\s+(?:explicitly |specifically )?(?:contain|include|mention|specify|provide|state|list|cover|say)\b",
    rf"\b(?:i'm|i am) {_NOT}\s?(?:aware|sure|able)\b",
    r"\b(?:unable|not able) to (?:find|provide|answer|access|determine|confirm|share)\b",
    r"\b(?:can't|cannot|can not|couldn't) (?:find|answer|provide|access|determine|confirm|say|tell|share)\b",
    r"\b(?:outside|beyond) (?:the )?scope\b",
    r"\bnot (?:in|part of|found in) (?:the |this |my )?(?:fact sheet|information|data)\b",
    r"\bplease (?:contact|check with|reach out to|refer to)\b",
]
_REFUSAL_RE = [re.compile(p) for p in REFUSAL_PATTERNS]
_HEDGE_RE = re.compile(r"\b(?:however|but|that said|although|though|nevertheless|nonetheless)\b")

# Sending the student elsewhere instead of answering. Not a hallucination by itself, but not a
# clean refusal either: these go to the judge.
_REDIRECT_VERB = (
    r"(?:check|contact|ask|visit|speak|reach out|log in|email|call|refer to)"
    r"|(?:checking|contacting|asking|visiting|speaking|reaching out|logging in|emailing|calling|referring to)"
)
_REDIRECT_RE = re.compile(
    r"\b(?:i (?:would |can |could |'d )?(?:recommend|suggest|advise)|you (?:would |may |might |will |should )?(?:need|want|have) to|"
    rf"(?:please|you can|you could|you should|try to)) (?:\w+ ){{0,2}}?(?:{_REDIRECT_VERB})\b"
)
# Asking the student for details, which implies an answer exists if they reply.
_CLARIFY_RE = re.compile(
    r"\b(?:(?:could|can|would) you (?:please )?(?:tell|share|provide|let me know|confirm|specify)"
    r"|please (?:provide|share|tell)|i (?:would )?need (?:to know|more|some|your|a (?:little|bit) more)|i'll need (?:to know|more|some|your))\b"
)


@dataclass(frozen=True)
class RefusalResult:
    is_refusal: bool
    exact: bool = False  # the answer is the fixed refusal sentence and nothing else
    hedged: bool = False  # refusal followed by "however/but ..." and further content
    redirect: bool = False  # tells the student to check/contact somewhere else
    clarifying: bool = False  # asks the student for more details
    pattern: str | None = None


def detect_refusal(answer: str) -> RefusalResult:
    text = normalise(answer)
    if not text:
        return RefusalResult(is_refusal=False)
    if text.rstrip(". ") == normalise(REFUSAL_PHRASE).rstrip(". "):
        return RefusalResult(is_refusal=True, exact=True, pattern="exact")

    redirect = bool(_REDIRECT_RE.search(text))
    clarifying = bool(_CLARIFY_RE.search(text))

    first = None
    for pattern in _REFUSAL_RE:
        m = pattern.search(text)
        if m and (first is None or m.start() < first[0]):
            first = (m.start(), m.end(), pattern.pattern)
    if first is None:
        return RefusalResult(is_refusal=False, redirect=redirect, clarifying=clarifying)

    tail = text[first[1]:]
    hedge = _HEDGE_RE.search(tail)
    # Only count it as hedged if there is real content after the hedge word.
    hedged = bool(hedge and len(tail[hedge.end():].split()) >= 4)
    # A hedge that only leads into a redirect ("however, please contact the office") is still a refusal.
    if hedged and redirect and not _HEDGE_RE.search(tail[hedge.end():]):
        after = tail[hedge.end():]
        if _REDIRECT_RE.search(after) and len(after.split()) <= 40:
            hedged = False
    return RefusalResult(is_refusal=True, hedged=hedged, redirect=redirect, clarifying=clarifying, pattern=first[2])
