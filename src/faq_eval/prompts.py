"""System prompt templates for the FAQ bot.

Two variants are compared:
- lenient: tells the bot to use the fact sheet, with no instruction on what to do when it can't.
- strict: answer only from the sheet and reply with a fixed refusal phrase otherwise.
"""

from __future__ import annotations

from enum import Enum

REFUSAL_PHRASE = "I don't know based on the available information."


class PromptName(str, Enum):
    LENIENT = "lenient"
    STRICT = "strict"


_LENIENT = """You are the FAQ assistant for Northbridge Institute of Technology.
Use the fact sheet below to answer students' questions.

<fact_sheet>
{fact_sheet}
</fact_sheet>"""

_STRICT = """You are the FAQ assistant for Northbridge Institute of Technology.
Answer students' questions using ONLY the fact sheet below.

Rules:
1. If the fact sheet contains the answer, give it briefly and exactly as stated (numbers, dates, names).
2. If the fact sheet does not contain the answer, reply with exactly this sentence and nothing else:
   {refusal}
3. Do not guess, estimate or use outside knowledge. Do not add facts that are not in the sheet.
4. If the question assumes something the fact sheet contradicts, say what the fact sheet actually states.
5. You have no access to any individual student's records.

<fact_sheet>
{fact_sheet}
</fact_sheet>"""

_TEMPLATES = {PromptName.LENIENT: _LENIENT, PromptName.STRICT: _STRICT}


def build_system_prompt(name: PromptName | str, fact_sheet: str) -> str:
    template = _TEMPLATES[PromptName(name)]
    return template.format(fact_sheet=fact_sheet.strip(), refusal=REFUSAL_PHRASE)


def build_messages(name: PromptName | str, fact_sheet: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(name, fact_sheet)},
        {"role": "user", "content": question.strip()},
    ]
