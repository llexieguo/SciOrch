from __future__ import annotations

import ast
import re
from typing import Any

from sciorch.core.parsing import extract_unique_boxed_letter


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OPTION_PREFIX_RE = re.compile(r"^\s*(?:\(([A-Z])\)|([A-Z])[.)])\s*")
ANSWER_RE = re.compile(r"(?:answer|final answer)\s*[:：]\s*\(?\s*([A-Z])\s*\)?", re.IGNORECASE)


def normalize_answer_letter(answer: Any, options_len: int) -> str | None:
    value = answer
    if isinstance(answer, str):
        text = answer.strip()
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list) and parsed:
                value = parsed[0]
            else:
                value = text
        else:
            value = text

    if isinstance(value, int):
        return LETTERS[value] if 0 <= value < min(options_len, len(LETTERS)) else None

    if isinstance(value, str):
        text = value.strip().upper()
        if text.isdigit():
            index = int(text)
            return LETTERS[index] if 0 <= index < min(options_len, len(LETTERS)) else None
        match = re.search(r"[A-Z]", text)
        if match:
            letter = match.group(0)
            max_letter = LETTERS[min(options_len, len(LETTERS)) - 1] if options_len else "Z"
            return letter if "A" <= letter <= max_letter else None
    return None


def strip_option_prefix(option: str) -> str:
    return OPTION_PREFIX_RE.sub("", str(option).strip(), count=1).strip()


def format_options(options: list[Any]) -> str:
    lines = []
    for index, option in enumerate(options):
        label = LETTERS[index]
        lines.append(f"({label}) {strip_option_prefix(str(option))}")
    return "\n".join(lines)


def format_query(sample: dict[str, Any]) -> str:
    question = str(sample.get("question", "")).strip()
    options = sample.get("options") or []
    if not isinstance(options, list):
        raise ValueError(f"Sample {sample.get('id')} has non-list options")

    prompt = question
    prompt += "\n\nChoose the correct answer from the following options:\n"
    prompt += format_options(options)
    prompt += "\n\nReason carefully, then state the final answer exactly as \\boxed{LETTER}."
    return prompt


def extract_answer_letter(text: str, options_len: int) -> tuple[str | None, str | None]:
    boxed, error = extract_unique_boxed_letter(text)
    if boxed is not None:
        return boxed, None

    match = ANSWER_RE.search(text or "")
    if match:
        letter = match.group(1).upper()
        max_letter = LETTERS[min(options_len, len(LETTERS)) - 1] if options_len else "Z"
        if "A" <= letter <= max_letter:
            return letter, None

    return None, error or "No final answer letter found"
