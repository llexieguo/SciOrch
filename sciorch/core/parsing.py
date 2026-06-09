from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{\s*(?:\\text\{\s*)?([A-Za-z])\s*\}?\s*\}")
CONFIDENCE_PATTERN = re.compile(r"confidence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
THINK_BLOCK_PATTERN = re.compile(r"<think>\s*[\s\S]*?\s*</think>", re.IGNORECASE)


def normalize_boxed_text(text: str) -> str:
    if not text:
        return ""
    # JSON strings like "\boxed{A}" decode \b as a backspace character.
    return text.replace("\x08oxed", "\\boxed")


def strip_thinking_blocks(text: str) -> str:
    if not text:
        return ""
    return THINK_BLOCK_PATTERN.sub("", text).strip()


def iter_json_object_candidates(text: str) -> list[str]:
    if not text:
        return []
    candidates: list[str] = []
    stack: list[int] = []
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            stack.append(index)
            continue
        if char == "}" and stack:
            start = stack.pop()
            if not stack:
                candidates.append(text[start : index + 1].strip())
    return candidates


def _repair_json_escapes(s: str) -> str:
    """Fix invalid backslash escapes in JSON strings produced by LLMs.

    Models sometimes emit raw LaTeX inside JSON string values.  This walks
    the string and doubles every lone backslash that is NOT already part of
    a valid JSON escape sequence (``\\``, ``\"``, ``\/``, ``\b``, ``\f``,
    ``\n``, ``\r``, ``\t``, ``\\uXXXX``).
    """
    _VALID_AFTER = set('"\\/bfnrtu')
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        # We have a backslash at position i
        if i + 1 >= len(s):
            out.append("\\\\")  # lone trailing backslash -> double it
            i += 1
            continue
        nxt = s[i + 1]
        if nxt in _VALID_AFTER:
            # Already a valid JSON escape — keep as-is
            out.append(ch)
            out.append(nxt)
            i += 2
        else:
            # Invalid escape — double the backslash
            out.append("\\\\")
            # Don't consume nxt; it will be processed next iteration
            i += 1
    return "".join(out)


def parse_json_fragment(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    candidates: list[str] = []
    cleaned = strip_thinking_blocks(text)
    block = JSON_BLOCK_PATTERN.search(text)
    if block:
        candidates.append(block.group(1).strip())
    cleaned_block = JSON_BLOCK_PATTERN.search(cleaned)
    if cleaned_block:
        candidates.append(cleaned_block.group(1).strip())

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    cleaned_stripped = cleaned.strip()
    if cleaned_stripped:
        candidates.append(cleaned_stripped)

    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start : end + 1])

    candidates.extend(iter_json_object_candidates(text))
    candidates.extend(iter_json_object_candidates(cleaned))

    seen: set[str] = set()
    deduped_candidates: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_candidates.append(normalized)

    for candidate in reversed(deduped_candidates):
        try:
            loaded = json.loads(candidate)
        except Exception:
            # Try repairing invalid backslash escapes (e.g. LaTeX in JSON)
            try:
                loaded = json.loads(_repair_json_escapes(candidate))
            except Exception:
                continue
        if isinstance(loaded, dict):
            return loaded

    return None


def extract_boxed_letters(text: str) -> list[str]:
    if not text:
        return []
    return [match.upper() for match in BOXED_PATTERN.findall(normalize_boxed_text(text))]


def extract_unique_boxed_letter(text: str) -> tuple[str | None, str | None]:
    letters = extract_boxed_letters(text)
    if not letters:
        return None, "No boxed answer found"

    unique = sorted(set(letters))
    if len(unique) > 1:
        return None, f"Conflicting boxed answers: {unique}"

    return unique[0], None


def extract_boxed_letter_from_payload(
    raw_text: str,
    json_payload: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if json_payload:
        boxed_letter = json_payload.get("boxed_letter")
        if isinstance(boxed_letter, str):
            candidate = boxed_letter.strip().upper()
            if len(candidate) == 1 and candidate.isalpha():
                return candidate, None

        final_answer = json_payload.get("final_answer")
        if isinstance(final_answer, str):
            letter, error = extract_unique_boxed_letter(final_answer)
            if letter is not None:
                return letter, None
            if error and "Conflicting" in error:
                return None, error
            # Fallback: bare letter from final_answer (e.g. "C", "(A)", "A.")
            stripped = final_answer.strip().strip("'\"").strip()
            bare = re.match(r'^[({]?\s*([A-Ja-j])\s*[)}.\\]?$', stripped)
            if bare:
                return bare.group(1).upper(), None

    # Try boxed from raw_text first
    letter, error = extract_unique_boxed_letter(raw_text)
    if letter is not None:
        return letter, None

    # Last resort: extract letter from final_answer with common patterns
    if json_payload:
        fa = json_payload.get("final_answer")
        if isinstance(fa, str):
            # "answer is C", "Answer: C", "answer is (C)"
            m = re.search(r'(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Ja-j])\)?', fa, re.IGNORECASE)
            if m:
                return m.group(1).upper(), None
            # Trailing standalone letter: "... C", "... C."
            m = re.search(r'\b([A-Ja-j])\s*[.)]*\s*$', fa.strip())
            if m:
                return m.group(1).upper(), None
            # Single A-J letter anywhere (only if exactly one distinct)
            found = re.findall(r'[A-J]', fa.upper())
            if found and len(set(found)) == 1:
                return found[0], None

    return None, error


def clamp_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def extract_confidence(raw_text: str, json_payload: dict[str, Any] | None) -> float | None:
    if json_payload and "confidence" in json_payload:
        try:
            return clamp_confidence(float(json_payload["confidence"]))
        except Exception:
            pass

    if not raw_text:
        return None

    match = CONFIDENCE_PATTERN.search(raw_text)
    if match:
        try:
            value = float(match.group(1))
            if value > 1:
                value = value / 100.0
            return clamp_confidence(value)
        except Exception:
            return None

    percent_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", raw_text)
    if percent_match:
        try:
            return clamp_confidence(float(percent_match.group(1)) / 100.0)
        except Exception:
            return None

    return None
