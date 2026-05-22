"""Helpers for long-term memory: IDs, embedding text, correctness, prompt formatting."""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_country(country: str) -> str:
    return _normalize_whitespace(country).lower()


def _canonical_options_easy(options: Dict[str, str]) -> str:
    parts = []
    for key in sorted(options.keys()):
        parts.append(f"{key}:{_normalize_whitespace(str(options[key]))}")
    return "|".join(parts)


def _canonical_options_hard(prompt_options: List[str]) -> str:
    return "|".join(_normalize_whitespace(str(o)) for o in prompt_options)


def compute_question_id(
    question: str,
    country: str,
    options: Optional[Dict[str, str]] = None,
    prompt_options: Optional[List[str]] = None,
) -> str:
    """Stable id for dedup and exclude-current filtering."""
    q = _normalize_whitespace(question)
    c = _normalize_country(country)
    if options is not None:
        opt_part = _canonical_options_easy(options)
    elif prompt_options is not None:
        opt_part = _canonical_options_hard(prompt_options)
    else:
        opt_part = ""
    payload = f"{q}\n{c}\n{opt_part}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_embedding_text_easy(question: str, country: str, options: Dict[str, str]) -> str:
    lines = [
        f"Country: {_normalize_whitespace(country)}",
        f"Question: {_normalize_whitespace(question)}",
    ]
    for key in ["A", "B", "C", "D"]:
        if key in options:
            lines.append(f"{key}. {_normalize_whitespace(str(options[key]))}")
    return "\n".join(lines)


def build_embedding_text_hard(question: str, country: str, prompt_options: List[str]) -> str:
    lines = [
        f"Country: {_normalize_whitespace(country)}",
        f"Question: {_normalize_whitespace(question)}",
    ]
    for idx, opt in enumerate(prompt_options, start=1):
        lines.append(f"{idx}. {_normalize_whitespace(str(opt))}")
    return "\n".join(lines)


def easy_correctness_score(model_answer: str, correct_answer: str) -> float:
    if str(model_answer).upper().strip() == str(correct_answer).upper().strip():
        return 1.0
    return 0.0


def hard_correctness_score(rows: List[Dict[str, Any]]) -> float:
    """rows: 4 sub-rows with model_answer and correct_answer."""
    correct_count = 0
    for row in rows:
        answer = row.get("model_answer", "")
        correct_str = str(row.get("correct_answer", "")).lower().strip()
        expected = "true" if correct_str in ["1", "true"] else "false"
        if str(answer).lower().strip() == expected:
            correct_count += 1
    return correct_count / 4.0


def format_options_for_prompt_easy(options: Dict[str, str]) -> str:
    lines = []
    for key in ["A", "B", "C", "D"]:
        if key in options:
            lines.append(f"{key}. {options[key]}")
    return "\n".join(lines)


def format_options_for_prompt_hard(prompt_options: List[str]) -> str:
    return "\n".join(f"{i}. {opt}" for i, opt in enumerate(prompt_options, start=1))


def format_long_term_memory_block(memory: Dict[str, Any]) -> str:
    """Format one retrieved memory summary for the refine prompt."""
    return (memory.get("summary") or "").strip()


def format_long_term_memories(memories: List[Dict[str, Any]]) -> str:
    if not memories:
        return ""
    blocks = []
    for i, mem in enumerate(memories, start=1):
        text = format_long_term_memory_block(mem)
        if text:
            blocks.append(f"[Memory {i}]\n{text}")
    return "\n\n".join(blocks)
