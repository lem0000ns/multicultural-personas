"""Parse and validate LLM JSON responses for CulturalBench."""

import json
import re
from typing import Optional, Tuple

import json_repair


def parse_easy_answer(response: Optional[str]) -> Optional[Tuple[str, str]]:
    """Parse {\"answer\": \"A\", \"reasoning\": \"...\"} from model output."""
    if not response or not str(response).strip():
        return None

    text = str(response).strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        result = json_repair.loads(text)
    except Exception:
        result = None

    if isinstance(result, dict):
        answer = result.get("answer") or result.get("Answer")
        reasoning = result.get("reasoning") or result.get("Reasoning") or ""
        if answer:
            return str(answer).upper().strip(), str(reasoning).strip()

    # Fallback: extract answer letter from text
    m = re.search(r'"answer"\s*:\s*"([A-D])"', text, re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
        rm = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.IGNORECASE | re.DOTALL)
        reasoning = rm.group(1).strip() if rm else ""
        return answer, reasoning

    m = re.search(r"\b([A-D])\b", text)
    if m and len(text) < 200:
        return m.group(1).upper(), text

    return None


def parse_hard_answer(response: Optional[str]) -> Optional[Tuple[str, str]]:
    """Parse {\"correct\": \"true/false\", \"reasoning\": \"...\"}."""
    if not response or not str(response).strip():
        return None

    text = str(response).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        result = json_repair.loads(text)
    except Exception:
        result = None

    if isinstance(result, dict):
        correct = result.get("correct") or result.get("Correct")
        reasoning = result.get("reasoning") or result.get("Reasoning") or ""
        if correct is not None:
            val = str(correct).lower().strip()
            thinks = "true" if "true" in val else "false"
            return thinks, str(reasoning).strip()

    m = re.search(r'"correct"\s*:\s*"(true|false)"', text, re.IGNORECASE)
    if m:
        thinks = "true" if "true" in m.group(1).lower() else "false"
        return thinks, ""

    return None
