"""LLM summaries for long-term memory metadata (retrieval uses question+options embeddings)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from tools import llm_utils
from tools.llm_utils import async_generate, get_llm

MEMORY_SUMMARIZE_SYSTEM = """
You write compact memory summaries for a cultural persona refinement system.

These summaries are retrieved to help future persona refinements learn transferable strategies from past iterations.

Focus primarily on:
- what cultural distinction, reasoning pattern, or failure mode mattered
- how the refined persona improved its approach
- what perspective, expertise, framing, or sensitivity the refined persona adopted

Include only minimal question context needed to understand the refinement.

Avoid restating full question details, answer options, or generic topic summaries unless necessary.

Write 3-5 concise sentences as a single paragraph.

Output only the summary paragraph.
"""

def _source_hash(record: Dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "question": record.get("question", ""),
            "country": record.get("country", ""),
            "options_text": record.get("options_text", ""),
            "persona": record.get("persona", ""),
            "persona_reasoning": record.get("persona_reasoning", ""),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SummaryCache:
    """JSON cache for memory summaries under the Chroma persist dir."""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.memory_path = os.path.join(persist_dir, "memory_summaries.json")
        self._memory: Dict[str, Dict[str, str]] = self._load(self.memory_path)

    @staticmethod
    def _load(path: str) -> Dict[str, Dict[str, str]]:
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _save_memory(self) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self._memory, f, ensure_ascii=False, indent=2)

    def get_memory(self, memory_id: str, source_hash: str) -> Optional[str]:
        entry = self._memory.get(memory_id)
        if entry and entry.get("source_hash") == source_hash:
            return entry.get("summary")
        return None

    def set_memory(self, memory_id: str, source_hash: str, summary: str) -> None:
        self._memory[memory_id] = {"source_hash": source_hash, "summary": summary}

    def flush(self) -> None:
        self._save_memory()


def _memory_user_content(record: Dict[str, Any]) -> str:
    return (
        f"Country: {record.get('country', '')}\n\n"
        f"Question: {record.get('question', '')}\n\n"
        f"Options:\n{record.get('options_text', '')}\n\n"
        f"Revised persona:\n{record.get('persona', '')}\n\n"
        f"Persona reasoning (guidance only):\n{record.get('persona_reasoning', '') or '(none)'}"
    )


async def _summarize_chat(system: str, user: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        llm = get_llm()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        _, response = await async_generate(
            llm, messages, use_steering=False, max_tokens=512
        )
        return (response or "").strip()


async def summarize_memory_record(
    record: Dict[str, Any], sem: asyncio.Semaphore
) -> str:
    return await _summarize_chat(
        MEMORY_SUMMARIZE_SYSTEM,
        _memory_user_content(record),
        sem,
    )


async def ensure_memory_summaries(
    records: List[Dict[str, Any]],
    cache: SummaryCache,
    *,
    on_progress: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Attach ``summary`` to each record; use cache and LLM for misses."""
    sem = asyncio.Semaphore(llm_utils.MAX_CONCURRENT)
    pending: List[Dict[str, Any]] = []

    for rec in records:
        sh = _source_hash(rec)
        cached = cache.get_memory(rec["memory_id"], sh)
        if cached:
            rec["summary"] = cached
        else:
            pending.append(rec)

    if pending:
        print(
            f"Summarizing {len(pending)} new/updated memories "
            f"({len(records) - len(pending)} cached)...",
            flush=True,
        )

        async def one(rec: Dict[str, Any]) -> Dict[str, Any]:
            summary = await summarize_memory_record(rec, sem)
            sh = _source_hash(rec)
            cache.set_memory(rec["memory_id"], sh, summary)
            rec["summary"] = summary
            return rec

        tasks = [one(rec) for rec in pending]
        done = 0
        total = len(pending)
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done == total or done == 1 or done % max(1, total // 5) == 0:
                print(f"  Summarized {done}/{total} memories...", flush=True)
            if on_progress and done % 50 == 0:
                on_progress(done, total)

        cache.flush()
        print(f"Finished summarizing {total} memories.", flush=True)

    return records
