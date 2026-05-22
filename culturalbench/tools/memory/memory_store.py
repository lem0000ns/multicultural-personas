"""Chroma-backed long-term memory store synced from SQLite results."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from tools.db.db_utils import load_results

from .memory_summarizer import SummaryCache, ensure_memory_summaries
from .memory_utils import (
    build_embedding_text_easy,
    build_embedding_text_hard,
    compute_question_id,
    format_long_term_memories,
    format_options_for_prompt_easy,
    format_options_for_prompt_hard,
)

_STORE_CACHE: Dict[str, "MemoryStore"] = {}

RETRIEVAL_CANDIDATE_MULTIPLIER = 10
TOP_K = 5
_PRINT_LOCK: Optional[asyncio.Lock] = None


def memory_dir_from_db(db_path: str) -> str:
    if db_path.endswith(".db"):
        return db_path[:-3] + "_memory"
    return db_path + "_memory"


def _sanitize_collection_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)[:200]


def _get_embedding_function():
    """Chroma default ONNX embedder (all-MiniLM-L6-v2) — no sentence-transformers import."""
    from chromadb.utils import embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()


class MemoryStore:
    """Vector store scoped to one SQLite results database."""

    def __init__(
        self,
        db_path: str,
        difficulty: str,
        mode: str,
        enabled: bool = True,
        *,
        debug_retrieval: bool = False,
    ):
        self.db_path = db_path
        self.difficulty = difficulty
        self.mode = mode
        self.enabled = enabled
        self.debug_retrieval = debug_retrieval
        self._client = None
        self._collection = None
        self._ef = None
        self._summary_cache: Optional[SummaryCache] = None

    @property
    def persist_dir(self) -> str:
        return memory_dir_from_db(self.db_path)

    @staticmethod
    async def _print_retrieval_debug(
        *,
        current_iteration: int,
        country: str,
        question: str,
        memories: List[Dict[str, Any]],
        question_index: Optional[int] = None,
    ) -> None:
        global _PRINT_LOCK
        if _PRINT_LOCK is None:
            _PRINT_LOCK = asyncio.Lock()

        prev_iteration = current_iteration - 1
        idx_label = f"#{question_index} " if question_index is not None else ""
        q_preview = question.replace("\n", " ")
        if len(q_preview) > 160:
            q_preview = q_preview[:160] + "..."

        lines = [
            "",
            "=" * 72,
            f"[memory retrieve] {idx_label}iter={current_iteration} "
            f"(searching iter {prev_iteration}) | {country}",
            f"  Current question: {q_preview}",
        ]
        if not memories:
            lines.append("  -> no memories retrieved")
        else:
            lines.append(f"  -> {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}:")
            for i, mem in enumerate(memories, start=1):
                sim = mem.get("semantic_similarity", 0.0)
                summary = (mem.get("summary") or "").strip().replace("\n", " ")
                if len(summary) > 400:
                    summary = summary[:400] + "..."
                lines.append(f"     [{i}] sim={sim:.4f}")
                lines.append(f"         {summary}")
        lines.append("=" * 72)

        async with _PRINT_LOCK:
            print("\n".join(lines), flush=True)

    def _get_summary_cache(self) -> SummaryCache:
        if self._summary_cache is None:
            self._summary_cache = SummaryCache(self.persist_dir)
        return self._summary_cache

    def _ensure_collection(self):
        if self._collection is not None:
            return
        import chromadb
        from chromadb.config import Settings

        os.makedirs(self.persist_dir, exist_ok=True)
        self._ef = _get_embedding_function()
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        coll_name = _sanitize_collection_name(
            f"{self.mode}_{self.difficulty}_{os.path.basename(self.db_path)}"
        )
        self._collection = self._client.get_or_create_collection(
            name=coll_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def sync_from_sqlite(self) -> int:
        """Rebuild vector index from SQLite (runs async summarization)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.sync_from_sqlite_async())
        raise RuntimeError(
            "sync_from_sqlite() called inside a running event loop; "
            "use await sync_from_sqlite_async() instead."
        )

    async def sync_from_sqlite_async(self) -> int:
        """Rebuild vector index: embed question+options; store LLM summary in metadata."""
        if not self.enabled or self.mode != "eng":
            return 0
        if not os.path.exists(self.db_path):
            return 0

        self._ensure_collection()
        records = self._records_from_sqlite()
        if not records:
            print("Memory sync: no records in SQLite.", flush=True)
            return 0

        print(f"Memory sync: {len(records)} records from SQLite.", flush=True)
        cache = self._get_summary_cache()
        records = await ensure_memory_summaries(records, cache)
        print("Memory sync: writing Chroma index...", flush=True)

        ids = [r["memory_id"] for r in records]
        documents = [r["embedding_text"] for r in records]
        metadatas = [
            {
                "question_id": rec["question_id"],
                "iteration": rec["iteration"],
                "summary": rec["summary"],
            }
            for rec in records
        ]

        existing = self._collection.get()
        if existing and existing.get("ids"):
            self._collection.delete(ids=existing["ids"])

        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        print(
            f"Memory store synced: {len(ids)} records -> {self.persist_dir}",
            flush=True,
        )
        return len(ids)

    def _records_from_sqlite(self) -> List[Dict[str, Any]]:
        rows = load_results(
            self.db_path, difficulty=self.difficulty, mode=self.mode
        )
        if not rows:
            return []

        records: List[Dict[str, Any]] = []
        if self.difficulty == "Easy":
            for row in rows:
                rec = self._easy_row_to_record(row)
                if rec:
                    records.append(rec)
        else:
            for i in range(0, len(rows), 4):
                chunk = rows[i : i + 4]
                if len(chunk) < 4:
                    continue
                rec = self._hard_chunk_to_record(chunk)
                if rec:
                    records.append(rec)
        return self._dedupe_records_by_memory_id(records)

    @staticmethod
    def _dedupe_records_by_memory_id(
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep the latest SQLite row per memory_id (handles corrupted duplicate inserts)."""
        best: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            mid = rec["memory_id"]
            row_id = rec.get("sqlite_row_id", 0)
            if mid not in best or row_id > best[mid].get("sqlite_row_id", 0):
                best[mid] = rec
        return list(best.values())

    def _easy_row_to_record(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        options = row.get("options") or {}
        if not options:
            return None
        question = row.get("question", "")
        country = row.get("country", "")
        qid = compute_question_id(question, country, options=options)
        iteration = int(row.get("iteration", 1))
        options_text = format_options_for_prompt_easy(options)
        return {
            "sqlite_row_id": int(row.get("id", 0)),
            "memory_id": f"{qid}_{iteration}",
            "question_id": qid,
            "iteration": iteration,
            "embedding_text": build_embedding_text_easy(question, country, options),
            "country": country,
            "question": question,
            "options_text": options_text,
            "persona": row.get("persona_description") or "",
            "persona_reasoning": row.get("refine_reasoning") or "",
        }

    def _hard_chunk_to_record(self, chunk: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        question = chunk[0].get("question", "")
        country = chunk[0].get("country", "")
        prompt_options = [row.get("prompt_option", "") for row in chunk]
        qid = compute_question_id(question, country, prompt_options=prompt_options)
        iteration = int(chunk[0].get("iteration", 1))
        options_text = format_options_for_prompt_hard(prompt_options)
        return {
            "sqlite_row_id": max(int(row.get("id", 0)) for row in chunk),
            "memory_id": f"{qid}_{iteration}",
            "question_id": qid,
            "iteration": iteration,
            "embedding_text": build_embedding_text_hard(question, country, prompt_options),
            "country": country,
            "question": question,
            "options_text": options_text,
            "persona": chunk[0].get("persona_description") or "",
            "persona_reasoning": chunk[0].get("refine_reasoning") or "",
        }

    async def retrieve(
        self,
        question: str,
        country: str,
        *,
        current_iteration: int,
        options: Optional[Dict[str, str]] = None,
        prompt_options: Optional[List[str]] = None,
        top_k: int = TOP_K,
        question_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k memories (semantic match on question+options; prompt uses summary metadata)."""
        if not self.enabled or self.mode != "eng":
            return []
        if current_iteration < 2:
            return []

        prev_iteration = current_iteration - 1

        self._ensure_collection()
        count = self._collection.count()
        if count == 0:
            return []

        where_prev = {"iteration": prev_iteration}
        try:
            prev_count = self._collection.count(where=where_prev)
        except Exception:
            prev_count = count
        if prev_count == 0:
            return []

        if options is not None:
            exclude_qid = compute_question_id(question, country, options=options)
            query_text = build_embedding_text_easy(question, country, options)
        else:
            exclude_qid = compute_question_id(
                question, country, prompt_options=prompt_options or []
            )
            query_text = build_embedding_text_hard(
                question, country, prompt_options or []
            )

        n_candidates = min(
            max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k),
            prev_count,
        )

        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_candidates,
            where=where_prev,
            include=["metadatas", "distances"],
        )

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        scored: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas, distances):
            if not meta:
                continue
            if int(meta.get("iteration", 0)) != prev_iteration:
                continue
            qid = meta.get("question_id", "")
            if qid == exclude_qid:
                continue
            semantic_sim = max(0.0, 1.0 - float(dist))
            summary = (meta.get("summary") or "").strip()
            scored.append(
                {
                    "question_id": qid,
                    "semantic_similarity": semantic_sim,
                    "summary": summary,
                }
            )

        best_by_qid: Dict[str, Dict[str, Any]] = {}
        for item in scored:
            qid = item["question_id"]
            if qid not in best_by_qid or item["semantic_similarity"] > best_by_qid[qid]["semantic_similarity"]:
                best_by_qid[qid] = item

        ranked = sorted(best_by_qid.values(), key=lambda x: x["semantic_similarity"], reverse=True)
        result = ranked[:top_k]

        if self.debug_retrieval:
            await self._print_retrieval_debug(
                current_iteration=current_iteration,
                country=country,
                question=question,
                memories=result,
                question_index=question_index,
            )

        return result

    def format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        return format_long_term_memories(memories)


def get_memory_store(
    db_path: str,
    difficulty: str,
    mode: str,
    enabled: bool = True,
    *,
    debug_retrieval: bool = False,
) -> MemoryStore:
    key = f"{db_path}|{difficulty}|{mode}|{enabled}"
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = MemoryStore(
            db_path, difficulty, mode, enabled=enabled, debug_retrieval=debug_retrieval
        )
    else:
        store = _STORE_CACHE[key]
        store.enabled = enabled
        store.debug_retrieval = debug_retrieval
    return _STORE_CACHE[key]
