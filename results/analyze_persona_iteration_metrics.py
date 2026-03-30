"""
Summarize persona-refinement runs: mean accuracy and mean step-wise improvement
across iterations for each (mode × difficulty) for a given model.

Example (from repo root):
  python results/analyze_persona_iteration_metrics.py --model qwen3-14b
  python results/analyze_persona_iteration_metrics.py --model Qwen/Qwen3-14B --temperature 0.6
"""

from __future__ import annotations

import argparse
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from culturalbench.tools.db.db_utils import get_accuracies, load_results

# Same slug keys as culturalbench/evaluators.py model_to_save (for default DB filenames)
MODEL_NAME_TO_SLUG = {
    "Qwen/Qwen3-32B": "qwen3_32b",
    "google/gemma-2-27b-it": "gemma2_27b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama33_70b",
    "Qwen/Qwen3-4B": "qwen3_4b",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b",
    "Qwen/Qwen3-14B": "qwen3_14b",
    "google/gemma-3-12b-it": "gemma3_12b",
    "Qwen/Qwen3-0.6B": "gemma3_12b",
    "mistral-3-14b-instruct-2512": "gemma3_12b",
    "Qwen/Qwen3.5-35B-A3B": "qwen3.5_35b",
    "zai-org/GLM-4-9B-0414": "glm4_9b",
}

# Folder name (results/{mode}/{folder}/...) -> slug when user passes folder only
FOLDER_TO_SLUG = {
    "qwen3-14b": "qwen3_14b",
    "qwen3-4b": "qwen3_4b",
    "qwen3-32b": "qwen3_32b",
    "qwen3.5-35b": "qwen3.5_35b",
    "llama3-8b-instruct": "llama3_8b",
    "gemma-3-12b-it": "gemma3_12b",
    "gemma-2-27b-it": "gemma2_27b",
    "llama-3.3-70b-instruct": "llama33_70b",
    "glm4-9b": "glm4_9b",
}

MODES = ("eng", "ling", "e2l", "l2e")


def _resolve_model_folder_and_slug(model: str) -> tuple[str, str]:
    """Return (results subfolder name, filename slug)."""
    from culturalbench.token_counter import get_model_folder

    if model in FOLDER_TO_SLUG:
        return model, FOLDER_TO_SLUG[model]
    if model in MODEL_NAME_TO_SLUG:
        return get_model_folder(model), MODEL_NAME_TO_SLUG[model]
    # HF-style id not in dict: folder from helper, slug from last path segment
    folder = get_model_folder(model)
    slug = model.split("/")[-1].replace("-", "_").lower()
    return folder, slug


def _default_db_path(
    results_root: str,
    mode: str,
    model_folder: str,
    difficulty: str,
    temperature: float,
    slug: str,
    custom: str | None,
) -> str:
    base = f"{difficulty.lower()}_t{temperature}_{slug}"
    if custom:
        base += f"_{custom}"
    return os.path.join(results_root, mode, model_folder, f"{base}.db")


def _accuracy_from_rows(results: list, difficulty: str) -> float:
    """Match culturalbench.iterate.calculate_accuracy_from_db logic."""
    if difficulty == "Hard":
        total = len(results) // 4
        if total == 0:
            return 0.0
        correct = 0
        for i in range(0, len(results), 4):
            ok = True
            for j in range(i, i + 4):
                row = results[j]
                answer = row.get("model_answer", row.get("persona_answer"))
                expected = "true" if str(row["correct_answer"]) == "1" else "false"
                if str(answer).lower().strip() != expected:
                    ok = False
                    break
            if ok:
                correct += 1
        return correct / total
    total = len(results)
    if total == 0:
        return 0.0
    correct = 0
    for row in results:
        answer = row.get("model_answer", row.get("persona_answer"))
        if str(answer).upper().strip() == str(row["correct_answer"]).upper().strip():
            correct += 1
    return correct / total


def _iteration_accuracies(db_path: str, difficulty: str) -> dict[int, float]:
    """Map iteration -> accuracy using metadata, or recompute from results."""
    if not os.path.isfile(db_path):
        return {}

    meta = get_accuracies(db_path)
    cap_diff = difficulty.capitalize()
    from_meta: dict[int, float] = {}
    for row in meta:
        if row.get("difficulty") == cap_diff:
            from_meta[int(row["iteration"])] = float(row["accuracy"])

    if from_meta:
        return dict(sorted(from_meta.items()))

    # No metadata: derive from results table
    out: dict[int, float] = {}
    conn_iterations = set()
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT iteration FROM results ORDER BY iteration")
    conn_iterations = {int(r[0]) for r in cur.fetchall()}
    conn.close()

    for it in conn_iterations:
        rows = load_results(db_path, iteration=it)
        if not rows:
            continue
        out[it] = _accuracy_from_rows(rows, cap_diff)
    return dict(sorted(out.items()))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _metrics(accs: dict[int, float]) -> tuple[float | None, float | None, float | None]:
    """
    Returns (mean_accuracy, mean_consecutive_gain, net_gain last-first).
    Gains use only existing consecutive iteration pairs.
    """
    if not accs:
        return None, None, None
    ordered = sorted(accs.items())
    vals = [v for _, v in ordered]
    mean_acc = _mean(vals)
    step_gains = []
    for i in range(1, len(ordered)):
        step_gains.append(ordered[i][1] - ordered[i - 1][1])
    mean_step = _mean(step_gains) if step_gains else None
    net = ordered[-1][1] - ordered[0][1] if len(ordered) >= 2 else None
    return mean_acc, mean_step, net


def run(
    model: str,
    results_root: str,
    temperature: float,
    custom: str | None,
    modes: tuple[str, ...],
) -> None:
    folder, slug = _resolve_model_folder_and_slug(model)
    print(f"Model folder: {folder}  |  DB slug: {slug}  |  t={temperature}")
    if custom:
        print(f"Custom suffix: {custom}")
    print()

    header = (
        f"{'mode':<6} {'diff':<5} {'iters':^7} "
        f"{'avg_acc':>10} {'avg_step':>10} {'net_iN-i1':>10}  path_ok"
    )
    print(header)
    print("-" * len(header))

    for mode in modes:
        for diff in ("Easy", "Hard"):
            db_path = _default_db_path(
                results_root, mode, folder, diff, temperature, slug, custom
            )
            accs = _iteration_accuracies(db_path, diff)
            mean_acc, mean_step, net = _metrics(accs)
            n_it = len(accs)

            def fmt(x: float | None) -> str:
                return f"{x:>10.4f}" if x is not None else f"{'n/a':>10}"

            ok = os.path.isfile(db_path)
            short = db_path if len(db_path) < 72 else "…" + db_path[-68:]
            print(
                f"{mode:<6} {diff:<5} {n_it:^7} "
                f"{fmt(mean_acc)} {fmt(mean_step)} {fmt(net)}  {ok}"
            )
            if not ok and not accs:
                print(f"         (missing: {short})")

    print()
    print(
        "avg_acc: mean of per-iteration accuracies. "
        "avg_step: mean of (acc_{k+1}-acc_k) over consecutive pairs. "
        "net_iN-i1: last iteration accuracy minus first."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Mean accuracy and improvement across refinement iterations "
        "for each mode (eng/ling/e2l/l2e) and difficulty (easy/hard)."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Results folder name (e.g. qwen3-14b) or HF id (e.g. Qwen/Qwen3-14B)",
    )
    p.add_argument(
        "--results-root",
        type=str,
        default=os.path.join(_root, "results"),
        help="Directory containing eng/, ling/, ... (default: <repo>/results)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature in DB filename (default 0.6)",
    )
    p.add_argument(
        "--custom",
        type=str,
        default=None,
        help="Optional suffix before .db (e.g. no_thinking), same as iterate --custom",
    )
    p.add_argument(
        "--modes",
        type=str,
        default=",".join(MODES),
        help=f"Comma-separated modes (default: {','.join(MODES)})",
    )
    args = p.parse_args()
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    run(args.model, args.results_root, args.temperature, args.custom, modes)


if __name__ == "__main__":
    main()
