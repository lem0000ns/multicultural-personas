"""
Per-country CulturalBench difference: ling - eng (t=0.6, one iteration).
Negative = ling worse than eng on that country.
Usage: python per_country_eng_ling_diff.py [--results_root RESULTS] [--model MODEL] [--iteration I]
"""
import argparse
import os
import sys
from collections import defaultdict

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from culturalbench.tools.db.db_utils import load_results

FILE_ID_TO_FOLDER = {
    "llama3_8b": "llama3-8b-instruct",
    "qwen3_4b": "qwen3-4b",
    "qwen3_14b": "qwen3-14b",
    "qwen3_32b": "qwen3-32b",
}

TEMPERATURE = 0.6


def _norm_ans(a):
    if a is None:
        return ""
    s = str(a).strip().upper()
    return s[0] if len(s) > 1 else s


def _hard_correct(row):
    return "true" if ("true" in str(row.get("model_answer", "")).lower().strip()) else "false"


def _hard_expected(row):
    raw = str(row.get("correct_answer", "")).lower().strip()
    return "true" if raw in ("1", "true", "yes") else "false"


def per_country_accuracy_easy(rows):
    """Returns dict country -> (correct, total)."""
    by_country = defaultdict(lambda: [0, 0])
    for r in rows:
        c = r.get("country") or "unknown"
        gold = _norm_ans(r.get("correct_answer"))
        pred = _norm_ans(r.get("model_answer"))
        by_country[c][1] += 1
        if pred == gold:
            by_country[c][0] += 1
    return {c: (correct, total) for c, (correct, total) in by_country.items()}


def per_country_accuracy_hard(rows):
    """Hard: 4 rows per question set. Returns dict country -> (correct_sets, total_sets)."""
    by_country = defaultdict(lambda: [0, 0])
    i = 0
    while i + 4 <= len(rows):
        chunk = rows[i : i + 4]
        country = chunk[0].get("country") or "unknown"
        ok = True
        for r in chunk:
            if _hard_correct(r) != _hard_expected(r):
                ok = False
                break
        by_country[country][1] += 1
        if ok:
            by_country[country][0] += 1
        i += 4
    return {c: (correct, total) for c, (correct, total) in by_country.items()}


def main():
    p = argparse.ArgumentParser(description="Per-country ling - eng accuracy difference (ling perspective)")
    p.add_argument("--results_root", type=str, default=None, help="Root dir containing eng/ and ling/")
    p.add_argument("--model", type=str, default="qwen3_14b", help="Model id, e.g. qwen3_14b, llama3_8b")
    p.add_argument("--iteration", type=int, default=5, help="Iteration to use (default 5)")
    args = p.parse_args()

    if args.results_root is None:
        args.results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.abspath(args.results_root)
    model_folder = FILE_ID_TO_FOLDER.get(args.model, args.model.replace("_", "-"))
    t = TEMPERATURE

    paths = {
        "eng_easy": os.path.join(results_root, "eng", model_folder, f"easy_t{t}_{args.model}.db"),
        "eng_hard": os.path.join(results_root, "eng", model_folder, f"hard_t{t}_{args.model}.db"),
        "ling_easy": os.path.join(results_root, "ling", model_folder, f"easy_t{t}_{args.model}.db"),
        "ling_hard": os.path.join(results_root, "ling", model_folder, f"hard_t{t}_{args.model}.db"),
    }
    for k, path in paths.items():
        if not os.path.isfile(path):
            print(f"Missing: {path}")
            return

    eng_easy = load_results(paths["eng_easy"], iteration=args.iteration)
    eng_hard = load_results(paths["eng_hard"], iteration=args.iteration)
    ling_easy = load_results(paths["ling_easy"], iteration=args.iteration)
    ling_hard = load_results(paths["ling_hard"], iteration=args.iteration)

    eng_easy_acc = per_country_accuracy_easy(eng_easy)
    eng_hard_acc = per_country_accuracy_hard(eng_hard)
    ling_easy_acc = per_country_accuracy_easy(ling_easy)
    ling_hard_acc = per_country_accuracy_hard(ling_hard)

    countries = sorted(
        set(eng_easy_acc) | set(eng_hard_acc) | set(ling_easy_acc) | set(ling_hard_acc)
    )

    # ling - eng (so negative = ling worse)
    print(f"Model: {args.model}  |  t={t}  |  iteration={args.iteration}")
    print("Difference = ling - eng  (negative means ling worse)")
    print("=" * 90)
    print(f"{'country':<25} {'eng_easy':>8} {'ling_easy':>8} {'diff_easy':>9}  |  {'eng_hard':>8} {'ling_hard':>8} {'diff_hard':>9}")
    print("-" * 90)

    for c in countries:
        def pct(d, key):
            if c not in d or d[c][1] == 0:
                return None, 0
            correct, total = d[c]
            return 100.0 * correct / total, total

        eng_e, _ = pct(eng_easy_acc, c)
        ling_e, _ = pct(ling_easy_acc, c)
        eng_h, _ = pct(eng_hard_acc, c)
        ling_h, _ = pct(ling_hard_acc, c)

        diff_e = (ling_e - eng_e) if (ling_e is not None and eng_e is not None) else None
        diff_h = (ling_h - eng_h) if (ling_h is not None and eng_h is not None) else None

        eng_e_str = f"{eng_e:.1f}%" if eng_e is not None else "  —"
        ling_e_str = f"{ling_e:.1f}%" if ling_e is not None else "  —"
        diff_e_str = f"{diff_e:+.1f}" if diff_e is not None else "  —"
        eng_h_str = f"{eng_h:.1f}%" if eng_h is not None else "  —"
        ling_h_str = f"{ling_h:.1f}%" if ling_h is not None else "  —"
        diff_h_str = f"{diff_h:+.1f}" if diff_h is not None else "  —"

        print(f"{c:<25} {eng_e_str:>8} {ling_e_str:>8} {diff_e_str:>9}  |  {eng_h_str:>8} {ling_h_str:>8} {diff_h_str:>9}")
    print("=" * 90)


if __name__ == "__main__":
    main()
