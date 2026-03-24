"""
Find CulturalBench questions that were wrong in eng mode (after 5 iterations) but correct in ling mode.
Usage: python eng_ling_compare.py [--results_root RESULTS] [--model MODEL] [--temperature T] [--iteration I] [--max_questions N]
"""
import argparse
import os
import sys

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


def _norm_ans(a):
    if a is None:
        return ""
    s = str(a).strip().upper()
    return s[0] if len(s) > 1 else s


def main():
    p = argparse.ArgumentParser(description="Find questions wrong in eng (iter 5) but correct in ling")
    p.add_argument("--results_root", type=str, default=None, help="Root dir containing eng/ and ling/ (default: results/)")
    p.add_argument("--model", type=str, default="qwen3_32b", help="Model id, e.g. qwen3_32b, llama3_8b")
    p.add_argument("--temperature", type=float, default=0.6, help="Temperature in DB filename, e.g. 0.6")
    p.add_argument("--iteration", type=int, default=5, help="Iteration to compare (default 5)")
    p.add_argument("--max_questions", type=int, default=10, help="Max number of questions to print (default 10)")
    args = p.parse_args()

    if args.results_root is None:
        args.results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.abspath(args.results_root)
    model_folder = FILE_ID_TO_FOLDER.get(args.model, args.model.replace("_", "-"))
    easy_eng = os.path.join(results_root, "eng", model_folder, f"easy_t{args.temperature}_{args.model}.db")
    easy_ling = os.path.join(results_root, "ling", model_folder, f"easy_t{args.temperature}_{args.model}.db")

    if not os.path.isfile(easy_eng):
        print(f"Eng DB not found: {easy_eng}")
        return
    if not os.path.isfile(easy_ling):
        print(f"Ling DB not found: {easy_ling}")
        return

    eng_rows = load_results(easy_eng, iteration=args.iteration)
    ling_rows = load_results(easy_ling, iteration=args.iteration)
    if not eng_rows or not ling_rows:
        print("No rows for iteration", args.iteration)
        return

    # Match by question text (order should align if same dataset)
    eng_by_question = {r["question"]: r for r in eng_rows}
    ling_by_question = {r["question"]: r for r in ling_rows}
    wrong_eng_correct_ling = []
    for q, eng_r in eng_by_question.items():
        if q not in ling_by_question:
            continue
        ling_r = ling_by_question[q]
        gold = _norm_ans(eng_r.get("correct_answer"))
        eng_ans = _norm_ans(eng_r.get("model_answer"))
        ling_ans = _norm_ans(ling_r.get("model_answer"))
        eng_ok = eng_ans == gold
        ling_ok = ling_ans == gold
        if not eng_ok and ling_ok:
            wrong_eng_correct_ling.append((q, eng_r, ling_r, gold, eng_ans, ling_ans))

    n = min(args.max_questions, len(wrong_eng_correct_ling))
    print(f"Found {len(wrong_eng_correct_ling)} questions wrong in eng (iter {args.iteration}) but correct in ling.")
    print(f"Showing {n}:\n")
    print("=" * 80)
    for i, (q, eng_r, ling_r, gold, eng_ans, ling_ans) in enumerate(wrong_eng_correct_ling[:n]):
        opts = eng_r.get("options") or {}
        if isinstance(opts, str):
            import json
            try:
                opts = json.loads(opts)
            except Exception:
                opts = {}
        opt_text = {k: opts.get(k, "") for k in "ABCD"}
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {q[:500]}{'...' if len(q) > 500 else ''}")
        for k in "ABCD":
            print(f"  {k}. {str(opt_text.get(k, ''))[:200]}")
        print(f"Correct: {gold}  |  Eng (iter {args.iteration}): {eng_ans}  |  Ling (iter {args.iteration}): {ling_ans}")
        print()
    print("=" * 80)


if __name__ == "__main__":
    main()
