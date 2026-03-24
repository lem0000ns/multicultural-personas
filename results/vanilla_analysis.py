"""
Analyze vanilla evaluation results: per-run accuracy from metadata, majority-vote accuracy,
and LLM-as-a-judge accuracy (judge picks best answer across 5 runs, then we score vs ground truth).
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

from tqdm import tqdm

# Allow running from results/ or project root
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from culturalbench.tools.db.db_utils import get_accuracies, load_results


def _call_llm_judge(system_prompt: str, user_prompt: str, model_name: str, max_tokens: int = 256) -> str:
    """Call the same model as being evaluated (via culturalbench tools.llm_utils). Returns stripped response text."""
    try:
        from culturalbench.tools import llm_utils
    except ImportError:
        raise ImportError("tools.llm_utils required (run from culturalbench or set PYTHONPATH)")
    llm_utils.MODEL_NAME = model_name
    if model_name not in llm_utils.generate_text_funcs:
        raise ValueError("Unknown model for judge: %s" % model_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    llm = llm_utils.get_llm()
    gen_fn = llm_utils.generate_text_funcs[model_name]
    _, content = gen_fn(llm, messages, max_tokens=max_tokens, enable_thinking_bool=False)
    return (content or "").strip()


def _vanilla_dir(model_name: str, results_root: str = "results") -> str:
    return os.path.join(results_root, "vanilla", model_name)


def print_per_run_accuracy(model_name: str, results_root: str = "results"):
    base = _vanilla_dir(model_name, results_root)
    if not os.path.isdir(base):
        print(f"Directory not found: {base}")
        return
    print(f"Per-run accuracy (model: {model_name})")
    print("-" * 50)
    for difficulty in ("easy", "hard"):
        for run in range(1, 6):
            db_path = os.path.join(base, f"{difficulty}_{run}.db")
            accs = get_accuracies(db_path)
            if not accs:
                print(f"  Run {run} {difficulty}: (no metadata)")
                continue
            row = accs[0]
            acc = row["accuracy"]
            correct = row["correct_count"]
            total = row["total_count"]
            print(f"  Run {run} {difficulty}: {acc:.4f}  ({correct}/{total})")
    print()


def majority_vote_accuracy(model_name: str, results_root: str = "results"):
    base = _vanilla_dir(model_name, results_root)
    if not os.path.isdir(base):
        print(f"Directory not found: {base}")
        return
    for difficulty in ("easy", "hard"):
        all_runs = []
        for run in range(1, 6):
            db_path = os.path.join(base, f"{difficulty}_{run}.db")
            rows = load_results(db_path, iteration=1)
            if not rows:
                continue
            all_runs.append(rows)
        if len(all_runs) != 5:
            print(f"Majority vote {difficulty}: need 5 run DBs, found {len(all_runs)}")
            continue
        n = len(all_runs[0])
        if difficulty == "easy":
            correct = 0
            for i in range(n):
                votes = [all_runs[r][i]["model_answer"] for r in range(5)]
                vote_str = [str(v).strip().upper() for v in votes]
                majority = Counter(vote_str).most_common(1)[0][0]
                if len(majority) > 1:
                    majority = majority[0]
                gold = str(all_runs[0][i]["correct_answer"]).strip().upper()
                if len(gold) > 1:
                    gold = gold[0]
                if majority == gold:
                    correct += 1
            acc = correct / n if n else 0
            print(f"Majority vote Easy: {acc:.4f}  ({correct}/{n})")
        else:
            n_sets = n // 4
            correct_sets = 0
            for s in range(n_sets):
                set_correct = True
                for j in range(4):
                    idx = s * 4 + j
                    votes = [all_runs[r][idx]["model_answer"] for r in range(5)]
                    vote_str = [str(v).strip().lower() for v in votes]
                    majority = Counter(vote_str).most_common(1)[0][0]
                    if "true" in majority:
                        pred = "true"
                    else:
                        pred = "false"
                    gold_raw = all_runs[0][idx]["correct_answer"]
                    gold = "true" if str(gold_raw).strip().lower() in ("1", "true", "yes") else "false"
                    if pred != gold:
                        set_correct = False
                        break
                if set_correct:
                    correct_sets += 1
            acc = correct_sets / n_sets if n_sets else 0
            print(f"Majority vote Hard: {acc:.4f}  ({correct_sets}/{n_sets})")
    print()


def llm_judge_accuracy(model_name: str, results_root: str = "results", verbose: bool = True):
    """Use LLM-as-a-judge to pick best answer across 5 runs; score judge choice vs ground truth. Presents judge choice and (for MCQ) option text."""
    base = _vanilla_dir(model_name, results_root)
    if not os.path.isdir(base):
        print(f"Directory not found: {base}")
        return
    all_runs = []
    for run in range(1, 6):
        run_rows = {}
        for difficulty in ("easy", "hard"):
            db_path = os.path.join(base, f"{difficulty}_{run}.db")
            rows = load_results(db_path, iteration=1) if os.path.isfile(db_path) else []
            run_rows[difficulty] = rows
        if run_rows.get("easy") and run_rows.get("hard"):
            all_runs.append(run_rows)
    if len(all_runs) != 5:
        print(f"LLM judge: need 5 run DBs, found {len(all_runs)}")
        return

    easy_runs = [r["easy"] for r in all_runs]
    n_easy = len(easy_runs[0])
    correct_easy = 0
    judge_choices_easy = []
    for i in tqdm(range(n_easy), desc="Easy MCQ"):
        row0 = easy_runs[0][i]
        question = row0.get("question", "")
        options = row0.get("options") or {}
        opt_text = {k: options.get(k, "") for k in "ABCD"}
        votes = []
        for r in range(5):
            ans = str(easy_runs[r][i].get("model_answer", "")).strip().upper()
            if len(ans) > 1:
                ans = ans[0]
            votes.append((r + 1, ans, opt_text.get(ans, "(no text)")))
        user = "Question:\n" + question + "\n\nOptions:\n"
        for k in "ABCD":
            user += "  " + k + ". " + str(opt_text.get(k, "")) + "\n"
        user += "\nAcross 5 runs the model chose:\n"
        for run_num, letter, text in votes:
            user += "  Run " + str(run_num) + ": " + letter + " - " + str(text) + "\n"
        user += "\nWhich single letter (A, B, C, or D) is the best answer? Reply with only that letter."
        system = "You are a judge. Given a multiple-choice question and how 5 runs answered, choose the single best answer. Output only one letter: A, B, C, or D."
        try:
            out = _call_llm_judge(system, user, model_name)
        except Exception as e:
            if verbose:
                print("Judge error (easy q " + str(i) + "):", e)
            out = ""
        pred = re.search(r"\b([ABCD])\b", out.upper()) or re.search(r"^([ABCD])", out.upper())
        pred_letter = pred.group(1) if pred else None
        gold = str(row0.get("correct_answer", "")).strip().upper()
        if len(gold) > 1:
            gold = gold[0]
        if pred_letter and pred_letter == gold:
            correct_easy += 1
        chosen_text = opt_text.get(pred_letter, "") if pred_letter else ""
        judge_choices_easy.append((pred_letter, chosen_text, gold))
    acc_easy = correct_easy / n_easy if n_easy else 0
    print("=" * 60)
    print("LLM-as-Judge (CulturalBench Easy MCQ, 5 runs)")
    print("=" * 60)
    print("Judge accuracy: " + str(round(acc_easy, 4)) + "  (" + str(correct_easy) + "/" + str(n_easy) + ")")
    if verbose and judge_choices_easy:
        print("\nFirst 3 judge choices (Judge choice, option text, gold):")
        for idx, (letter, text, gold) in enumerate(judge_choices_easy[:3]):
            t = (text or "")[:60]
            print("  Q" + str(idx+1) + ": judge=" + str(letter) + " | option=\"" + t + "...\" | gold=" + str(gold))
    print()

    hard_runs = [r["hard"] for r in all_runs]
    n_hard = len(hard_runs[0])
    n_sets = n_hard // 4
    correct_sets_hard = 0
    judge_choices_hard = []
    for s in tqdm(range(n_sets), desc="Hard T/F"):
        start = s * 4
        row0 = hard_runs[0][start]
        question = row0.get("question", "")
        option_texts = [hard_runs[0][start + j].get("prompt_option", "") for j in range(4)]
        run_answers = []
        for r in range(5):
            tfs = []
            for j in range(4):
                ans = str(hard_runs[r][start + j].get("model_answer", "")).strip().lower()
                tfs.append("true" if ans in ("1", "true", "yes") else "false")
            run_answers.append((r + 1, tfs))
        user = "Question:\n" + question + "\n\n"
        for j in range(4):
            user += "Option " + str(j+1) + ": " + str(option_texts[j]) + "\n"
        user += "\nAcross 5 runs, per option the model said True or False:\n"
        for run_num, tfs in run_answers:
            user += "  Run " + str(run_num) + ": " + ", ".join("Option" + str(i+1) + "=" + tfs[i] for i in range(4)) + "\n"
        user += "\nWhat is the correct True/False for each of the 4 options? Reply with exactly 4 comma-separated values: true or false."
        system = "You are a judge. Given a question and 4 options and how 5 runs answered (true/false per option), output the correct true/false for each option. Output only 4 comma-separated values."
        try:
            out = _call_llm_judge(system, user, model_name, max_tokens=128)
        except Exception as e:
            if verbose:
                print("Judge error (hard set " + str(s) + "):", e)
            out = ""
        pred_tfs = []
        for part in re.split(r"[,;\s]+", out.lower()):
            if part in ("true", "false"):
                pred_tfs.append(part)
        pred_tfs = (pred_tfs + ["false"] * 4)[:4]
        gold_tfs = []
        for j in range(4):
            g = str(hard_runs[0][start + j].get("correct_answer", "")).strip().lower()
            gold_tfs.append("true" if g in ("1", "true", "yes") else "false")
        if all(pred_tfs[j] == gold_tfs[j] for j in range(4)):
            correct_sets_hard += 1
        judge_choices_hard.append((pred_tfs, gold_tfs))
    acc_hard = correct_sets_hard / n_sets if n_sets else 0
    print("=" * 60)
    print("LLM-as-Judge (CulturalBench Hard True/False, 5 runs)")
    print("=" * 60)
    print("Judge accuracy (full set correct): " + str(round(acc_hard, 4)) + "  (" + str(correct_sets_hard) + "/" + str(n_sets) + ")")
    if verbose and judge_choices_hard:
        print("\nFirst 2 judge choices (pred, gold):")
        for idx, (pred, gold) in enumerate(judge_choices_hard[:2]):
            print("  Set " + str(idx+1) + ": judge=" + str(pred) + " | gold=" + str(gold))
    print()


def main():
    parser = argparse.ArgumentParser(description="Vanilla evaluation analysis")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Model name, e.g. mistral-3-14b-instruct-2512 or Qwen/Qwen3-14B",
    )
    parser.add_argument("--majority", action="store_true", help="Run majority-vote accuracy across 5 runs (easy MCQ + hard True/False)")
    parser.add_argument("--llm-judge", action="store_true", help="Run LLM-as-a-judge to pick best answer across 5 runs and report accuracy")
    parser.add_argument("--results-root", type=str, default=None, help="Results root (default: results/ relative to this script)")
    args = parser.parse_args()
    results_root = args.results_root or "../results"
    if not os.path.isabs(results_root):
        results_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), results_root))
    model_name = args.model
    if args.majority:
        print("Majority vote accuracy (across 5 runs):")
        majority_vote_accuracy(model_name, results_root)
    if args.llm_judge:
        print("LLM-as-Judge accuracy (judge picks best across 5 runs, then score vs ground truth):")
        llm_judge_accuracy(model_name, results_root)
    if not args.majority and not args.llm_judge:
        parser.print_help()
        print("\nUse --majority and/or --llm-judge to run analysis.")


if __name__ == "__main__":
    main()
