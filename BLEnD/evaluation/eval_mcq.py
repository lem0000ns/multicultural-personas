#!/usr/bin/env python3
import json
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
def calculate_mc_accuracy(results_file=None):
    if results_file is None:
        results_file = input("Results file (e.g. llama-3-8b-instruct-mc_res_r1.csv): ")

    if not os.path.exists(results_file):
        base = os.path.basename(results_file)
        mc_data_dir = os.path.join(os.path.dirname(__file__), 'mc_data')
        found = None
        # If filename looks like <model>-mc_res_*.csv, try mc_data/<model>/<filename> first
        if '-mc_' in base:
            model_sub = base.split('-mc_')[0]
            candidate = os.path.join(mc_data_dir, model_sub, base)
            if os.path.isfile(candidate):
                found = candidate
        if found:
            results_file = found
        else:
            print(f"Error: File not found: {results_file}")
            print(f"Also tried: mc_data/<model>/{base} (with model from filename) and scanning mc_data subfolders")
            sys.exit(1)

    df = pd.read_csv(results_file, encoding='utf-8')

    # Check if iteration column exists
    has_iterations = 'iteration' in df.columns

    if has_iterations:
        iterations = sorted(df['iteration'].unique())
        print(f"\n{'='*60}")
        print(f"Multiple Choice Evaluation Results (Iterations: {len(iterations)})")
        print(f"{'='*60}")
        print(f"Results file: {results_file}\n")
        
        for iteration in iterations:
            iter_df = df[df['iteration'] == iteration]
            correct = sum(iter_df['answer_idx'] == iter_df['final_ans'])
            total = len(iter_df)
            accuracy = correct / total if total > 0 else 0
            
            print(f"Iteration {iteration}:")
            print(f"  Total questions: {total}")
            print(f"  Correct answers: {correct}")
            print(f"  Accuracy: {accuracy:.2%} ({accuracy:.4f})")
            print()

        # Per-country improvements across iterations (one section per country)
        # if 'country' in df.columns:
        #     print(f"{'='*60}")
        #     print("Per-country accuracy by iteration (and improvement)")
        #     print(f"{'='*60}\n")
        #     countries = sorted(df['country'].unique())
        #     improvement_by_country = []  # (country, improvement_pct) for plotting
        #     for country in countries:
        #         country_df = df[df['country'] == country]
        #         print(f"--- {country} ---")
        #         accs = []
        #         for iteration in iterations:
        #             iter_df = country_df[country_df['iteration'] == iteration]
        #             if len(iter_df) == 0:
        #                 accs.append(None)
        #                 continue
        #             correct = sum(iter_df['answer_idx'] == iter_df['final_ans'])
        #             total = len(iter_df)
        #             acc = correct / total if total > 0 else 0
        #             accs.append(acc)
        #             print(f"  Iter {iteration}: {acc:.2%} ({int(correct)}/{int(total)})")
        #         if len(accs) >= 2 and accs[0] is not None and accs[-1] is not None:
        #             improvement = accs[-1] - accs[0]
        #             improvement_by_country.append((country, improvement * 100))
        #             print(f"  Improvement (iter 1 -> iter {iterations[-1]}): {improvement:+.2%} ({accs[0]:.2%} -> {accs[-1]:.2%})")
        #         else:
        #             improvement_by_country.append((country, None))
        #         print()

        #     # Bar graph: countries (x) vs improvement % (y), sorted by increasing improvement
        #     if improvement_by_country:
        #         valid = [(c, imp) for c, imp in improvement_by_country if imp is not None]
        #         if valid:
        #             valid.sort(key=lambda x: x[1])
        #             plot_countries = [x[0] for x in valid]
        #             plot_improvements = [x[1] for x in valid]
        #             fig, ax = plt.subplots(figsize=(max(8, len(plot_countries) * 0.5), 5))
        #             ax.bar(plot_countries, plot_improvements, color='steelblue', edgecolor='black', linewidth=0.5)
        #             ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        #             ax.set_xlabel('Country')
        #             ax.set_ylabel('Improvement %')
        #             ax.set_title(f'Per-country improvement (iter 1 → iter {iterations[-1]})')
        #             plt.xticks(rotation=45, ha='right')
        #             plt.tight_layout()
        #             out_basename = os.path.splitext(os.path.basename(results_file))[0]
        #             out_dir = os.path.dirname(results_file)
        #             out_path = os.path.join(out_dir, f'{out_basename}_improvement.png')
        #             plt.savefig(out_path, dpi=150, bbox_inches='tight')
        #             plt.close()
        #             print(f"Saved improvement bar chart to: {out_path}\n")

        #     print(f"{'='*60}\n")
        
        print(f"{'='*60}\n")
    else:
        # Calculate overall accuracy (no iterations)
        correct = sum(df['answer_idx'] == df['final_ans'])
        total = len(df)
        accuracy = correct / total if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"Multiple Choice Evaluation Results")
        print(f"{'='*60}")
        print(f"Results file: {results_file}")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Overall accuracy: {accuracy:.2%} ({accuracy:.4f})")
        print(f"{'='*60}\n")

        # Calculate per-country accuracy if country column exists
        if 'country' in df.columns:
            print("Per-country accuracy:")
            print(f"{'-'*60}")
            country_stats = df.groupby('country').apply(
                lambda x: pd.Series({
                    'total': len(x),
                    'correct': sum(x['answer_idx'] == x['final_ans']),
                    'accuracy': sum(x['answer_idx'] == x['final_ans']) / len(x) if len(x) > 0 else 0
                }), include_groups=False
            ).sort_values('accuracy', ascending=False)
            
            for country, row in country_stats.iterrows():
                print(f"{country:20s}: {row['accuracy']:.2%} ({int(row['correct'])}/{int(row['total'])})")
            print(f"{'-'*60}\n")

def majority_vote_accuracy():
    model_name = input("Model name of folder in mc_data (e.g. llama3-8b): ")
    mc_data_dir = os.path.join(os.path.dirname(__file__), 'mc_data', model_name)
    if not os.path.exists(mc_data_dir):
        print(f"Error: Directory not found: {mc_data_dir}")
        sys.exit(1)
    
    questions_model_answers = defaultdict(dict)
    question_answers = dict()
    baseline_files = list()
    for file in os.listdir(mc_data_dir):
        import re
        if re.search(r'baseline_r[1-5]\.csv$', file):
            baseline_files.append(os.path.join(mc_data_dir, file))
    
    for baseline_file in baseline_files:
        df = pd.read_csv(baseline_file, encoding='utf-8')
        for index, row in df.iterrows():
            question_id = row['MCQID']
            correct_answer = row['answer_idx']
            question_answers[question_id] = correct_answer
            model_answer = row['final_ans']
            if model_answer in questions_model_answers[question_id]:
                questions_model_answers[question_id][model_answer] += 1
            else:
                questions_model_answers[question_id][model_answer] = 1
    
    correct = 0
    total = 0
    for question_id, model_answers in questions_model_answers.items():
        max_answer = max(model_answers.items(), key=lambda x: x[1])[0]
        if max_answer == question_answers[question_id]:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Majority vote accuracy: {accuracy:.2%} ({correct}/{total})")


# Map mc_data folder name to BLEnD utils model key for get_model_response (SGLang/local)
_JUDGE_MODEL_KEY = {
    "llama3-8b": "llama-3-8b-instruct",
    "llama-3-8b-instruct": "llama-3-8b-instruct",
    "qwen3-4b": "qwen3-4b",
    "qwen3-14b": "qwen3-14b",
    "qwen3.5-35b": "qwen3.5-35b",
}


def _call_llm_judge(system_prompt: str, user_prompt: str, model_key: str, max_tokens: int = 256) -> str:
    """Call the same model as being evaluated (via BLEnD utils.get_model_response / SGLang)."""
    import sys
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from utils import get_model_response
    prompt = (system_prompt + "\n\n" + user_prompt) if system_prompt else user_prompt
    out = get_model_response(
        model_key,
        prompt,
        model=None,
        tokenizer=None,
        temperature=0,
        top_p=1,
        gpt_azure=False,
        system_message=None,
        max_tokens=max_tokens,
    )
    return (out or "").strip()


def llm_judge_mcq_accuracy(model_name: str, verbose: bool = True):
    """
    Use LLM-as-a-judge (same model as data) to pick the best answer across 5 runs (baseline_r1..r5). No ground truth given to judge.
    Score the judge's choice vs answer_idx. Present judge's choice and the option text for that choice (MCQ).
    """
    model_key = _JUDGE_MODEL_KEY.get(model_name, model_name)
    mc_data_dir = os.path.join(os.path.dirname(__file__), "mc_data", model_name)
    if not os.path.exists(mc_data_dir):
        print(f"Error: Directory not found: {mc_data_dir}")
        return
    import re as re_mod
    baseline_files = sorted(
        [f for f in os.listdir(mc_data_dir) if re_mod.search(r"baseline_r[1-5]\.csv$", f)],
        key=lambda f: int(re_mod.search(r"r(\d+)", f).group(1)),
    )
    if len(baseline_files) < 5:
        print(f"Need 5 baseline files (baseline_r1..r5), found {len(baseline_files)}")
        return
    baseline_files = [os.path.join(mc_data_dir, f) for f in baseline_files[:5]]
    # Per question: list of (run_num, final_ans), and one row for options/question
    by_q = defaultdict(list)
    question_info = {}
    for run_num, path in enumerate(baseline_files, start=1):
        df = pd.read_csv(path, encoding="utf-8")
        for _, row in df.iterrows():
            qid = row["MCQID"]
            by_q[qid].append((run_num, str(row.get("final_ans", "")).strip().upper()))
            if qid not in question_info:
                question_info[qid] = {
                    "answer_idx": str(row.get("answer_idx", "")).strip().upper(),
                    "prompt": row.get("prompt", ""),
                    "choices": row.get("choices", "{}"),
                }
    correct = 0
    total = 0
    judge_choices = []
    for qid, votes in tqdm(by_q.items(), desc="MCQ judge"):
        if len(votes) != 5:
            continue
        info = question_info[qid]
        try:
            choices_dict = json.loads(info["choices"]) if isinstance(info["choices"], str) else info["choices"]
        except Exception:
            choices_dict = {}
        opt_text = {k: choices_dict.get(k, "") for k in "ABCD"}
        question_text = (info["prompt"] or "").strip()
        if "\n\nA." in question_text:
            question_text = question_text.split("\n\nA.")[0].strip()
        user = "Question:\n" + question_text + "\n\nOptions:\n"
        for k in "ABCD":
            user += "  " + k + ". " + str(opt_text.get(k, "")) + "\n"
        user += "\nAcross 5 runs the model chose:\n"
        for run_num, letter in votes:
            if len(letter) > 1:
                letter = letter[0]
            user += "  Iteration " + str(run_num) + ": " + letter + " - " + str(opt_text.get(letter, "(no text)")) + "\n"
        user += "\nAmong these 5 answers only, which one is best? Reply with only that letter (A, B, C, or D)."
        system = "You are a judge. You must pick exactly one of the 5 run answers above as the best. Output only one letter: A, B, C, or D."
        try:
            out = _call_llm_judge(system, user, model_key)
        except Exception as e:
            if verbose:
                print("Judge error (q " + str(qid) + "):", e)
            out = ""
        pred = re_mod.search(r"\b([ABCD])\b", out.upper()) or re_mod.search(r"^([ABCD])", out.upper())
        pred_letter = pred.group(1) if pred else None
        gold = info["answer_idx"]
        if len(gold) > 1:
            gold = gold[0]
        if pred_letter and pred_letter == gold:
            correct += 1
        total += 1
        chosen_text = opt_text.get(pred_letter, "") if pred_letter else ""
        judge_choices.append((qid, pred_letter, chosen_text, gold))
    acc = correct / total if total else 0
    print("=" * 60)
    print("LLM-as-Judge (BLEnD MCQ, 5 runs)")
    print("=" * 60)
    print("Judge accuracy: {:.4f}  ({}/{})".format(acc, correct, total))
    if verbose and judge_choices:
        print("\nFirst 3: (MCQID, judge choice, option text, gold)")
        for (qid, letter, text, gold) in judge_choices[:3]:
            print("  {} | judge={} | option=\"{}\" | gold={}".format(qid, letter, (text or "")[:50], gold))
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="BLEnD MCQ evaluation")
    p.add_argument("--llm-judge", action="store_true", help="LLM-as-a-judge over 5 baseline runs (baseline_r1..r5)")
    p.add_argument("--model", type=str, default=None, help="Model name for --llm-judge (e.g. qwen3-32b); must have mc_data/<model>/baseline_r1..r5.csv")
    p.add_argument("--results_file", type=str, default=None, help="MCQ results CSV to evaluate (e.g. qwen3-32b-mc_res_baseline_r1.csv); resolved under mc_data/<model>/ if needed")
    args = p.parse_args()
    if args.llm_judge:
        if not args.model:
            p.error("--llm-judge requires --model")
        llm_judge_mcq_accuracy(args.model)
    else:
        calculate_mc_accuracy(args.results_file)