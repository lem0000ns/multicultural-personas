#!/usr/bin/env python3
import argparse
import os
import re
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calculate SAQ evaluation summary and (if multiple iterations) iteration improvement')
parser.add_argument('--results_file', type=str, default='saq_baseline_results.csv', help='The file containing the evaluation results')
parser.add_argument('--model', type=str, default='llama-3-8b-instruct', help='name of subfolder in saq_results')
parser.add_argument('--llm-judge', action='store_true', help='Run LLM-as-a-judge: pick best answer across iterations, then score vs ground truth (SEM-B/SEM-W)')
parser.add_argument('--judge-mode', type=str, default='best', choices=('best', 'majority'), help='With --llm-judge: best = LLM picks best answer; majority = use most common answer across runs (no LLM)')
parser.add_argument('--response_dir', type=str, default=None, help='Response dir(s) for --llm-judge: one dir with iteration column, or comma-separated baseline run dirs (e.g. ../qwen3-32b_baseline_r1,../qwen3-32b_baseline_r2,...)')
parser.add_argument('--annotation_dir', type=str, default=None, help='Annotation directory for --llm-judge (e.g. ../data/annotations)')
args = parser.parse_args()

# Resolve path
results_file = args.results_file
model_filename = args.model
eval_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(results_file):
    if results_file.startswith("saq_results" + os.sep) or results_file.startswith("saq_results/"):
        results_file = os.path.join(eval_dir, results_file)
    else:
        results_file = os.path.join(eval_dir, 'saq_results', model_filename, results_file)

if not os.path.exists(results_file):
    print(f"File not found: {results_file}")
    exit(1)

df = pd.read_csv(results_file)
if "iteration" in df.columns:
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")

# Deduplicate if multiple rows per (model, country, language, eval_method, iteration)
key = [c for c in ['model', 'country', 'language', 'eval_method', 'score', 'iteration'] if c in df.columns]
if key:
    df = df.drop_duplicates(subset=key, keep='last')
print(df)

# ---------------------------------------------------------------------------
# 1. Always print SEM-B / SEM-W summary (overall + per country)
# ---------------------------------------------------------------------------
sem_b = df[df['eval_method'] == 'SEM-B']['score']
sem_w = df[df['eval_method'] == 'SEM-W']['score']

print(f"\nResults file: {results_file}\n")
print("=" * 60)
print("SAQ EVALUATION SUMMARY")
print("=" * 60)
print("\nOVERALL AVERAGES (all countries)")
print("-" * 40)
if len(sem_b):
    print(f"  SEM-B (avg): {sem_b.mean():.4f}")
if len(sem_w):
    print(f"  SEM-W (avg): {sem_w.mean():.4f}")
print()

if 'country' in df.columns:
    print("PER-COUNTRY AVERAGES (SEM-B, SEM-W)")
    print("-" * 40)
    by_country = df.pivot_table(index='country', columns='eval_method', values='score', aggfunc='mean')
    for m in ['SEM-B', 'SEM-W']:
        if m not in by_country.columns:
            continue
        by_country[m] = by_country[m].round(4)
    if 'SEM-B' in by_country.columns and 'SEM-W' in by_country.columns:
        by_country = by_country[['SEM-B', 'SEM-W']]
    for country in sorted(by_country.index):
        row = by_country.loc[country]
        b = row.get('SEM-B', 0)
        w = row.get('SEM-W', 0)
        if pd.isna(b):
            b = 0
        if pd.isna(w):
            w = 0
        print(f"  {country:20s}  SEM-B: {b:.4f}   SEM-W: {w:.4f}")
    print()

# ---------------------------------------------------------------------------
# 2. Iteration improvement (only if we have multiple iterations)
# ---------------------------------------------------------------------------
has_iteration = 'iteration' in df.columns
if has_iteration:
    iterations = sorted(df['iteration'].dropna().unique())
else:
    iterations = []

# Average SEM-B / SEM-W per iteration
if len(iterations) >= 1:
    print("=" * 60)
    print("AVERAGE SEM-B / SEM-W PER ITERATION")
    print("=" * 60)
    for it in iterations:
        sub = df[(df['iteration'] == it) & (df['eval_method'].isin(['SEM-B', 'SEM-W']))]
        sem_b_it = sub[sub['eval_method'] == 'SEM-B']['score']
        sem_w_it = sub[sub['eval_method'] == 'SEM-W']['score']
        b_avg = sem_b_it.mean() if len(sem_b_it) else 0
        w_avg = sem_w_it.mean() if len(sem_w_it) else 0
        print(f"  Iteration {it}:  SEM-B (avg): {b_avg:.4f}   SEM-W (avg): {w_avg:.4f}")
    print()

if len(iterations) >= 2:
    iter_first, iter_last = iterations[0], iterations[-1]
    df_filtered = df[df['iteration'].isin([iter_first, iter_last])]

    pivot = df_filtered.pivot_table(
        index=['country', 'language', 'eval_method'],
        columns='iteration',
        values='score',
        aggfunc='first'
    ).reset_index()

    if iter_first in pivot.columns and iter_last in pivot.columns:
        pivot['improvement'] = pivot[iter_last] - pivot[iter_first]
        pivot['improvement_pct'] = (pivot[iter_last] - pivot[iter_first]) / pivot[iter_first].replace(0, 1e-9) * 100
        pivot = pivot.rename(columns={iter_first: 'iter_first_score', iter_last: 'iter_last_score'})
        pivot = pivot.sort_values('improvement', ascending=False)

        print("=" * 60)
        print(f"SAQ ITERATION IMPROVEMENT (iter {iter_first} → iter {iter_last})")
        print("=" * 60)
        print(f"\nOverall: avg improvement {pivot['improvement'].mean():.4f}, "
              f"median {pivot['improvement'].median():.4f}, "
              f"std {pivot['improvement'].std():.4f}\n")

        print(f"{'Country':<20} {'Language':<12} {'Method':<8} "
              f"{'First':<10} {'Last':<10} {'Improve':<10} {'%':<8}")
        print("-" * 100)
        for _, row in pivot.iterrows():
            print(f"{row['country']:<20} {row['language']:<12} {row['eval_method']:<8} "
                  f"{row['iter_first_score']:<10.4f} {row['iter_last_score']:<10.4f} "
                  f"{row['improvement']:<10.4f} {row['improvement_pct']:<8.2f}")

        print("\nBy evaluation method:")
        for method in ['SEM-B', 'SEM-W']:
            m = pivot[pivot['eval_method'] == method]
            if len(m):
                print(f"  {method}: avg improvement {m['improvement'].mean():.4f}, "
                      f"positive {int((m['improvement'] > 0).sum())}/{len(m)}")

        print("\nBy country (avg improvement):")
        country_summary = pivot.groupby('country').agg({
            'improvement': ['mean', 'std', 'count'],
            'improvement_pct': 'mean'
        }).round(4)
        country_summary.columns = ['avg_improvement', 'std_improvement', 'count', 'avg_pct']
        country_summary = country_summary.sort_values('avg_improvement', ascending=False)
        for country, row in country_summary.iterrows():
            print(f"  {country:<20} {row['avg_improvement']:.4f}  (n={int(row['count'])})")

        output_file = os.path.join(os.path.dirname(__file__), 'saq_iteration_improvement.csv')
        pivot.to_csv(output_file, index=False)
        print(f"\nImprovement table saved to: {output_file}")
else:
    if has_iteration and len(iterations) == 1:
        print("=" * 60)
        print(f"Single iteration (iter {iterations[0]}) — no improvement analysis.")
    elif not has_iteration:
        print("=" * 60)
        print("No iteration column — no improvement analysis.")
    print()

# LLM-as-a-judge: same model as data; pick best answer across iterations, score vs ground truth, present judge choice
if getattr(args, "llm_judge", False) and getattr(args, "response_dir", None) and getattr(args, "annotation_dir", None):
    try:
        from evaluation_utils import get_model_response_file, get_annotations
        from exact_match import soft_exact_match
        import sys as _sys
        _blend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _blend not in _sys.path:
            _sys.path.insert(0, _blend)
        from utils import get_model_response
    except ImportError as e:
        print("LLM judge SAQ needs evaluation_utils, exact_match, utils:", e)
    else:
        import glob
        judge_mode = getattr(args, "judge_mode", "best")
        def _judge_call(sys_p, user_p, max_t=128):
            prompt = (sys_p + "\n\n" + user_p) if sys_p else user_p
            out = get_model_response(args.model, prompt, model=None, tokenizer=None, temperature=0, top_p=1, gpt_azure=False, system_message=None, max_tokens=max_t)
            return (out or "").strip()
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        adir = os.path.normpath(os.path.join(parent, args.annotation_dir)) if not os.path.isabs(args.annotation_dir) else args.annotation_dir
        # Support single dir (iteration column) or comma-separated baseline run dirs (r1, r2, ...)
        rdirs_raw = [d.strip() for d in args.response_dir.split(",") if d.strip()]
        rdirs = []
        for d in rdirs_raw:
            rdirs.append(os.path.normpath(os.path.join(parent, d)) if not os.path.isabs(d) else d)
        if not rdirs:
            print("No response_dir(s) given.")
        else:
            files = glob.glob(os.path.join(rdirs[0], "*_English_result.csv"))
            def _country_from_basename(basename, model_key):
                stem = basename.replace("_English_result.csv", "")
                if model_key and stem.startswith(model_key + "-"):
                    return stem[len(model_key) + 1:]  # e.g. qwen3-32b-Algeria -> Algeria
                if "-" in stem:
                    return stem.split("-", 1)[1].replace("-", "_")
                return stem
            countries = sorted(set(_country_from_basename(os.path.basename(f), args.model) for f in files if "_English_result.csv" in f))
            all_b, all_w, samples = [], [], []
            for country in countries:
                ann = get_annotations(data_dir=adir, country=country, template="{country}_data.json".replace("{country}", country))
                if not ann:
                    continue
                # Load one dataframe per run (or one df with iteration column if single dir)
                run_dfs = []
                for rdir in rdirs:
                    try:
                        df_run = get_model_response_file(data_dir=rdir, model=args.model, country=country, language="English")
                        run_dfs.append(df_run)
                    except Exception:
                        run_dfs.append(None)
                run_dfs = [x for x in run_dfs if x is not None and len(x) > 0]
                if len(run_dfs) < 2:
                    continue
                # If single dir and has iteration column: use iterations
                use_iterations = len(rdirs) == 1 and "iteration" in run_dfs[0].columns and run_dfs[0]["iteration"].notna().any()
                if use_iterations:
                    res_df = run_dfs[0]
                    judge_rows = []
                    for qid, grp in tqdm(list(res_df.groupby("ID")), desc="Judge %s" % country):
                        if qid not in ann:
                            continue
                        rows = grp.sort_values("iteration")
                        it_resps = [(int(row["iteration"]), str(row.get("response","") or "").strip()) for _, row in rows.iterrows() if str(row.get("response","") or "").strip()]
                        if len(it_resps) < 2:
                            continue
                        q = (rows["Translation"].iloc[0] or "").strip() if "Translation" in rows.columns else ""
                        user = "Question: " + q + "\n\nModel stated:\n" + "\n".join("  Answer " + str(i) + ": " + r for i, r in it_resps)
                        if judge_mode == "majority":
                            user += "\n\nWhich answer is the most common or majority choice? Treat answers that mean the same (e.g. minor wording or punctuation differences) as the same. Output only that answer, no explanation."
                            sys_p = "You are a judge. Identify the majority answer among the stated answers; treat semantically equivalent answers as the same. Output only that single short answer."
                        else:
                            user += "\n\nBest single answer (no explanation)?"
                            sys_p = "You are a judge. Choose the single best short answer. Output only that answer."
                        try:
                            out = _judge_call(sys_p, user)
                        except Exception:
                            out = ""
                        judge_rows.append({"ID": qid, "response": out, "prompt": ""})
                        if len(samples) < 6:
                            samples.append((country, qid, out, q[:50]))
                else:
                    # Multiple run dirs: one response per run per question (no iteration column)
                    id_col, r_col, q_col = "ID", "response", "Translation"
                    all_qids = set()
                    for d in run_dfs:
                        all_qids.update(d[id_col].astype(str).tolist())
                    judge_rows = []
                    for qid in tqdm(all_qids, desc="Judge %s" % country):
                        if qid not in ann:
                            continue
                        run_resps = []
                        question_text = ""
                        for run_num, d in enumerate(run_dfs, start=1):
                            row = d[d[id_col].astype(str) == str(qid)]
                            if row is None or len(row) == 0:
                                continue
                            row = row.iloc[0]
                            r = str(row.get(r_col, "") or "").strip()
                            if q_col in row:
                                question_text = (row[q_col] or "").strip()
                            run_resps.append((run_num, r))
                        if len(run_resps) < 2:
                            continue
                        user = "Question: " + question_text + "\n\nModel stated:\n" + "\n".join("  Run " + str(i) + ": " + r for i, r in run_resps)
                        if judge_mode == "majority":
                            user += "\n\nWhich answer is the most common or majority choice? Treat answers that mean the same (e.g. minor wording or punctuation differences) as the same. Output only that answer, no explanation."
                            sys_p = "You are a judge. Identify the majority answer among the stated answers; treat semantically equivalent answers as the same. Output only that single short answer."
                        else:
                            user += "\n\nBest single short answer (no explanation)?"
                            sys_p = "You are a judge. Choose the single best short answer. Output only that answer."
                        try:
                            out = _judge_call(sys_p, user)
                        except Exception:
                            out = ""
                        judge_rows.append({"ID": qid, "response": out, "prompt": ""})
                        if len(samples) < 6:
                            samples.append((country, qid, out, question_text[:50]))
                if not judge_rows:
                    continue
                judge_df = pd.DataFrame(judge_rows)
                b, w, _ = soft_exact_match(country=country, language="English", annotation_dict=ann, response_df=judge_df, id_col="ID", r_col="response", annotations_key="annotations")
                all_b.append(b)
                all_w.append(w)
            if all_b:
                print("=" * 60)
                print("Majority-vote (BLEnD SAQ)" if judge_mode == "majority" else "LLM-as-Judge (BLEnD SAQ)")
                print("=" * 60)
                print("SEM-B (judge vs ground truth): {:.4f}".format(sum(all_b)/len(all_b)))
                print("SEM-W (judge vs ground truth): {:.4f}".format(sum(all_w)/len(all_w)))
                if samples:
                    print("\nSample (country, qid, {} answer, question):".format("majority" if judge_mode == "majority" else "judge"))
                    for country, qid, ans, q in samples[:6]:
                        print("  {} | {} | \"{}\" | {}".format(country, qid, (ans or "")[:50], q))
                print()

print("=" * 60 + "\n")
