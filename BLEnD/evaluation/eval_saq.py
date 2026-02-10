#!/usr/bin/env python3
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate SAQ evaluation summary and (if multiple iterations) iteration improvement')
parser.add_argument('--results_file', type=str, default='saq_only_reasoning_results.csv', help='The file containing the evaluation results')
args = parser.parse_args()

# Resolve path
results_file = args.results_file
if not os.path.isabs(results_file):
    results_file = os.path.join(os.path.dirname(__file__), results_file)

if not os.path.exists(results_file):
    print(f"File not found: {results_file}")
    exit(1)

df = pd.read_csv(results_file)

# Deduplicate if multiple rows per (model, country, prompt_no, eval_method, iteration)
key = [c for c in ['model', 'country', 'prompt_no', 'eval_method', 'iteration'] if c in df.columns]
if key:
    df = df.drop_duplicates(subset=key, keep='last')

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

if len(iterations) >= 2:
    iter_first, iter_last = iterations[0], iterations[-1]
    df_filtered = df[df['iteration'].isin([iter_first, iter_last])]

    pivot = df_filtered.pivot_table(
        index=['country', 'language', 'prompt_no', 'eval_method'],
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

        print(f"{'Country':<20} {'Language':<12} {'Prompt':<10} {'Method':<8} "
              f"{'First':<10} {'Last':<10} {'Improve':<10} {'%':<8}")
        print("-" * 100)
        for _, row in pivot.iterrows():
            print(f"{row['country']:<20} {row['language']:<12} {row['prompt_no']:<10} {row['eval_method']:<8} "
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

print("=" * 60 + "\n")
