#!/usr/bin/env python3
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python calculate_mc_accuracy.py <results_file>")
    print("Example: python calculate_mc_accuracy.py llama-3-8b-instruct-mc_res.csv")
    sys.exit(1)

results_file = sys.argv[1]

# If file doesn't exist, try looking in mc_data/ directory
if not os.path.exists(results_file):
    mc_data_path = os.path.join(os.path.dirname(__file__), 'mc_data', os.path.basename(results_file))
    if os.path.exists(mc_data_path):
        results_file = mc_data_path
    else:
        print(f"Error: File not found: {results_file}")
        print(f"Also tried: {mc_data_path}")
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
    if 'country' in df.columns:
        print(f"{'='*60}")
        print("Per-country accuracy by iteration (and improvement)")
        print(f"{'='*60}\n")
        countries = sorted(df['country'].unique())
        improvement_by_country = []  # (country, improvement_pct) for plotting
        for country in countries:
            country_df = df[df['country'] == country]
            print(f"--- {country} ---")
            accs = []
            for iteration in iterations:
                iter_df = country_df[country_df['iteration'] == iteration]
                if len(iter_df) == 0:
                    accs.append(None)
                    continue
                correct = sum(iter_df['answer_idx'] == iter_df['final_ans'])
                total = len(iter_df)
                acc = correct / total if total > 0 else 0
                accs.append(acc)
                print(f"  Iter {iteration}: {acc:.2%} ({int(correct)}/{int(total)})")
            if len(accs) >= 2 and accs[0] is not None and accs[-1] is not None:
                improvement = accs[-1] - accs[0]
                improvement_by_country.append((country, improvement * 100))
                print(f"  Improvement (iter 1 -> iter {iterations[-1]}): {improvement:+.2%} ({accs[0]:.2%} -> {accs[-1]:.2%})")
            else:
                improvement_by_country.append((country, None))
            print()

        # Bar graph: countries (x) vs improvement % (y), sorted by increasing improvement
        if improvement_by_country:
            valid = [(c, imp) for c, imp in improvement_by_country if imp is not None]
            if valid:
                valid.sort(key=lambda x: x[1])
                plot_countries = [x[0] for x in valid]
                plot_improvements = [x[1] for x in valid]
                fig, ax = plt.subplots(figsize=(max(8, len(plot_countries) * 0.5), 5))
                ax.bar(plot_countries, plot_improvements, color='steelblue', edgecolor='black', linewidth=0.5)
                ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                ax.set_xlabel('Country')
                ax.set_ylabel('Improvement %')
                ax.set_title(f'Per-country improvement (iter 1 → iter {iterations[-1]})')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                out_basename = os.path.splitext(os.path.basename(results_file))[0]
                out_path = os.path.join(os.path.dirname(__file__), 'mc_data', f'{out_basename}_improvement.png')
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved improvement bar chart to: {out_path}\n")

        print(f"{'='*60}\n")
    
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
