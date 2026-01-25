#!/usr/bin/env python3
import pandas as pd
import sys
import os

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
        
        # Per-country breakdown for this iteration
        if 'country' in iter_df.columns:
            print(f"  Per-country accuracy:")
            country_stats = iter_df.groupby('country').apply(
                lambda x: pd.Series({
                    'total': len(x),
                    'correct': sum(x['answer_idx'] == x['final_ans']),
                    'accuracy': sum(x['answer_idx'] == x['final_ans']) / len(x) if len(x) > 0 else 0
                }), include_groups=False
            ).sort_values('accuracy', ascending=False)
            
            for country, row in country_stats.iterrows():
                print(f"    {country:20s}: {row['accuracy']:.2%} ({int(row['correct'])}/{int(row['total'])})")
        print()
    
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
