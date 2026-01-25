#!/usr/bin/env python3
import pandas as pd
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python view_mc_personas.py <results_file> [num_examples]")
    print("Example: python view_mc_personas.py llama-3-8b-instruct-mc_res.csv 10")
    sys.exit(1)

results_file = sys.argv[1]
num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# If file doesn't exist, try looking in mc_data/ directory
if not os.path.exists(results_file):
    mc_data_path = os.path.join(os.path.dirname(__file__), 'mc_data', os.path.basename(results_file))
    if os.path.exists(mc_data_path):
        results_file = mc_data_path
    else:
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

df = pd.read_csv(results_file, encoding='utf-8')

if 'persona' not in df.columns:
    print("Error: No 'persona' column found in results file")
    sys.exit(1)

print(f"\n{'='*80}")
print(f"MC Question Personas - Showing {num_examples} examples")
print(f"{'='*80}\n")

# Show personas for different iterations if available
if 'iteration' in df.columns:
    iterations = sorted(df['iteration'].unique())
    for iteration in iterations[:3]:  # Show first 3 iterations
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}")
        print(f"{'='*80}\n")
        
        iter_df = df[df['iteration'] == iteration].head(num_examples)
        
        for idx, row in iter_df.iterrows():
            print(f"Question ID: {row['MCQID']}")
            print(f"Country: {row['country']}")
            if 'prompt' in df.columns:
                # Truncate long prompts
                prompt_text = str(row['prompt'])[:200].replace('\n', ' ')
                print(f"Question: {prompt_text}...")
            print(f"\nPersona:\n{row['persona']}\n")
            print(f"{'-'*80}\n")
else:
    # No iterations, just show examples
    for idx, row in df.head(num_examples).iterrows():
        print(f"Question ID: {row['MCQID']}")
        print(f"Country: {row['country']}")
        if 'prompt' in df.columns:
            # Truncate long prompts
            prompt_text = str(row['prompt'])[:200].replace('\n', ' ')
            print(f"Question: {prompt_text}...")
        print(f"\nPersona:\n{row['persona']}\n")
        print(f"{'-'*80}\n")
