#!/usr/bin/env python3
import pandas as pd
import os

# Read the evaluation results
results_file = os.path.join(os.path.dirname(__file__), 'model_inference_results.csv')
df = pd.read_csv(results_file)

# Filter for iterations 1 and 5
df_filtered = df[df['iteration'].isin([1, 5])]

# Create pivot to get iteration 1 and 5 scores side by side
pivot = df_filtered.pivot_table(
    index=['country', 'language', 'prompt_no', 'eval_method'],
    columns='iteration',
    values='score',
    aggfunc='first'
).reset_index()

# Calculate the improvement (iteration 5 - iteration 1)
pivot['improvement'] = pivot[5] - pivot[1]
pivot['improvement_pct'] = ((pivot[5] - pivot[1]) / pivot[1] * 100)

# Rename columns for clarity
pivot.columns.name = None
pivot = pivot.rename(columns={1: 'iter_1_score', 5: 'iter_5_score'})

# Sort by improvement (descending)
pivot = pivot.sort_values('improvement', ascending=False)

print(f"\n{'='*100}")
print(f"SAQ Evaluation: Iteration 5 vs Iteration 1 Improvement Analysis")
print(f"{'='*100}\n")

# Overall statistics
print(f"Overall Statistics:")
print(f"  Average improvement: {pivot['improvement'].mean():.4f}")
print(f"  Median improvement: {pivot['improvement'].median():.4f}")
print(f"  Max improvement: {pivot['improvement'].max():.4f}")
print(f"  Min improvement: {pivot['improvement'].min():.4f}")
print(f"  Std deviation: {pivot['improvement'].std():.4f}")
print(f"\n{'='*100}\n")

# Print detailed results
print(f"{'Country':<20} {'Language':<12} {'Prompt':<10} {'Method':<8} {'Iter 1':<10} {'Iter 5':<10} {'Improve':<10} {'%':<8}")
print(f"{'-'*100}")

for _, row in pivot.iterrows():
    print(f"{row['country']:<20} {row['language']:<12} {row['prompt_no']:<10} {row['eval_method']:<8} "
          f"{row['iter_1_score']:<10.4f} {row['iter_5_score']:<10.4f} "
          f"{row['improvement']:<10.4f} {row['improvement_pct']:<8.2f}")

print(f"\n{'='*100}\n")

# Summary by evaluation method
print("Summary by Evaluation Method:")
print(f"{'-'*100}")
for method in ['SEM-B', 'SEM-W']:
    method_data = pivot[pivot['eval_method'] == method]
    print(f"\n{method}:")
    print(f"  Average improvement: {method_data['improvement'].mean():.4f}")
    print(f"  Median improvement: {method_data['improvement'].median():.4f}")
    print(f"  Positive improvements: {(method_data['improvement'] > 0).sum()} / {len(method_data)}")
    print(f"  Negative improvements: {(method_data['improvement'] < 0).sum()} / {len(method_data)}")

print(f"\n{'='*100}\n")

# Summary by country
print("Summary by Country:")
print(f"{'-'*100}")
country_summary = pivot.groupby('country').agg({
    'improvement': ['mean', 'std', 'count'],
    'improvement_pct': 'mean'
}).round(4)
country_summary.columns = ['avg_improvement', 'std_improvement', 'count', 'avg_pct']
country_summary = country_summary.sort_values('avg_improvement', ascending=False)

print(f"\n{'Country':<20} {'Avg Improve':<15} {'Std Dev':<15} {'Count':<10} {'Avg %':<10}")
print(f"{'-'*100}")
for country, row in country_summary.iterrows():
    print(f"{country:<20} {row['avg_improvement']:<15.4f} {row['std_improvement']:<15.4f} "
          f"{int(row['count']):<10} {row['avg_pct']:<10.2f}")

print(f"\n{'='*100}\n")

# Save to CSV
output_file = os.path.join(os.path.dirname(__file__), 'saq_iteration_improvement.csv')
pivot.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}\n")
