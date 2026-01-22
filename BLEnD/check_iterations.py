import pandas as pd

# Read the result file
df = pd.read_csv('model_inference_results/aya-101-UK_English_inst-1_result.csv')

# Check a few specific questions across iterations
test_ids = ['Al-en-01', 'Al-en-17', 'Al-en-02']

print("Checking if personas and responses differ across iterations:\n")
for test_id in test_ids:
    rows = df[df['ID'] == test_id].sort_values('iteration')
    if len(rows) > 0:
        print(f"\n{'='*80}")
        print(f"Question ID: {test_id}")
        print(f"Question: {rows.iloc[0]['Translation']}")
        print(f"{'='*80}")
        
        for idx, row in rows.iterrows():
            iter_num = row['iteration']
            persona = str(row.get('persona', 'N/A'))[:100]  # First 100 chars
            response = str(row.get('response', 'N/A'))[:100]  # First 100 chars
            print(f"\nIteration {iter_num}:")
            print(f"  Persona: {persona}...")
            print(f"  Response: {response}...")
        
        # Check if all personas are identical
        personas = rows['persona'].astype(str).values
        responses = rows['response'].astype(str).values
        persona_unique = len(set(personas)) == 1
        response_unique = len(set(responses)) == 1
        
        print(f"\n  All personas identical: {persona_unique}")
        print(f"  All responses identical: {response_unique}")
