import json
import sys

def parse_persona_answer(persona_answer):
    """
    Parse the persona's answer to determine if they said True or False.
    Returns True, False, or None if unclear.
    """
    answer = persona_answer.strip().lower()
    
    # Check for True indicators
    if answer.startswith('true'):
        return True
    
    # Check for False indicators  
    if answer.startswith('false'):
        return False
    
    # Handle other cases - these are considered incorrect
    return None

def calculate_group_accuracy():
    """
    Calculate accuracy where a group of 4 questions is only correct 
    if all 4 questions are answered correctly.
    """
    results = []
    
    # Read all results
    with open('results/vanilla/vanilla_Hard.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip non-JSON lines (like summary lines)
                    continue
    
    total_questions = len(results)
    print(f"Total individual questions: {total_questions}")
    
    # Group into sets of 4
    num_groups = total_questions // 4
    print(f"Total groups of 4: {num_groups}")
    
    if total_questions % 4 != 0:
        print(f"Warning: {total_questions % 4} questions left over (not forming a complete group)")
    
    correct_groups = 0
    
    # Check each group of 4
    for i in range(num_groups):
        start_idx = i * 4
        end_idx = start_idx + 4
        group = results[start_idx:end_idx]
        
        # Check if all 4 in the group are correct
        group_correct = True
        for item in group:
            correct_answer = item['correct_answer']
            persona_answer_parsed = parse_persona_answer(item['persona_answer'])
            
            if persona_answer_parsed != correct_answer:
                group_correct = False
        
        if group_correct:
            correct_groups += 1

    # Calculate accuracy
    accuracy = (correct_groups / num_groups * 100) if num_groups > 0 else 0
    
    print(f"\nCorrect groups: {correct_groups}")
    print(f"Incorrect groups: {num_groups - correct_groups}")
    print(f"Total groups: {num_groups}")
    print(f"Accuracy: {accuracy:.2f}%")

    
    return accuracy

if __name__ == "__main__":
    calculate_group_accuracy()

