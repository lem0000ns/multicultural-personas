import sys
import os
from db_utils import load_results


def calculate_consistency(mode, temperature, model_name):
    """Calculate easy-hard consistency for a given mode + prompt"""
    db_path_easy = f"../../../results/{mode[-2:]}/{mode[:-3]}/easy_t{temperature}_{model_name}.db"
    db_path_hard = f"../../../results/{mode[-2:]}/{mode[:-3]}/hard_t{temperature}_{model_name}.db"

    print("\n\nCalculating consistency for", mode, temperature, model_name)
    print("=" * 50)

    for iteration in range(1, 6):
        cur_results_easy = load_results(db_path_easy, iteration=iteration)
        cur_results_hard = load_results(db_path_hard, iteration=iteration)
        
        correct_easy = correct_hard = easy_total = hard_total = 0
        correct_both = wrong_both = only_easy = only_hard = 0
        wrong_easy_correct_hard_questions = []  # Track questions wrong in easy but correct in hard

        easy_question_dict = {}

        for result in cur_results_easy:
            cur_question = result['question']

            response_answer = result['model_answer']
            correct_answer = result['correct_answer']
            is_correct = response_answer.upper().strip() == correct_answer.upper().strip()
            correct_easy += is_correct

            easy_question_dict[cur_question] = is_correct
            easy_total += 1
        
        hard_question_dict = {}

        for i in range(0, len(cur_results_hard), 4):
            cur_question = cur_results_hard[i]['question']
            is_correct = True
            for j in range(4):

                thinks_correct = "true" if "true" in cur_results_hard[i+j]['model_answer'].lower().strip() else "false"

                correct_answer = cur_results_hard[i+j]['correct_answer']
                correct_str = str(correct_answer).lower().strip()
                if correct_str in ["1", "true"]:
                    expected_answer = "true"
                else:
                    expected_answer = "false"
                
                if str(thinks_correct).lower() == expected_answer:
                    continue
                else:
                    is_correct = False
                    break
            
            hard_question_dict[cur_question] = is_correct
            correct_hard += is_correct
            hard_total += 1
        
        total = 0
        consistent = 0
        for question in easy_question_dict:
            if question not in hard_question_dict:
                continue

            total += 1
            if easy_question_dict[question] == hard_question_dict[question]:
                consistent += 1

            if easy_question_dict[question] and not hard_question_dict[question]:
                only_easy += 1
            elif not easy_question_dict[question] and hard_question_dict[question]:
                only_hard += 1
                wrong_easy_correct_hard_questions.append(question)
            elif not easy_question_dict[question] and not hard_question_dict[question]:
                wrong_both += 1
            elif easy_question_dict[question] and hard_question_dict[question]:
                correct_both += 1
        
        easy_accuracy = correct_easy / easy_total
        hard_accuracy = correct_hard / hard_total
        random_consistency_baseline = (easy_accuracy * hard_accuracy) + (1 - easy_accuracy) * (1 - hard_accuracy)
        
        print(f"\nIteration {iteration}: {consistent}/{total} consistent ({consistent/total:.2%})")
        print(f"Random consistency baseline: {random_consistency_baseline:.2%}")
        print(f"Only easy: {only_easy}")
        print(f"Only hard: {only_hard}")
        print(f"Correct both: {correct_both}")
        print(f"Wrong both: {wrong_both}")
        
        # Print questions where model got wrong in easy but correct in hard
        if wrong_easy_correct_hard_questions:
            print(f"\nQuestions wrong in Easy but correct in Hard ({len(wrong_easy_correct_hard_questions)}):\n")
            for q in wrong_easy_correct_hard_questions[:3]:
                print(q)
        
        print('-' * 50)

if __name__ == "__main__":
    calculate_consistency("eng_p1", 0.0, "llama3_8b")
    # calculate_consistency("ling_p1", 0.0, "llama3_8b")
    # calculate_consistency("l2e_p1", 0.0, "llama3_8b")
    # calculate_consistency("e2l_p1", 0.0, "llama3_8b")