"""Evaluation functions for initial persona generation and testing."""

import json
import os
from datasets import load_dataset
from persona_generator import generate_persona_description, cap, sanitize_json
from tools.utils import country_to_language
from llm_utils import get_llm, generate_text


def is_valid_set(ds, i):
    """Check if a set of 4 options is complete and valid.
    
    Args:
        ds: Dataset
        i: Starting index
    
    Returns:
        True if all 4 options are valid, False otherwise
    """
    for j in range(4):
        cur_row = ds[i + j]
        prompt_question = cur_row["prompt_question"]
        prompt_option = cur_row["prompt_option"]
        prompt_answer = cur_row["answer"]
        country = cur_row["country"]
        if prompt_option is None or prompt_answer is None or prompt_question is None or country is None:
            return False
    return True


def evaluate_hard_initial(ds, mode):
    """Run initial evaluation for Hard difficulty.
    
    Args:
        ds: Dataset to evaluate
        mode: Mode (eng_*, ling_*, or e2l_*)
    
    Returns:
        Tuple of (data dict, correct count, total count)
    """
    data = {}
    correct = total = 0
    iteration = 1
    prev_prompt_question = persona_description = ""
    
    for i in range(0, len(ds), 4):
        # ensure set of 4 options is complete
        isValidSet = is_valid_set(ds, i)
        
        if not isValidSet:
            continue

        # iterate over one question at a time (4 options)
        isCorrect = True
        isError = False
        for j in range(4):
            cur_row = ds[i + j]
            prompt_question = cur_row["prompt_question"]
            prompt_option = cur_row["prompt_option"]
            prompt_answer = cur_row["answer"]
            country = cur_row["country"]
            if "eng" in mode or "e2l" in mode:
                language = "English"
            else:
                language = country_to_language[cap(country)].capitalize()
            # use same persona description for same question (4 at a time)
            if prev_prompt_question != prompt_question:
                persona_description = generate_persona_description(prompt_question, country, mode)
            prev_prompt_question = prompt_question
            system_prompt = persona_description
            chat_input = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "Is this answer true or false for this question? "
                        "You must choose either True or False, and provide a brief explanation for your answer. "
                        "Respond in valid JSON format with two keys: "
                        f"\"correct\" (either \"true\" or \"false\") and \"reasoning\" (a short explanation in {language}). "
                        "Example format: {\"correct\": \"{true/false}\", \"reasoning\": \"{reasoning}\"}"
                        f"IMPORTANT: The reasoning must be in {language}. "
                        f"\nQuestion: {prompt_question}\nAnswer: {prompt_option}"
                    ),
                },
            ]
            llm_instance = get_llm()
            response = generate_text(chat_input, llm_instance)
            response = sanitize_json(response, "hard")

            try:
                result = json.loads(response)
                thinks_correct = "true" if "true" in result["correct"].lower().strip() else "false"
                reasoning = result["reasoning"].strip()
            except json.JSONDecodeError:
                # fallback if model didn't return valid JSON
                response_lower = response.lower().strip()
                thinks_correct = "true" if "true" in response_lower else "false"
                reasoning = response_lower.strip()
            # if there's an error, disregard this question (set of 4 options)
            except:
                isError = True
                print("Error parsing response: " + response)
                break

            # store data in JSON format
            data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "persona_description": persona_description, "correct_answer": prompt_answer, "persona_answer": thinks_correct, "reasoning": reasoning, "country": country, "iteration": iteration}
            # normalize to strings for comparison
            if str(thinks_correct).lower() == str(prompt_answer).lower():
                continue
            else:
                isCorrect = False

        if isError:
            continue
        if isCorrect:
            correct += 1
        total += 1
    
    return data, correct, total


def evaluate_easy_initial(ds, mode):
    """Run initial evaluation for Easy difficulty.
    
    Args:
        ds: Dataset to evaluate
        mode: Mode (eng_*, ling_*, or e2l_*)
    
    Returns:
        Tuple of (data dict, correct count, total count)
    """
    data = {}
    correct = total = 0
    
    for i in range(len(ds)): 
        cur_row = ds[i]
        prompt_question = cur_row["prompt_question"]
        option_a = cur_row["prompt_option_a"]
        option_b = cur_row["prompt_option_b"]
        option_c = cur_row["prompt_option_c"]
        option_d = cur_row["prompt_option_d"]
        answer = cur_row["answer"]
        country = cur_row["country"]
        if option_a is None or option_b is None or option_c is None or option_d is None or answer is None or prompt_question is None or country is None:
            continue
        persona_description = generate_persona_description(prompt_question, country, mode)
        system_prompt = persona_description

        if "eng" in mode or "e2l" in mode:
            language = "English"
        else:
            language = country_to_language[cap(country)].capitalize()
        chat_input = [
            {"role": "system",
            "content": system_prompt},
            {"role": "user",
            "content": "Instruction: You must select one option among A,B,C,D. "
            "Respond in valid JSON format with two keys: "
            f"\"answer\" (either \"A\", \"B\", \"C\", or \"D\") and \"reasoning\" (a short explanation in {language}). "
            "Example format: {\"answer\": \"{A/B/C/D}\", \"reasoning\": \"{reasoning}\"}"
            f"IMPORTANT: The reasoning must be in {language}. "
            "\nQuestion: " + prompt_question + "\nA. " + option_a + "\nB. " + option_b + "\nC. " + option_c + "\nD. " + option_d}
        ]
        llm_instance = get_llm()
        response = generate_text(chat_input, llm_instance)
        response = sanitize_json(response, "easy")
        
        try:
            result = json.loads(response)
            response_answer = result["answer"].upper().strip()
            reasoning = result["reasoning"].strip()
            
            # store data in JSON format
            data[i] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "persona_description": persona_description, "correct_answer": answer, "persona_answer": response_answer, "reasoning": reasoning, "country": country, "iteration": 1}
            # normalize to strings for comparison
            if response_answer.upper() == answer.upper():
                correct += 1
            total += 1
        except:
            print("Error parsing response: " + response)
            continue
    
    return data, correct, total


def run_initial_eval(difficulty, mode, num_iterations):
    """Run initial evaluation (i1) for the given difficulty.
    
    Args:
        difficulty: "Easy" or "Hard"
        mode: Mode (eng_*, ling_*, or e2l_*)
        num_iterations: Total number of iterations (used for file naming)
    
    Returns:
        Tuple of (accuracy, file_name)
    """
    ds = load_dataset("kellycyy/CulturalBench", f"CulturalBench-{difficulty}", split="test")

    if difficulty == "Hard":
        data, correct, total = evaluate_hard_initial(ds, mode)
    else:
        data, correct, total = evaluate_easy_initial(ds, mode)
    
    # write results to file (initial write)
    file_name = f"../results/{mode[-2:]}/{mode[:-3]}/i{num_iterations}/persona_{difficulty}.jsonl"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for entry in data.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    accuracy = correct / total if total > 0 else 0
    print(f"Initial Persona Accuracy for {difficulty}: {accuracy}")
    return accuracy, file_name

