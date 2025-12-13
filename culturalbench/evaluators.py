"""Evaluation functions for initial persona generation and testing."""

import json
import os
from datasets import load_dataset
from persona_generator import generate_persona_description, cap
from tools.utils import country_to_language
from tools.llm_utils import get_llm, generate_text_funcs
from tools import llm_utils
from tools.db.db_utils import save_results, save_accuracy
import json_repair


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


async def evaluate_hard_initial(ds, mode):
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
    persona_description = ""
    
    for i in range(0, len(ds), 4):
        # ensure set of 4 options is complete
        isValidSet = is_valid_set(ds, i)
        
        if not isValidSet:
            continue

        # iterate over one question at a time (4 options)
        isCorrect = True
        isError = False
        cur_set_data = []
        
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
            if j == 0:
                pretranslated, translated = await generate_persona_description(
                    prompt_question, 
                    country, 
                    mode
                )
                if "l2e" in mode or "e2l" in mode:
                    persona_description = translated
                else:
                    persona_description = pretranslated
                # if not in correct language, disregard this question (set of 4 options)
                if persona_description is None:
                    break
            
            # For qwen3 thinking mode in "ling" mode, add instruction to think in the relevant language
            thinking_instruction = ""
            if "ling" in mode and llm_utils.MODEL_NAME == "Qwen/Qwen3-4B":
                thinking_instruction = f"You MUST write internal reasoning inside <think>...</think> in {language}. If any part of <think>...</think> is not {language}, regenerate the reasoning.\n\n"
            
            system_prompt = persona_description
            chat_input = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": (
                        thinking_instruction +
                        "Is this answer true or false for this question?\n"
                        "You must choose either True or False, and provide a brief "
                        "explanation for your answer.\n"
                        "Respond in valid JSON format with two keys: \n"
                        f"\"correct\" (either \"true\" or \"false\") and "
                        f"\"reasoning\" (a short explanation in {language}). \n"
                        "Example format: {\"correct\": \"{true/false}\", \"reasoning\": \"{reasoning}\"}\n"
                        f"IMPORTANT: The reasoning must be in {language}.\n"
                        f"Question: {prompt_question}\n"
                        f"Answer: {prompt_option}"
                    )
                }
            ]
            
            llm_instance = get_llm()
            thinking_content, response = generate_text_funcs[llm_utils.MODEL_NAME](llm_instance, chat_input, enable_thinking_bool=True)

            try:
                result = json_repair.loads(response)
                thinks_correct = (
                    "true" 
                    if "true" in result["correct"].lower().strip() 
                    else "false"
                )
                reasoning = result["reasoning"].strip()
            except Exception as e:
                isError = True
                print("Error parsing response: " + response + " " + str(e))
                break

            # store data in JSON format
            item_data = {
                "question": prompt_question,
                "prompt_option": prompt_option,
                "persona_description": persona_description,
                "correct_answer": prompt_answer,
                "model_answer": thinks_correct,
                "reasoning": reasoning,
                "country": country,
                "iteration": iteration
            }
            
            if "l2e" in mode or "e2l" in mode:
                item_data["pretranslated_persona"] = pretranslated
            if thinking_content is not None:
                item_data["thinking_content"] = thinking_content
                
            cur_set_data.append(item_data)
            
            # normalize to strings for comparison (convert 0/1 to false/true)
            correct_str = str(prompt_answer).lower().strip()
            if correct_str in ["1", "true"]:
                expected_answer = "true"
            else:
                expected_answer = "false"
            if str(thinks_correct).lower() == expected_answer:
                continue
            else:
                isCorrect = False

        if isError or len(cur_set_data) != 4:
            continue
        
        for j in range(4):
            data[i + j] = cur_set_data[j]
        
        if isCorrect:
            correct += 1
        total += 1
    
    return data, correct, total


async def evaluate_easy_initial(ds, mode):
    """Run initial evaluation for Easy difficulty.
    
    Args:
        ds: Dataset to evaluate
        mode: Mode (eng_*, ling_*, or e2l_*)
    
    Returns:
        Tuple of (data dict, correct count, total count)
    """
    data = {}
    correct = total = 0
    
    for i in range(4): 
        cur_row = ds[i]
        prompt_question = cur_row["prompt_question"]
        option_a = cur_row["prompt_option_a"]
        option_b = cur_row["prompt_option_b"]
        option_c = cur_row["prompt_option_c"]
        option_d = cur_row["prompt_option_d"]
        answer = cur_row["answer"]
        country = cur_row["country"]

        if (option_a is None or option_b is None or option_c is None or 
            option_d is None or answer is None or prompt_question is None or 
            country is None):
            continue

        pretranslated, translated = await generate_persona_description(
            prompt_question, 
            country, 
            mode
        )
        if "l2e" in mode or "e2l" in mode:
            persona_description = translated
        else:
            persona_description = pretranslated

        # if not in correct language, disregard this question
        if persona_description is None:
            continue
        
        system_prompt = persona_description

        if "eng" in mode or "e2l" in mode:
            language = "English"
        else:
            language = country_to_language[cap(country)].capitalize()
        
        # For qwen3 thinking mode in "ling" mode, add instruction to think in the relevant language
        thinking_instruction = ""
        if "ling" in mode and llm_utils.MODEL_NAME == "Qwen/Qwen3-4B":
            thinking_instruction = f"You MUST write internal reasoning inside <think>...</think> in {language}. If any part of <think>...</think> is not {language}, regenerate the reasoning.\n\n"
        
        chat_input = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    thinking_instruction +
                    "Instruction: You must select one option among A,B,C,D.\n"
                    "Respond in valid JSON format with two keys: \n"
                    f"\"answer\" (either \"A\", \"B\", \"C\", or \"D\") and "
                    f"\"reasoning\" (a short explanation in {language}). \n"
                    "Example format: {\"answer\": \"{A/B/C/D}\", \"reasoning\": \"{reasoning}\"}\n"
                    f"IMPORTANT: The reasoning must be in {language}.\n"
                    f"\nQuestion: {prompt_question}\n"
                    f"A. {option_a}\n"
                    f"B. {option_b}\n"
                    f"C. {option_c}\n"
                    f"D. {option_d}"
                )
            }
        ]
        
        llm_instance = get_llm()
        thinking_content, response = generate_text_funcs[llm_utils.MODEL_NAME](llm_instance, chat_input, enable_thinking_bool=True)
        try:
            result = json_repair.loads(response)
            response_answer = result["answer"].upper().strip()
            reasoning = result["reasoning"].strip()
            
            # store data in JSON format
            options_dict = {
                "A": option_a,
                "B": option_b,
                "C": option_c,
                "D": option_d
            }
            
            item_data = {
                "question": prompt_question,
                "options": options_dict,
                "persona_description": persona_description,
                "correct_answer": answer,
                "model_answer": response_answer,
                "reasoning": reasoning,
                "country": country,
                "iteration": 1
            }
            
            if "l2e" in mode or "e2l" in mode:
                item_data["pretranslated_persona"] = pretranslated
            if thinking_content is not None:
                item_data["thinking_content"] = thinking_content
            
            data[i] = item_data
            
            # normalize to strings for comparison
            if response_answer.upper() == answer.upper():
                correct += 1
            total += 1
        except:
            print("Error parsing response: " + response)
            continue
    
    return data, correct, total


async def run_initial_eval(difficulty, mode, custom=None):
    """Run initial evaluation (i1) for the given difficulty.
    
    Args:
        difficulty: "Easy" or "Hard"
        mode: Mode (eng_*, ling_*, or e2l_*)
        custom: Optional custom suffix to append to database path
    
    Returns:
        Tuple of (accuracy, db_path)
    """
    ds = load_dataset("kellycyy/CulturalBench", f"CulturalBench-{difficulty}", split="test")

    if difficulty == "Hard":
        data, correct, total = await evaluate_hard_initial(ds, mode)
    else:
        data, correct, total = await evaluate_easy_initial(ds, mode)

    model_to_save = {
        "Qwen/Qwen3-4B": "qwen3_4b",
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b",
    }
    
    # write results to database (initial write)
    db_path = f"../results/{mode[-2:]}/{mode[:-3]}/{difficulty.lower()}_t{llm_utils.TEMPERATURE}_{model_to_save[llm_utils.MODEL_NAME]}"
    if custom:
        db_path += f"_{custom}"
    db_path += ".db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    save_results(db_path, data, difficulty, mode)
    
    accuracy = correct / total if total > 0 else 0
    save_accuracy(db_path, 1, difficulty, mode, accuracy, correct, total)
    
    print(f"Initial Persona Accuracy for {difficulty}: {accuracy}")
    return accuracy, db_path

