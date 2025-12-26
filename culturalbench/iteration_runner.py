"""Functions for running iterative persona refinement."""

import json
from persona_generator import generate_new_persona, cap
from tools.utils import country_to_language
from tools.llm_utils import get_llm, generate_text_funcs, cleanup, pre_generate_all_feedback
from tools import llm_utils
from tools.db.db_utils import save_results, save_accuracy, load_previous_iteration, load_results
import json_repair


def append_to_db(db_path, new_data, correct, total, iteration, difficulty, mode):
    """Append iteration data to database.
    
    Args:
        db_path: Path to database file
        new_data: Dictionary of new data entries
        correct: Number of correct answers
        total: Total number of questions
        iteration: Current iteration number
        difficulty: "Easy" or "Hard"
        mode: Mode string (e.g., "eng")
    
    Returns:
        Accuracy for this iteration
    """
    mode_for_db = f"{mode}_p1"
    save_results(db_path, new_data, difficulty, mode_for_db)
    
    accuracy = correct / total if total > 0 else 0
    save_accuracy(db_path, iteration, difficulty, mode_for_db, accuracy, correct, total)
    
    print(f"Iteration {iteration} Accuracy: {accuracy}")
    return accuracy


async def run_easy_iterations(mode, num_iterations, db_path):
    """Run iterations for Easy difficulty.
    
    Args:
        mode: Mode (eng_*, ling_*, or e2l_*)
        num_iterations: Total number of iterations
        db_path: Path to database file containing results
    
    Returns:
        List of accuracies for each iteration
    """
    accuracies = []
    is_translation_mode = "e2l" in mode or "l2e" in mode
    
    for cur_iteration in range(2, num_iterations + 1):
        # Load data from previous iteration
        data = load_previous_iteration(db_path, cur_iteration)
        print(f"Currently running iteration {cur_iteration}")
        
        # Step 1: Cleanup main LLM
        print(f"[Iteration {cur_iteration}] Cleaning up main LLM...")
        cleanup()
        
        # Step 2: Pre-generate all feedback using external LLM
        print(f"[Iteration {cur_iteration}] Pre-generating feedback for {len(data)} questions...")
        pre_generate_all_feedback(data, "Easy")
        
        # Step 3: Cleanup external LLM
        print(f"[Iteration {cur_iteration}] Cleaning up external LLM...")
        cleanup()
        
        correct = total = 0
        new_data = {}
        for i, item in enumerate(data):

            print(f"Question {i} of {len(data)}")

            # parse response from self-refinement prompt with CoT reasoning
            try:
                # Use only the previous iteration
                old_persona = (
                    item["persona_description"] 
                    if "l2e" not in mode and "e2l" not in mode
                    else item["pretranslated_persona"]
                )
                # provide all 4 options for persona refinement
                prev_answers = ""
                for option in item["options"]:
                    prev_answers += option + ": " + item["options"][option] + "\n"
                prev_answers += "Model answer: " + item["model_answer"]
                previous_personas_data = {
                    'persona': old_persona,
                    'model_answer': prev_answers,
                    'reasoning': item["reasoning"]
                }

                pretranslated, refine_response = await generate_new_persona(
                    "Easy",
                    item["question"],
                    previous_personas_data,
                    mode,
                    item["country"],
                    question_index=i,
                )
                # if not in correct language, disregard this question
                if refine_response is None and is_translation_mode:
                    continue
                
                result = None
                if is_translation_mode:
                    result = json_repair.loads(refine_response)
                else:
                    result = json_repair.loads(pretranslated)

                new_persona = result["revised_persona"]
                refine_reasoning = result["reasoning"]

            except Exception as e:
                print(f"Error generating/parsing persona for question {i}: {type(e).__name__}: {str(e)}")
                continue

            prompt_question = item["question"]
            option_a = item["options"]["A"]
            option_b = item["options"]["B"]
            option_c = item["options"]["C"]
            option_d = item["options"]["D"]
            
            country_words = item["country"].split(" ")
            for j in range(len(country_words)):
                country_words[j] = country_words[j].capitalize()
            country = " ".join(country_words)
            
            if "eng" in mode or "e2l" in mode:
                language = "English"
            else:
                language = country_to_language[cap(country)]
            
            # For qwen3 thinking mode in "ling" mode, add instruction to think in the relevant language
            thinking_instruction = ""
            if "ling" in mode and llm_utils.MODEL_NAME == "Qwen/Qwen3-4B":
                thinking_instruction = f"You MUST write internal reasoning inside <think>...</think> in {language}. If any part of <think>...</think> is not {language}, regenerate the reasoning.\n\n"
            
            chat_input = [
                {
                    "role": "system",
                    "content": new_persona
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
                        f"Question: {prompt_question}\n"
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
            except Exception as e:
                print(f"Error sanitizing JSON for response: {response}")
                pass
            
            try:
                response_answer = result["answer"].upper().strip()
                reasoning = result["reasoning"].strip()
                correct_answer = item["correct_answer"]
                
                # store data in JSON format
                options_dict = {
                    "A": option_a,
                    "B": option_b,
                    "C": option_c,
                    "D": option_d
                }
                
                base_data = {
                    "question": prompt_question,
                    "options": options_dict,
                    "persona_description": new_persona,
                    "refine_reasoning": refine_reasoning,
                    "correct_answer": correct_answer,
                    "model_answer": response_answer,
                    "reasoning": reasoning,
                    "country": item["country"],
                    "iteration": cur_iteration
                }
                
                if "l2e" in mode or "e2l" in mode:
                    base_data["pretranslated_persona"] = pretranslated
                if thinking_content is not None:
                    base_data["thinking_content"] = thinking_content
                
                new_data[i] = base_data
                
                # normalize to strings for comparison
                if response_answer.upper() == correct_answer.upper():
                    correct += 1
                total += 1
            except Exception as e:
                print(f"Error generating answer for question {i}: {type(e).__name__}: {str(e)}")
                continue

        accuracy = append_to_db(db_path, new_data, correct, total, cur_iteration, "Easy", mode)
        accuracies.append(accuracy)
    
    return accuracies


async def run_hard_iterations(mode, num_iterations, db_path):
    """Run iterations for Hard difficulty.
    
    Args:
        mode: Mode (eng_*, ling_*, or e2l_*)
        num_iterations: Total number of iterations
        db_path: Path to database file containing results
    
    Returns:
        List of accuracies for each iteration
    """
    accuracies = []
    is_translation_mode = "e2l" in mode or "l2e" in mode
    
    for cur_iteration in range(2, num_iterations + 1):
        # Load data from previous iteration
        data = load_previous_iteration(db_path, cur_iteration)
        print(f"Currently running iteration {cur_iteration}")
        
        # Step 1: Cleanup main LLM
        print(f"[Iteration {cur_iteration}] Cleaning up main LLM...")
        cleanup()
        
        # Step 2: Pre-generate all feedback using external LLM
        print(f"[Iteration {cur_iteration}] Pre-generating feedback for {len(data)} questions...")
        pre_generate_all_feedback(data, "Hard")
        
        # Step 3: Cleanup external LLM
        print(f"[Iteration {cur_iteration}] Cleaning up external LLM...")
        cleanup()
        
        # Step 4: Initialize main LLM for processing
        print(f"[Iteration {cur_iteration}] Initializing main LLM...")
        get_llm()  # Initialize main LLM
        
        correct = total = 0
        new_data = {}
        
        for i in range(0, len(data), 4):
            prompt_question = data[i]["question"]
            
            isError = False
            isCorrect = True

            # parse response from self-refinement prompt with CoT reasoning
            try:
                # Use only the previous iteration
                old_persona = (
                    data[i]["persona_description"]
                    if "l2e" not in mode and "e2l" not in mode
                    else data[i]["pretranslated_persona"]
                )
                previous_personas_data = {
                    'persona': old_persona,
                    'reasoning': data[i]["reasoning"]
                }
                
                pretranslated, refine_response = await generate_new_persona(
                    "Hard",
                    prompt_question,
                    previous_personas_data,
                    mode,
                    data[i]["country"],
                    question_index=i,
                )
                # if not in correct language, disregard this question (set of 4 options)
                if refine_response is None and is_translation_mode:
                    continue
                
                result = None
                if is_translation_mode:
                    result = json_repair.loads(refine_response)
                else:
                    result = json_repair.loads(pretranslated)
                    
                new_persona = result["revised_persona"]
                refine_reasoning = result["reasoning"]
            except Exception as e:
                isError = True
                print(f"Error generating/parsing persona for question set {i//4}: {type(e).__name__}: {str(e)}")
                continue

            if "eng" in mode or "e2l" in mode:
                language = "English"
            else:
                language = country_to_language[cap(data[i]["country"])].capitalize()
            
            # For qwen3 thinking mode in "ling" mode, add instruction to think in the relevant language
            thinking_instruction = ""
            if "ling" in mode and llm_utils.MODEL_NAME == "Qwen/Qwen3-4B":
                thinking_instruction = f"You MUST write internal reasoning inside <think>...</think> in {language}. If any part of <think>...</think> is not {language}, regenerate the reasoning.\n\n"
            
            cur_set_data = []
            for j in range(4):
                prompt_option = data[i + j]["prompt_option"]
                correct_answer = data[i + j]["correct_answer"]
                
                chat_input = [
                    {
                        "role": "system",
                        "content": new_persona
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
                result = json_repair.loads(response)
                
                try:
                    thinks_correct = (
                        "true" 
                        if "true" in result["correct"].lower().strip() 
                        else "false"
                    )
                    reasoning = result["reasoning"].strip()
                except json.JSONDecodeError:
                    # fallback if model didn't return valid JSON
                    response_lower = response.lower().strip()
                    thinks_correct = "true" if "true" in response_lower else "false"
                    reasoning = response_lower.strip()
                # if there's an error, disregard this question (set of 4 options)
                except Exception as e:
                    isError = True
                    print(f"Error generating answer for option {j} in question set {i//4}: {type(e).__name__}: {str(e)}")
                    break

                # store data in JSON format
                item_data = {
                    "question": prompt_question,
                    "prompt_option": prompt_option,
                    "persona_description": new_persona,
                    "refine_reasoning": refine_reasoning,
                    "correct_answer": correct_answer,
                    "model_answer": thinks_correct,
                    "reasoning": reasoning,
                    "country": data[i + j]["country"],
                    "iteration": cur_iteration
                }
                
                if "l2e" in mode or "e2l" in mode:
                    item_data["pretranslated_persona"] = pretranslated
                if thinking_content is not None:
                    item_data["thinking_content"] = thinking_content
                
                cur_set_data.append(item_data)
                
                # normalize to strings for comparison (convert 0/1 to false/true)
                correct_str = str(correct_answer).lower().strip()
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
                new_data[i + j] = cur_set_data[j]
            
            if isCorrect:
                correct += 1
            total += 1

        accuracy = append_to_db(db_path, new_data, correct, total, cur_iteration, "Hard", mode)
        accuracies.append(accuracy)
    
    return accuracies


async def run_iterations(mode, num_iterations, difficulty, db_path):
    """Run iterations starting from iteration 2.
    
    Args:
        mode: Mode (eng_*, ling_*, or e2l_*)
        num_iterations: Total number of iterations
        difficulty: "Easy" or "Hard"
        db_path: Path to database file containing results
    
    Returns:
        List of accuracies for each iteration
    """
    if difficulty == "Easy":
        return await run_easy_iterations(mode, num_iterations, db_path)
    else:
        return await run_hard_iterations(mode, num_iterations, db_path)

