"""Functions for running iterative persona refinement."""

import json
from persona_generator import generate_new_persona, cap, sanitize_json
from tools.utils import country_to_language
from llm_utils import get_llm, generate_text


def append_to_file(file_name, new_data, correct, total, iteration):
    """Append iteration data to existing file.
    
    Args:
        file_name: Path to file
        new_data: Dictionary of new data entries
        correct: Number of correct answers
        total: Total number of questions
        iteration: Current iteration number
    
    Returns:
        Accuracy for this iteration
    """
    with open(file_name, "a") as f:
        for entry in new_data.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    accuracy = correct / total if total > 0 else 0
    print(f"Iteration {iteration} Accuracy: {accuracy}")
    return accuracy


def run_easy_iterations(mode, num_iterations, file_name):
    """Run iterations for Easy difficulty.
    
    Args:
        mode: Mode (eng_*, ling_*, or e2l_*)
        num_iterations: Total number of iterations
        file_name: Path to file containing results
    
    Returns:
        List of accuracies for each iteration
    """
    accuracies = []
    
    for cur_iteration in range(2, num_iterations + 1):
        # read all data from file (includes previous iterations)
        with open(file_name, "r") as f:
            lines = f.readlines()
            # only get the most recent iteration's data (from previous iteration)
            data = [json.loads(line) for line in lines if line.strip() and "Persona Accuracy" not in line]
            # filter to get only the previous iteration's data
            prev_iteration = cur_iteration - 1
            data = [item for item in data if item.get("iteration") == prev_iteration]
        
        correct = total = 0
        new_data = {}
        for i, item in enumerate(data):

            # parse response from self-refinement prompt with CoT reasoning
            try:
                refine_response = generate_new_persona("Easy", item["question"], item["persona_description"], item["options"][item["persona_answer"]], item["reasoning"], mode, item["country"], cur_iteration)
                result = json.loads(refine_response)
                new_persona = result["revised_persona"]
                refine_reasoning = result["reasoning"]
            except Exception as e:
                print(e)
                print("Error parsing response from refine prompt: " + refine_response)
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
            chat_input = [
                {"role": "system",
                "content": new_persona},
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
                correct_answer = item["correct_answer"]
                
                # store data in JSON format
                new_data[i] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "persona_description": new_persona, "refine_reasoning": refine_reasoning, "correct_answer": correct_answer, "persona_answer": response_answer, "reasoning": reasoning, "country": item["country"], "iteration": cur_iteration}
                # normalize to strings for comparison
                if response_answer.upper() == correct_answer.upper():
                    correct += 1
                total += 1
            except:
                print("Error parsing response: " + response)
                continue

        accuracy = append_to_file(file_name, new_data, correct, total, cur_iteration)
        accuracies.append(accuracy)
    
    return accuracies


def run_hard_iterations(mode, num_iterations, file_name):
    """Run iterations for Hard difficulty.
    
    Args:
        mode: Mode (eng_*, ling_*, or e2l_*)
        num_iterations: Total number of iterations
        file_name: Path to file containing results
    
    Returns:
        List of accuracies for each iteration
    """
    accuracies = []
    
    for cur_iteration in range(2, num_iterations + 1):
        # read all data from file (includes previous iterations)
        with open(file_name, "r") as f:
            lines = f.readlines()
            # only get the most recent iteration's data (from previous iteration)
            data = [json.loads(line) for line in lines if line.strip() and not line.startswith("Persona Accuracy")]
            # filter to get only the previous iteration's data
            prev_iteration = cur_iteration - 1
            data = [item for item in data if item.get("iteration") == prev_iteration]
        
        correct = total = 0
        new_data = {}
        for i in range(0, len(data), 4):
            prompt_question = data[i]["question"]
            qa_responses = ""
            for j in range(4):
                qa_responses += data[i + j]["prompt_option"] + ": " + data[i + j]["persona_answer"] + "\n" + "reasoning: " + data[i + j]["reasoning"] + "\n\n"
            isError = False
            isCorrect = True

            # parse response from self-refinement prompt with CoT reasoning
            try:
                refine_response = generate_new_persona("Hard", prompt_question, data[i]["persona_description"], qa_responses, data[i]["reasoning"], mode, data[i]["country"], cur_iteration)
                result = json.loads(refine_response)
                new_persona = result["revised_persona"]
                refine_reasoning = result["reasoning"]
            except:
                isError = True
                print("Error parsing response from refine prompt: " + refine_response)
                continue

            if "eng" in mode or "e2l" in mode:
                language = "English"
            else:
                language = country_to_language[cap(data[i]["country"])].capitalize()
            for j in range(4):
                prompt_option = data[i + j]["prompt_option"]
                correct_answer = data[i + j]["correct_answer"]
                chat_input = [
                    {
                        "role": "system",
                        "content": new_persona,
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
                new_data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "persona_description": new_persona, "refine_reasoning": refine_reasoning, "correct_answer": correct_answer, "persona_answer": thinks_correct, "reasoning": reasoning, "country": data[i + j]["country"], "iteration": cur_iteration}
                # normalize to strings for comparison
                if str(thinks_correct).lower() == str(correct_answer).lower():
                    continue
                else:
                    isCorrect = False

            if isError:
                continue
            if isCorrect:
                correct += 1
            total += 1

        accuracy = append_to_file(file_name, new_data, correct, total, cur_iteration)
        accuracies.append(accuracy)
    
    return accuracies


def run_iterations(mode, num_iterations, difficulty, file_name):
    """Run iterations starting from iteration 2.
    
    Args:
        mode: Mode (eng_*, ling_*, or e2l_*)
        num_iterations: Total number of iterations
        difficulty: "Easy" or "Hard"
        file_name: Path to file containing results
    
    Returns:
        List of accuracies for each iteration
    """
    if difficulty == "Easy":
        return run_easy_iterations(mode, num_iterations, file_name)
    else:
        return run_hard_iterations(mode, num_iterations, file_name)

