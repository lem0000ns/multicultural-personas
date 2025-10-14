from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
import os
import torch
import gc
from tools.utils import country_to_language
from tools.configs import system_prompts, self_refine_prompt_easy, self_refine_prompt_hard
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = None

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

def get_llm():
    global llm
    if llm is None:
        llm = LLM(model=model_name, tensor_parallel_size=2, dtype='half')
    return llm

def generate_text(chat_input, llm_instance):
    output = llm_instance.chat(chat_input, sampling_params)
    return output[0].outputs[0].text

def generate_persona_description(question, country, mode):   
    llm_instance = get_llm()
    language = country_to_language[country].lower()
    system_prompt = system_prompts[mode] if "eng" in mode else system_prompts[mode](language)
    chat_input = [
        {"role": "system",
        "content": system_prompt
        },
        {"role": "user",
        "content": "question: " + question + "\n\n" + "country: " + country + "\n\npersona description: "}
    ]
    response = generate_text(chat_input, llm_instance)
    return response

def is_valid_set(ds, i):
    for j in range(4):
        cur_row = ds[i + j]
        prompt_question = cur_row["prompt_question"]
        prompt_option = cur_row["prompt_option"]
        prompt_answer = cur_row["answer"]
        country = cur_row["country"]
        if prompt_option is None or prompt_answer is None or prompt_question is None or country is None:
            return False
    return True

def run_initial_eval(difficulty, mode):
    """Run initial evaluation (i1) - adapted from evaluate.py"""
    print(f"Running initial persona evaluation for {difficulty} difficulty with {mode} mode")
    ds = load_dataset("kellycyy/CulturalBench", f"CulturalBench-{difficulty}", split="test")

    data = {}
    correct = total = 0

    # CulturalBench-Hard
    if difficulty == "Hard":
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
                            "\"correct\" (either \"true\" or \"false\") and \"reasoning\" (a short explanation). "
                            "Example format: {\"correct\": \"true\", \"reasoning\": \"The answer is true because ...\"}"
                            f"\nQuestion: {prompt_question}\nAnswer: {prompt_option}"
                        ),
                    },
                ]
                llm_instance = get_llm()
                response = generate_text(chat_input, llm_instance)

                try:
                    result = json.loads(response)
                    thinks_correct = "true" if "true" in result["correct"].lower().strip() else "false"
                    reasoning = result["reasoning"].strip()
                except json.JSONDecodeError:
                    # Fallback if model didn't return valid JSON
                    response_lower = response.lower().strip()
                    thinks_correct = "true" if "true" in response_lower else "false"
                    reasoning = response_lower.strip()
                # if there's an error, disregard this question (set of 4 options)
                except:
                    isError = True
                    print("Error parsing response: " + response)
                    break

                # store data in JSON format
                data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "persona_description": persona_description, "correct_answer": prompt_answer, "persona_answer": thinks_correct, "reasoning": reasoning, "country": country}
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
    # CulturalBench-Easy
    else:
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
            
            chat_input = [
                {"role": "system",
                "content": system_prompt},
                {"role": "user",
                "content": "Instruction: You must select one option among A,B,C,D. "
                "Respond in valid JSON format with two keys: "
                "\"answer\" (either \"A\", \"B\", \"C\", or \"D\") and \"reasoning\" (a short explanation). "
                "Example format: {\"answer\": \"A\", \"reasoning\": \"The answer is A because ...\"}"
                "\nQuestion: " + prompt_question + "\nA. " + option_a + "\nB. " + option_b + "\nC. " + option_c + "\nD. " + option_d}
            ]
            llm_instance = get_llm()
            response = generate_text(chat_input, llm_instance)
            
            try:
                result = json.loads(response)
                response_answer = result["answer"].upper().strip()
                reasoning = result["reasoning"].strip()
                
                # store data in JSON format
                data[i] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "persona_description": persona_description, "correct_answer": answer, "persona_answer": response_answer, "reasoning": reasoning, "country": country}
                # normalize to strings for comparison
                if response_answer.upper() == answer.upper():
                    correct += 1
                total += 1
            except:
                print("Error parsing response: " + response)
                continue
    
    # write results to i1 folder
    file_name = f"../results/{mode[-2:]}/{mode[:-3]}/i1/persona_{difficulty}.jsonl"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for entry in data.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.write(f"Persona Accuracy for {difficulty}: {correct / total}\n")
    
    print(f"Initial Persona Accuracy for {difficulty}: {correct / total}")

def generate_new_persona(difficulty, question, old_persona, pred_ans, reasoning, mode, country, iteration):
    llm_instance = get_llm()
    if "eng" in mode:
        second_person_note = "Always start with 'You are'... followed by the persona description."
    else:
        second_person_note = "Always start with second-person pronoun followed by the persona description."
    
    # self-refinement
    if iteration > 1:
        if difficulty == "Easy":
            self_refine_prompt = self_refine_prompt_easy.format(second_person_note=second_person_note)
        else:
            self_refine_prompt = self_refine_prompt_hard.format(second_person_note=second_person_note)
        chat_input = [
            {"role": "system",
            "content": self_refine_prompt},
            {"role": "user",
            "content": "question: " + question + "\n\n" + "previous persona: " + old_persona + "\n\n" + "predicted answer: " + pred_ans + "\n\n" + "reasoning: " + reasoning + "\n\n" + "new persona: "}
        ]
    # initial prompt (no refinement)
    else:
        language = country_to_language[country].lower()
        chat_input = [
            {"role": "system",
            "content": system_prompts[mode] if "eng" in mode else system_prompts[mode](language)},
            {"role": "user",
            "content": "question: " + question + "\n\n" + "country: " + country + "\n\npersona description: "}
        ]
    response = generate_text(chat_input, llm_instance)
    return response

def write_to_file(mode, cur_iteration, difficulty, correct, total, new_data):
    file_name = f"../results/{mode[-2:]}/{mode[:-3]}/i{cur_iteration}/persona_{difficulty}.jsonl"

    # make sure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "w") as f:
        for entry in new_data.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.write(f"Persona Accuracy for {difficulty}: {correct / total}\n")
    
    print(f"Persona Accuracy for {difficulty}: {correct / total}")

def run_iterations(mode, num_iterations, difficulty="Easy"):
    """Run iterations starting from iteration 2 (i1 should already exist)"""
    if difficulty == "Easy":
        for cur_iteration in range(2, num_iterations + 1):
            prev_iteration = cur_iteration - 1
            with open(f"../results/{mode[-2:]}/{mode[:-3]}/i{prev_iteration}/persona_Easy.jsonl", "r") as f:
                lines = f.readlines()
                data = [json.loads(line) for line in lines[:-1]]
            correct = total = 0
            new_data = {}
            for i, item in enumerate(data):
                new_persona = generate_new_persona(difficulty, item["question"], item["persona_description"], item["options"][item["persona_answer"]], item["reasoning"], mode, item["country"], cur_iteration)
                prompt_question = item["question"]
                option_a = item["options"]["A"]
                option_b = item["options"]["B"]
                option_c = item["options"]["C"]
                option_d = item["options"]["D"]
                chat_input = [
                    {"role": "system",
                    "content": new_persona},
                    {"role": "user",
                    "content": "Instruction: You must select one option among A,B,C,D. "
                    "Respond in valid JSON format with two keys: "
                    "\"answer\" (either \"A\", \"B\", \"C\", or \"D\") and \"reasoning\" (a short explanation). "
                    "Example format: {\"answer\": \"A\", \"reasoning\": \"The answer is A because ...\"}"
                    "\nQuestion: " + prompt_question + "\nA. " + option_a + "\nB. " + option_b + "\nC. " + option_c + "\nD. " + option_d}
                ]
                llm_instance = get_llm()
                response = generate_text(chat_input, llm_instance)
                try:
                    result = json.loads(response)
                    response_answer = result["answer"].upper().strip()
                    reasoning = result["reasoning"].strip()
                    correct_answer = item["correct_answer"]
                    
                    # Store data in JSON format
                    new_data[i] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "persona_description": new_persona, "correct_answer": correct_answer, "persona_answer": response_answer, "reasoning": reasoning, "country": item["country"]}
                    # Normalize to strings for comparison
                    if response_answer.upper() == correct_answer.upper():
                        correct += 1
                    total += 1
                except:
                    print("Error parsing response: " + response)
                    continue

            write_to_file(mode, cur_iteration, difficulty, correct, total, new_data)
    else:
        for cur_iteration in range(2, num_iterations + 1):
            prev_iteration = cur_iteration - 1
            correct = total = 0
            new_data = {}
            with open(f"../results/{mode[-2:]}/{mode[:-3]}/i{prev_iteration}/persona_Hard.jsonl", "r") as f:
                lines = f.readlines()
                data = [json.loads(line) for line in lines[:-1]]
            for i in range(0, len(data), 4):
                prompt_question = data[i]["question"]
                qa_responses = ""
                for j in range(4):
                    qa_responses += data[i + j]["prompt_option"] + ": " + data[i + j]["persona_answer"] + "\n" + "reasoning: " + data[i + j]["reasoning"] + "\n\n"
                new_persona = generate_new_persona(difficulty, prompt_question, qa_responses, data[i]["persona_description"], mode, data[i]["country"], cur_iteration)
                isCorrect = True
                isError = False
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
                                "\"correct\" (either \"true\" or \"false\") and \"reasoning\" (a short explanation). "
                                "Example format: {\"correct\": \"true\", \"reasoning\": \"The answer is true because ...\"}"
                                f"\nQuestion: {prompt_question}\nAnswer: {prompt_option}"
                            ),
                        },
                    ]
                    llm_instance = get_llm()
                    response = generate_text(chat_input, llm_instance)
                    try:
                        result = json.loads(response)
                        thinks_correct = "true" if "true" in result["correct"].lower().strip() else "false"
                        reasoning = result["reasoning"].strip()
                    except json.JSONDecodeError:
                        # Fallback if model didn't return valid JSON
                        response_lower = response.lower().strip()
                        thinks_correct = "true" if "true" in response_lower else "false"
                        reasoning = response_lower.strip()
                    # if there's an error, disregard this question (set of 4 options)
                    except:
                        isError = True
                        print("Error parsing response: " + response)
                        break

                    # store data in JSON format
                    new_data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "persona_description": new_persona, "correct_answer": correct_answer, "persona_answer": thinks_correct, "reasoning": reasoning, "country": data[i + j]["country"]}
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

            write_to_file(mode, cur_iteration, difficulty, correct, total, new_data)
                
def cleanup():
    """Clean up GPU memory and close LLM instance"""
    global llm
    if llm is not None:
        print("Cleaning up GPU memory for LLM instance...")
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared")
            del llm
            llm = None
            print("LLM instance deleted")
            gc.collect()
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Run initial evaluation and iterations")
        parser.add_argument("--mode", type=str, required=True, help="Mode to run (e.g., ling_p2, eng_p1)")
        parser.add_argument("--num_iterations", type=int, required=True, help="Total number of iterations including initial evaluation")
        parser.add_argument("--difficulty", type=str, required=True, choices=["easy", "hard", "Easy", "Hard"], help="Difficulty level")
        args = parser.parse_args()
        
        difficulty = args.difficulty.capitalize()
        
        run_initial_eval(difficulty, args.mode)
        
        if args.num_iterations > 1:
            run_iterations(args.mode, args.num_iterations, difficulty)
        else:
            print("\nNo additional iterations to run (num_iterations = 1)")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()