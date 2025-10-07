from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
import os
import torch
import gc
from tools.utils import country_to_language, modes_list, modes_list_p1, modes_list_p2
from tools.configs import system_prompts
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM(model=model_name, tensor_parallel_size=2, dtype='half')

# Create separate sampling params for different question types
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

def generate_text(chat_input, llm):
    output = llm.chat(chat_input, sampling_params)
    return output[0].outputs[0].text

def generate_persona_description(question, country, mode):   
    chat_input = [
        {"role": "system",
        "content": system_prompts[mode].format(language={"English" if "eng" in mode else country_to_language[country]})
        },
        {"role": "user",
        "content": "question: " + question + "\n\n" + "country: " + country + "\n\npersona description: "}
    ]
    response = generate_text(chat_input, llm)
    return response

def run_eval(type, difficulty, mode):
    print(f"Running {type} evaluation for {difficulty} difficulty with {mode} mode")
    ds = load_dataset("kellycyy/CulturalBench", f"CulturalBench-{difficulty}", split="test")

    data = {}
    correct = total = 0

    # CulturalBench-Hard
    if difficulty == "Hard":
        prev_prompt_question = persona_description = ""
        for i in range(0, len(ds), 4):
            # iterate over one question at a time (4 options)
            isCorrect  = True
            isError = False
            for j in range(4):
                cur_row = ds[i + j]
                prompt_question = cur_row["prompt_question"]
                prompt_option = cur_row["prompt_option"]
                prompt_answer = cur_row["answer"]
                country = cur_row["country"]
                if prompt_option is None or prompt_answer is None or prompt_question is None or country is None:
                    continue
                # Use same persona description for same question (4 at a time)
                if type == "persona" and prev_prompt_question != prompt_question:
                    persona_description = generate_persona_description(prompt_question, country, mode)
                prev_prompt_question = prompt_question
                if type == "persona":
                    system_prompt = persona_description
                else:
                    system_prompt = "You are a helpful assistant."
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
                response = generate_text(chat_input, llm)

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

                # Store data in JSON format
                if type == "persona":
                    data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "persona_description": persona_description, "correct_answer": prompt_answer, "persona_answer": thinks_correct, "reasoning": reasoning, "country": country}
                else:
                    data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "correct_answer": prompt_answer, "vanilla_answer": thinks_correct, "reasoning": reasoning, "country": country}
                # Normalize to strings for comparison
                if str(thinks_correct).lower() == str(prompt_answer).lower():
                    continue
                else:
                    isCorrect = False
                    break

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
            if type == "persona":
                persona_description = generate_persona_description(prompt_question, country, mode)
                system_prompt = persona_description
            else:
                system_prompt = "You are a helpful assistant."
            
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
            response = generate_text(chat_input, llm)
            
            try:
                result = json.loads(response)
                response_answer = result["answer"].upper().strip()
                reasoning = result["reasoning"].strip()
                
                # Store data in JSON format
                if type == "persona":
                    data[i] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "persona_description": persona_description, "correct_answer": answer, "persona_answer": response_answer, "reasoning": reasoning, "country": country}
                else:
                    data[i] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "correct_answer": answer, "vanilla_answer": response_answer, "reasoning": reasoning, "country": country}
                # Normalize to strings for comparison
                if response_answer.upper() == answer.upper():
                    correct += 1
                total += 1
            except:
                print("Error parsing response: " + response)
                continue
    # places results according to prompt number (last 2 characters)
    if type == "vanilla":
        file_name = f"../results/vanilla/vanilla_{difficulty}.jsonl"
    else:
        file_name = f"../results/{mode[-2:]}/{mode[:-3]}/{type}_{difficulty}.jsonl"
    with open(file_name, "w") as f:
        for entry in data.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.write(f"{type.capitalize()} Accuracy for {difficulty}: {correct / total}\n")
    
    print(f"{type.capitalize()} Accuracy for {difficulty}: {correct / total}")

def cleanup():
    """Clean up GPU memory and close LLM instance"""
    print("Cleaning up GPU memory for LLM generate instance...")
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
        
        global llm
        # Close the LLM instance
        if llm is not None:
            del llm
            llm = None
            print("LLM instance deleted")
        
        # Force garbage collection
        gc.collect()
            
    except Exception as e:
        print(f"Error during cleanup: {e}")

def diff_eval_clf(difficulty, mode):
    if difficulty == "hard":
        run_eval("persona", "Hard", mode)
    elif difficulty == "easy":
        run_eval("persona", "Easy", mode)
    elif difficulty == "all":
        run_eval("persona", "Hard", mode)
        run_eval("persona", "Easy", mode)
    elif difficulty == "vanilla":
        run_eval("vanilla", "Hard", "vanilla")
        run_eval("vanilla", "Easy", "vanilla")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, choices=["p1", "p2", "all"])
    parser.add_argument("--mode", type=str, nargs="+", choices=["eng_p1", "eng_p2", "ling_p1", "ling_p2", "all"])
    parser.add_argument("--difficulty", type=str, nargs="+", choices=["hard", "easy", "all", "vanilla"])
    args = parser.parse_args()
    
    # Check if at least one argument is provided
    if not any([args.prompt, args.mode, args.difficulty]):
        print("Please specify at least one argument: --prompt, --mode, or --difficulty")
        exit(1)
    
    try:
        # Determine which modes to run
        cur_modes = []
        
        # If mode is specified, use only that mode (or all if "all")
        if args.mode and args.mode != "all":
            cur_modes = args.mode
        elif args.mode == "all":
            cur_modes = modes_list

        # If prompt is specified, determine modes based on prompt
        if args.prompt == "p1" or args.prompt == "all":
            for mode in modes_list_p1:
                if mode not in cur_modes:
                    cur_modes.append(mode)
        if args.prompt == "p2" or args.prompt == "all":
            for mode in modes_list_p2:
                if mode not in cur_modes:
                    cur_modes.append(mode)
    
        if not args.prompt and not args.mode:
            cur_modes = modes_list
        
        # Determine which difficulties to run
        difficulties = ["hard", "easy"]
        if args.difficulty == "vanilla":
            difficulties = ["vanilla"]
        elif args.difficulty and args.difficulty != "all":
            difficulties = args.difficulty
        
        print("*" * 100)
        print(cur_modes)
        print(difficulties)
        print("*" * 100)
        # Run evaluations
        for mode in cur_modes:
            for difficulty in difficulties:
                diff_eval_clf(difficulty, mode)
                    
    except Exception as e:
        print(e)
    finally:
        cleanup()
        exit(1)