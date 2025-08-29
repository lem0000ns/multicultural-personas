from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
import os
import torch
import gc
import sys
import re
from utils import country_to_language
from configs import system_prompts

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM(model=model_name, tensor_parallel_size=4, dtype='half')

# Create separate sampling params for different question types
mcq_sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1)
tf_sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=2)
persona_sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

def generate_text(chat_input, llm, sampling_params):
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
    response = generate_text(chat_input, llm, persona_sampling_params)
    return response

def run_eval(type, difficulty, mode):
    ds = load_dataset("kellycyy/CulturalBench", f"CulturalBench-{difficulty}", split="test")

    data = {}
    correct = total = 0

    # CulturalBench-Hard
    if difficulty == "Hard":
        prev_prompt_question = persona_description = ""
        for i in range(0, len(ds), 4):
            # iterate over one question at a time (4 options)
            isCorrect = noError = True
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
                    {"role": "system",
                    "content": system_prompt},
                    {"role": "user",
                    "content": "Is this answer true or false for this question? You must choose either True or False.\nQuestion: " + prompt_question + "\nAnswer: " + prompt_option}
                ]
                response = generate_text(chat_input, llm, tf_sampling_params)

                try:
                    # Check if response contains "true" or "false"
                    response_lower = response.lower().strip()
                    response_answer = "true" if "true" in response_lower else "false"
                    
                    # Store data in JSON format
                    if type == "persona":
                        data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "persona_description": persona_description, "correct_answer": prompt_answer, "persona_answer": response_answer, "country": country}
                    else:
                        data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "correct_answer": prompt_answer, "vanilla_answer": response_answer, "country": country}

                    # Normalize to strings for comparison
                    if str(response_answer).lower() == str(prompt_answer).lower():
                        continue
                    else:
                        isCorrect = False
                        break
                except:
                    noError = False
                    print("Error parsing response: " + response)
                    continue
            if not noError:
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
                "content": "Instruction: You must select one option among A,B,C,D. Do not output any other things.\nQuestion: " + prompt_question + "\nA. " + option_a + "\nB. " + option_b + "\nC. " + option_c + "\nD. " + option_d}
            ]
            response = generate_text(chat_input, llm, mcq_sampling_params)
            
            try:
                response_answer = response.upper().strip()
                
                # Store data in JSON format
                if type == "persona":
                    data[i] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "persona_description": persona_description, "correct_answer": answer, "persona_answer": response_answer, "country": country}
                else:
                    data[i] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "correct_answer": answer, "vanilla_answer": response_answer, "country": country}
                # Normalize to strings for comparison
                if response_answer.upper() == answer.upper():
                    correct += 1
                total += 1
            except:
                print("Error parsing response: " + response)
                continue
    
    with open(f"{mode}/{type}_{difficulty}.jsonl", "w") as f:
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

if __name__ == "__main__":
    try:
        run_type = sys.argv[1] if len(sys.argv) > 1 else "all"
        mode = sys.argv[2] if len(sys.argv) > 2 else "eng"

        options = {
            "all": [
                lambda: run_eval("persona", "Hard", mode),
                lambda: run_eval("persona", "Easy", mode),
            ],
            "hard": [
                lambda: run_eval("persona", "Hard", mode),
            ],
            "easy": [
                lambda: run_eval("persona", "Easy", mode),
            ],
            "vanilla": [
                lambda: run_eval("vanilla", "Hard", mode),
                lambda: run_eval("vanilla", "Easy", mode),
            ]
        }

        if run_type not in options:
            print("Invalid run type. Valid types are: " + ", ".join(options.keys()))
            exit(1)

        for func in options[run_type]:
            func()
    except Exception as e:
        print(e)
    finally:
        cleanup()
        exit(1)