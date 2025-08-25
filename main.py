from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
import os
import torch
import gc
import sys
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM(model=model_name, tensor_parallel_size=4, dtype='half')
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

def generate_text(chat_input, llm):
    output = llm.chat(chat_input, sampling_params)
    return output[0].outputs[0].text

def generate_persona_description(question, country):   
    chat_input = [
        {"role": "system",
        "content": """You are an expert in crafting personas that will be used as a system prompt to a language model in answering a given question accurately and completely in a direct and concise manner. You will be given the question and the relevant country that the question content is related to. \n
        Your task is to generate a persona that MUST:
        1. Have RELEVANT expertise that directly relates to the question content
        2. Provide cultural/linguistic context that enhances understanding
        3. Guide the model toward more accurate, complete, and helpful responses
        4. Not include any other text other than the persona description
        5. Be in English
        6. BE IN SECOND PERSON; in other words, DO NOT include fictional characters, names, or any other non-real entities. You are addressing the language model directly.

        # IMPORTANT #
        Always start with "You are" ...
        
        """},
        {"role": "user",
        "content": "question: " + question + "\n\n" + "country: " + country}
    ]
    response = generate_text(chat_input, llm)
    return response

def run_eval(type, difficulty):
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
                    persona_description = generate_persona_description(prompt_question, country)
                prev_prompt_question = prompt_question
                if type == "persona":
                    system_prompt = f"{persona_description}\n\nYou will be given a question and an answer. Output \"true\" if the answer is correct, otherwise output \"false\". Do not output anything else. Structure your output as follows: {{\"answer\": \"true\" or \"false\"}}"
                else:
                    system_prompt = "You are a helpful assistant. You are a given a question and an answer. Output \"true\" if the answer is correct, otherwise output \"false\". Do not output anything else. Structure your output as follows: {{\"answer\": \"true\" or \"false\"}}"
                chat_input = [
                    {"role": "system",
                    "content": system_prompt},
                    {"role": "user",
                    "content": "question: " + prompt_question + "\n\n" + "answer: " + prompt_option}
                ]
                response = generate_text(chat_input, llm)

                try:
                    response_json = json.loads(response)
                    # Store data in JSON format
                    if type == "persona":
                        data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "persona_description": persona_description, "correct_answer": prompt_answer, "persona_answer": response_json["answer"]}
                    else:
                        data[i + j] = {"question": prompt_question, "prompt_option": prompt_option, "correct_answer": prompt_answer, "vanilla_answer": response_json["answer"]}

                    # Normalize to strings for comparison
                    if str(response_json["answer"]).lower() == str(prompt_answer).lower():
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
                persona_description = generate_persona_description(prompt_question, country)
                system_prompt = f"{persona_description}\n\nYou will be given a question and an answer. Output the letter of the correct answer in string format. Do not output anything else. Structure your output as follows: {{\"answer\": \"A\", \"B\", \"C\", or \"D\"}}"
            else:
                system_prompt = "You are a helpful assistant. You are a given a question and an answer. Output the letter of the correct answer in string format. Do not output anything else. Structure your output as follows: {{\"answer\": \"A\", \"B\", \"C\", or \"D\"}}"
            
            chat_input = [
                {"role": "system",
                "content": system_prompt},
                {"role": "user",
                "content": "question: " + prompt_question + "\n\n" + "A: " + option_a + "\n\n" + "B: " + option_b + "\n\n" + "C: " + option_c + "\n\n" + "D: " + option_d}
            ]
            response = generate_text(chat_input, llm)
            response = re.sub(r'{"answer":\s*([A-D])}', r'{"answer": "\1"}', response)
            try:
                response_json = json.loads(response)
                # Store data in JSON format
                if type == "persona":
                    data[i + j] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "persona_description": persona_description, "correct_answer": answer, "persona_answer": response_json["answer"]}
                else:
                    data[i + j] = {"question": prompt_question, "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d}, "correct_answer": answer, "vanilla_answer": response_json["answer"]}
                # Normalize to strings for comparison
                if response_json["answer"].upper() == answer.upper():
                    correct += 1
                total += 1
            except:
                print("Error parsing response: " + response)
                continue
    
    with open(f"data/{type}_data_{difficulty}.jsonl", "w") as f:
        for entry in data.values():
            f.write(json.dumps(entry) + "\n")
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

        options = {
            "all": [
                lambda: run_eval("vanilla", "Hard"),
                lambda: run_eval("persona", "Hard"),
                lambda: run_eval("vanilla", "Easy"),
                lambda: run_eval("persona", "Easy"),
            ],
            "hard": [
                lambda: run_eval("vanilla", "Hard"),
                lambda: run_eval("persona", "Hard"),
            ],
            "easy": [
                lambda: run_eval("vanilla", "Easy"),
                lambda: run_eval("persona", "Easy"),
            ],
            "vanilla": [
                lambda: run_eval("vanilla", "Hard"),
                lambda: run_eval("vanilla", "Easy"),
            ],
            "persona": [
                lambda: run_eval("persona", "Hard"),
                lambda: run_eval("persona", "Easy"),
            ],
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