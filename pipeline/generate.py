import random
import json
import sys
from vllm import LLM, SamplingParams
import torch
import gc
import os
from utils.configs import configs

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

languages = ['afar', 'arabic', 'balochi', 'chinese', 'english', 'faroese', 'fijian', 'german', 'hebrew', 'hiligaynon', 'hindi', 'hungarian', 'japanese', 'kirundi', 'korean', 'papiamento', 'pashto', 'russian', 'samoan', 'spanish', 'tongan', 'tswana', 'wolof']

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM(model=model_name, tensor_parallel_size=4, dtype='half')

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

def generate_text(chat_input, llm, sampling_params):
    output = llm.chat(chat_input, sampling_params)
    return output[0].outputs[0].text
# automatic pipeline to add persona answer to test set
def add_persona(config_name, type, data):
    try:
        # generate persona for question
        for i in range(len(data)):
            curQuestion = data[i]["question"]
            chat_input = [
                {"role": "system",
                "content": configs[config_name]["system_prompt"].format(language=data[i]["language"])},
                {"role": "user",
                "content": curQuestion}
            ]
            persona = generate_text(chat_input, llm, sampling_params)
            
            # Parse JSON response to extract persona text
            try:
                persona_json = json.loads(persona)
                persona_text = persona_json.get("persona", persona)  # Fallback to original if parsing fails
            except (json.JSONDecodeError, KeyError):
                # If JSON parsing fails, try to extract persona from the response
                persona_text = persona.strip()
                # Remove common prefixes like "Sure! here is your persona: "
                if persona_text.startswith("Sure!") or persona_text.startswith("Here is your persona:"):
                    persona_text = persona_text.split(":", 1)[-1].strip()
            
            data[i]["persona"] = persona_text

            # Since persona_text is now in English, we need to ensure the model responds in the target language
            # Add explicit instruction to respond in the target language
            language_instruction = f"IMPORTANT: You must respond to the question in {data[i]['language']}. "
            
            content = f"{persona_text}. {language_instruction}Answer the question in a way that is consistent with the knowledge, style, and perspective of the persona, but do not speak as the persona or use first-person language."
            # generate persona-model answer
            chat_input = [
                {"role": "system",
                "content": content},
                {"role": "user",
                "content": f"Question: {curQuestion}"}
            ]
            persona_model_answer = generate_text(chat_input, llm, sampling_params)
            data[i]["persona_model_answer"] = persona_model_answer
        
        with open(f"personaData/{type}/{config_name}.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Persona added to {type}/{config_name} successfully!") 
    except Exception as e:
        print(f"Error adding persona to {type}: {e}")

def sample_questions(config_name, type, n_questions, n_languages):
    sample_questions = []
    ground_truths = []
    
    print("Loading questions from datasets...")
    for cur_lang in languages[:n_languages]:
        json_file = f"CaLMQA/data/datasets/dataset-{type}-{cur_lang}.json"
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            random_entries = random.sample(data["entries"], n_questions)
            for random_entry in random_entries:
                question = random_entry["question"]
                sample_questions.append(question["translations"][cur_lang.capitalize()]["text"])
                try:
                    if type == "agnostic":
                        ground_truths.append(question["translations"][cur_lang.capitalize()]["prompting_state"]["prompt"])
                    else:
                        ground_truths.append(question["translations"]["English"]["prompting_state"]["prompt"])
                except Exception as e:
                    print(f"Error getting ground truth for {question['translations'][cur_lang.capitalize()]['text']}: {e}")
                    ground_truths.append("None")
        except FileNotFoundError:
            print(f"Warning: Could not find {json_file}")
            sample_questions.append(f"Sample question in {cur_lang}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            sample_questions.append(f"Sample question in {cur_lang}")
    
    model_outputs = []
    for i in range(len(sample_questions)):
        question = sample_questions[i]
        ground_truth = ground_truths[i]
        try:
            chat_input = [
                {"role": "system",
                "content": f"You are a helpful assistant that responds in {languages[i // n_questions]}."},
                {"role": "user",
                "content": question}
            ]
            model_answer = generate_text(chat_input, llm, sampling_params)
            model_outputs.append({
                "question": question,
                "model_answer": model_answer,
                "language": languages[i // n_questions],
                "old_ground_truth": ground_truth
            })
        except Exception as e:
            print(f"Error generating response for question: {e}")
    
    add_persona(config_name, type, model_outputs)

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
        n_questions = int(sys.argv[1])
        n_languages = int(sys.argv[2]) if int(sys.argv[2]) != -1 else len(languages)
        
        print(f"Generating {n_questions} questions for {n_languages} languages")

        for config_name in configs:
            sample_questions(config_name, "agnostic", n_questions, n_languages)
            sample_questions(config_name, "specific", n_questions, n_languages)
    finally:
        cleanup()
        print("Cleanup completed for LLM generate instance!")