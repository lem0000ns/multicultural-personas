import random
import json
from vllm import LLM, SamplingParams
import torch
import gc

languages = ['afar', 'arabic', 'balochi', 'chinese', 'english']

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM(model=model_name, tensor_parallel_size=4, dtype='half')

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

def generate_text(chat_input, llm, sampling_params):
    output = llm.chat(chat_input, sampling_params)
    return output[0].outputs[0].text
# automatic pipeline to add persona answer to test set
def add_persona(type, data):
    try:
        # generate persona for question
        for i in range(len(data)):
            curQuestion = data[i]["question"]
            chat_input = [
                {"role": "system",
                "content": f"You are an expert in crafting concise and effective persona descriptions for language models. Your task is to generate a persona for a given question that will guide the model’s behavior. Write the persona in second person and focus exclusively on the model's background, expertise, and relevant traits that are directly useful for the task. Do not include names, fictional roles, or occupations. Limit the description to no more than 5 sentences and write it in the target language specified by {data[i]['language']}."},
                {"role": "user",
                "content": curQuestion}
            ]
            persona = generate_text(chat_input, llm, sampling_params)
            data[i]["persona"] = persona
            # generate persona-model answer
            chat_input = [
                {"role": "system",
                "content": f"{persona}. Answer the question using {data[i]['language']} in a way that is consistent with your persona."},
                {"role": "user",
                "content": f"Question: {curQuestion}"}
            ]
            persona_model_answer = generate_text(chat_input, llm, sampling_params)
            data[i]["persona_model_answer"] = persona_model_answer
        
        with open(f"personaData/{type}-persona-no-compare.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Persona added to {type} successfully!") 
    except Exception as e:
        print(f"Error adding persona to {type}: {e}")

def sample_questions(type, n_questions):
    print("Initializing LLM...")
    print("LLM initialized successfully!")
    
    sample_questions = []
    ground_truths = []
    
    print("Loading questions from datasets...")
    for cur_lang in languages:
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
                "content": "You are a helpful assistant."},
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
    
    add_persona(type, model_outputs)

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
        sample_questions("agnostic", 1)
        sample_questions("specific", 1)
    finally:
        cleanup()
        print("Cleanup completed for LLM generate instance!")