from vllm import LLM, SamplingParams
import json
import torch

llm = LLM(model='Qwen/Qwen2.5-32B-Instruct', tensor_parallel_size=4, dtype='half')

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

def generate_text(chat_input, llm, sampling_params):
    output = llm.chat(chat_input, sampling_params)
    return output[0].outputs[0].text

def cleanup():
    """Clean up GPU memory and close LLM instance"""
    print("Cleaning up GPU memory for LLM extractGT instance...")
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
    except Exception as e:
        print(f"Error during cleanup: {e}")

def extract_ground_truth(type):
    with open(f"personaData/{type}-pj.json", "r") as f:
        data = json.load(f)
    for entry in data:
        if entry["old_ground_truth"] == "None":
            entry["ground_truth"] = "None"
            continue
        old_ground_truth = entry["old_ground_truth"]
        temp_language1 = "English" if type == "agnostic" else entry["language"].capitalize()
        temp_language2 = "English" if type == "specific" else entry["language"].capitalize()
        chat_input = [
            {"role": "system",
            "content": f"""You are a text extraction assistant. Your only task is to extract the exact ground truth text from the given prompt. \n\n⚠️ Do not translate, rewrite, summarize, or modify any words.\n\n⚠️ Do not interpret the meaning of the text.\n\n⚠️ Output the ground truth exactly as it appears, character for character.\n\n⚠️ Oftentimes, the ground truth follows the phrase: \"Here is the {temp_language1} answer. Use it as the context to make the translation sound natural in the {temp_language2}.\". If there is no ground truth, output \"None\".\n\n
            EXAMPLE:
            Prompt: Your task is to translate a question from English into Samoan. You will be given the English answer as the context.\n\nHere is the English answer. Use it as the context to make the translation sound natural in the Samoan:\nA larger lens gathers more light. Period. A larger lens weighs more which requires a larger frame to carry it. A larger frame allows for a larger sensor. A larger sensor generates more data which makes it easier to perform noise reduction. In short, that much larger camera is also MUCH more capable than your cell phone.\n\nTranslate the following question from English into Samoan. Make it sound as natural as possible:\nWhy do we still use gigantic TV studio cameras when the same technology is now cell phone sized?\n

            OUTPUT:
            A larger lens gathers more light. Period. A larger lens weighs more which requires a larger frame to carry it. A larger frame allows for a larger sensor. A larger sensor generates more data which makes it easier to perform noise reduction. In short, that much larger camera is also MUCH more capable than your cell phone.
            
            """},
            {"role": "user",
            "content": f"Prompt: {old_ground_truth}"}
        ]
        ground_truth = generate_text(chat_input, llm, sampling_params)
        entry["ground_truth"] = ground_truth
    
    with open(f"personaData/{type}-pj.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    try:
        extract_ground_truth("ag")
        extract_ground_truth("sp")
    finally:
        cleanup()
        print("Cleanup completed for LLM extractGT instance!")