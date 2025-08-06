import json
from vllm import LLM, SamplingParams
import torch
import gc
import weave

weave.init("mc-personas-judge")
llm = LLM(model='Qwen/Qwen2.5-32B-Instruct', tensor_parallel_size=8, dtype='half')

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

JUDGE_PROMPT = """Judge whether the following response to a question is roughly correct or not based on the precise and unambiguous correct answer provided. Respond in English.

Your judgement must be in the format and criteria specified below:

reasoning: Explain why the response is correct or incorrect based on the correct_answer, focusing only on if there are meaningful differences between the correct_answer and the response. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than the correct_answer, focus only on whether the answers match. A response is roughly correct if it captures the main idea or meaning of the correct answer, even if it is incomplete, imprecise, or uses different wording. Do not provide background information, do not solve the problem yourself, and do not suggest alternative answers.

Be concise and objective. Your goal is to assess whether the response aligns with the correct_answer in substance.

correct: Answer 'yes' if the response matches the correct_answer given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the response is incorrect.

confidence: The extracted confidence score between 0% and 100% from the response. Put 100 if there is no confidence score available."""

JUDGE_EYE_TEST = """Determine which of the following two responses better answers the question, based on detail, accuracy, and completeness. Focus only on how well each response addresses the substance of the question.

Your judgement must be in the format and criteria specified below:

verdict: Answer "Answer 1" if the first response better answers the question or "Answer 2" if the second response better answers the question.

reasoning: In English, explain why the chosen response is better, focusing on accuracy, relevance, clarity, and completeness of the answer. Do not solve the question yourself or suggest alternative answers.

confidence: The extracted confidence score between 0% and 100% from the response. Put 100 if there is no confidence score available"""

def cleanup():
    """Clean up GPU memory and close LLM instance"""
    print("Cleaning up GPU memory for LLM judge instance...")
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

def generate_text(chat_input, llm, sampling_params):
    output = llm.chat(chat_input, sampling_params)
    return output[0].outputs[0].text

def parse_response_eye_test(response):
    comparison = {
        "verdict": "",
        "reasoning": "",
        "confidence": "",
    }
    try:
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("verdict:"):
                comparison["verdict"] = line.replace("verdict:", "").strip()
            elif line.startswith("reasoning:"):
                comparison["reasoning"] = line.replace("reasoning:", "").strip()
            elif line.startswith("confidence:"):
                comparison["confidence"] = line.replace("confidence:", "").strip()
        return comparison
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None

def parse_response(response):
    """
    Parse the response to extract reasoning, correct, and confidence fields
    Returns a dictionary with reasoning, correct, and confidence fields
    """
    comparison = {
        "correct": "",
        "reasoning": "",
        "confidence": "",
    }
    
    try:
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("reasoning:"):
                comparison["reasoning"] = line.replace("reasoning:", "").strip()
            elif line.startswith("correct:"):
                comparison["correct"] = line.replace("correct:", "").strip()
            elif line.startswith("confidence:"):
                comparison["confidence"] = line.replace("confidence:", "").strip()
    except Exception as e:
        print(f"Error parsing response: {e}")
        # Fallback: store the raw response
        comparison["response"] = response
    
    return comparison

# automatic pipeline to compare answers
@weave.op()
def compare_answers(type):
    with open(f"personaData/{type}-pj.json", "r") as f:
        data = json.load(f)
    try:
        for i in range(len(data)):
            question = data[i]["question"]
            model_answer = data[i]["model_answer"]
            persona_model_answer = data[i]["persona_model_answer"]

            # IF NO GROUND TRUTH AVAILABLE
            if data[i]["ground_truth"] == "None":

                # HANDLING ANY SURFACE LEVEL ISSUES
                if data[i]["vanilla_issue"] and data[i]["persona_issue"]:
                    data[i]["comparison"] = {
                        "verdict": "None",
                        "reasoning": "Both responses have surface level issues.",
                        "confidence": "100",
                    }
                    continue
                elif data[i]["vanilla_issue"]:
                    data[i]["comparison"] = {
                        "verdict": "Answer 2",
                        "reasoning": "The vanilla response has surface level issues.",
                        "confidence": "100",
                    }
                    continue
                elif data[i]["persona_issue"]:
                    data[i]["comparison"] = {
                        "verdict": "Answer 1",
                        "reasoning": "The persona response has surface level issues.",
                        "confidence": "100",
                    }
                    continue

                # PERFORMING EYE TEST
                user_prompt = f"""Question: {question}\n\nAnswer 1: {model_answer}\n\nAnswer 2: {persona_model_answer}"""
                chat_input = [
                    {"role": "system", "content": JUDGE_EYE_TEST},
                    {"role": "user", "content": "prompt: " + user_prompt}
                ]
                response = generate_text(chat_input, llm, sampling_params)
                comparison = parse_response_eye_test(response)
                data[i]["comparison"] = comparison

            # IF GROUND TRUTH AVAILABLE, COMPARE TO GROUND TRUTH
            else:

                # VANILLA COMPARISON
                if not data[i]["vanilla_issue"]:
                    user_prompt = f"""Question: {question}\n\nResponse: {model_answer}\n\nCorrect Answer: {data[i]["ground_truth"]}"""
                    chat_input = [
                        {"role": "system", "content": JUDGE_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                    vanilla_response = generate_text(chat_input, llm, sampling_params)
                    comparison = parse_response(vanilla_response)
                    data[i]["vanilla_comparison"] = comparison
                else:
                    data[i]["vanilla_comparison"] = {
                        "correct": "no",
                        "reasoning": "The vanilla response has surface level issues.",
                        "confidence": "100",
                    }

                # PERSONA COMPARISON
                if not data[i]["persona_issue"]:
                    user_prompt = f"""Question: {question}\n\nResponse: {persona_model_answer}\n\nCorrect Answer: {data[i]["ground_truth"]}"""
                    chat_input = [
                        {"role": "system", "content": JUDGE_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                    persona_response = generate_text(chat_input, llm, sampling_params)
                    comparison = parse_response(persona_response)
                    data[i]["persona_comparison"] = comparison
                else:
                    data[i]["persona_comparison"] = {
                        "correct": "no",
                        "reasoning": "The persona response has surface level issues.",
                        "confidence": "100",
                    }

        with open(f"personaData/{type}-j.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Comparison added to {type} successfully!")
        return data
    except Exception as e:
        print(f"Error comparing answers for {type}: {e}")
        return None

if __name__ == "__main__":
    try:
        compare_answers("ag")
        compare_answers("sp")
    finally:
        cleanup()
        print("Cleanup done for LLM judge instance!")