"""LLM utilities for model initialization and text generation."""

import os
import torch
import gc
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
TEMPERATURE = 0.0   

# Global LLM instance
llm = None
external_llm = None

# Pre-generated feedback storage: dict keyed by (difficulty, question_index) -> feedback
pre_generated_feedback = None

def mistral_7b_generate(llm_instance, messages, max_tokens=1024, enable_thinking_bool=False):
    SAMPLING_PARAMS = SamplingParams(temperature=TEMPERATURE, top_p=0.95, max_tokens=max_tokens)
    output = llm_instance.chat(messages, SAMPLING_PARAMS)
    return None, output[0].outputs[0].text

def qwen3_4b_generate_thinking(llm_instance, messages, enable_thinking_bool=False):
    """Generate thinking and content from Qwen3"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, 
        enable_thinking=enable_thinking_bool
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(llm_instance.device)
    
    do_sample_bool = True if TEMPERATURE > 0.0 else False

    # conduct text completion
    generated_ids = llm_instance.generate(
        **model_inputs,
        max_new_tokens=2048,  # Reduced from 32768 to prevent OOM
        temperature=TEMPERATURE,
        top_p=0.95,
        do_sample=do_sample_bool,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_decoded = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    # Only use text inside <think> and </think> tags, if present
    import re
    think_match = re.search(r"<think>(.*?)</think>", thinking_decoded, re.DOTALL)
    thinking_content = think_match.group(1).strip() if think_match else ""
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # Clear GPU cache after generation to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return thinking_content, content

def llama_3_8b_instruct_generate(llm_instance, messages, max_tokens=1024, enable_thinking_bool=False):
    """Generate text from chat input using the LLM.
    
    Args:
        llm_instance: The LLM instance to use for generation
        messages: Chat messages in format [{"role": "system/user", "content": "..."}]
    
    Returns:
        Generated text string
    """
    SAMPLING_PARAMS = SamplingParams(temperature=TEMPERATURE, top_p=0.95, max_tokens=max_tokens)
    output = llm_instance.chat(messages, SAMPLING_PARAMS)
    return None, output[0].outputs[0].text

generate_text_funcs = {
   "Qwen/Qwen3-4B": qwen3_4b_generate_thinking,
   "meta-llama/Meta-Llama-3-8B-Instruct": llama_3_8b_instruct_generate,
   "mistralai/Mistral-7B-Instruct-v0.3": mistral_7b_generate,
}


def get_llm():
    """Get or initialize the primary LLM instance."""
    if external_llm is not None:
        cleanup()
    global llm
    if llm is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if MODEL_NAME == "Qwen/Qwen3-4B":
            # Standard Transformers loading
            llm = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype="auto",
                device_map="auto"
            )
            llm.eval()
        elif MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
            # vLLM loading with restricted memory reservation (uses GPUs 0,1)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Cannot initialize vLLM.")
            
            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {num_gpus}")
            
            llm = LLM(
                model=MODEL_NAME, 
                tensor_parallel_size=4, 
                dtype='half', 
                gpu_memory_utilization=0.50, 
                max_model_len=4096,          
                enforce_eager=False          
            )
    return llm

def get_external_llm():
    """Get or initialize the external LLM instance for feedback generation."""
    global external_llm

    if llm is not None:
        cleanup()
    
    if external_llm is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot initialize external LLM.")
        
        print("[External LLM] Initializing external LLM (Mistral-7B-Instruct)...")
        external_llm = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.3", 
            tensor_parallel_size=4, 
            gpu_memory_utilization=0.20, 
            max_model_len=512,
            enforce_eager=True,         
            disable_custom_all_reduce=True
        )
        print("[External LLM] Successfully initialized")
    
    return external_llm

def pre_generate_all_feedback(data, difficulty):
    """Pre-generate all feedback for a list of questions using external LLM.
    
    Args:
        data: List of question data dictionaries. Each dict should have:
            - 'question': question text
            - 'persona_description' or 'pretranslated_persona': persona text
            - 'model_answer': model answer (for Easy mode only)
        difficulty: "Easy" or "Hard"
    
    Returns:
        Dictionary mapping question index to feedback string
    """
    global pre_generated_feedback
    from tools.configs import feedback_prompt_easy, feedback_prompt_hard
    from vllm import SamplingParams

    feedback_dict = {}
    
    external_llm_instance = get_external_llm()
    SAMPLING_PARAMS = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024)
    
    for i, item in enumerate(data):
        question = item["question"]
        persona = item.get("persona_description") or item.get("pretranslated_persona", "")
        model_answer = item.get("model_answer") if difficulty == "Easy" else None
        
        if difficulty == "Easy":
            feedback_prompt = feedback_prompt_easy
            user_content = f"Question: {question}\n\nPersona: {persona}\n\nPredicted answer: {model_answer}"
        else:
            feedback_prompt = feedback_prompt_hard
            user_content = f"Question: {question}\n\nPersona: {persona}"
        
        chat_input = [
            { "role": "system", "content": feedback_prompt },
            { "role": "user", "content": user_content }
        ]
        
        try:
            output = external_llm_instance.chat(chat_input, SAMPLING_PARAMS)
            feedback = output[0].outputs[0].text
            feedback_dict[i] = feedback
        except Exception as e:
            feedback_dict[i] = ""
    
    pre_generated_feedback = feedback_dict
    return feedback_dict

def cleanup():
    """Clean up GPU memory by deleting instances and clearing cache."""
    import time
    import subprocess
    global llm, external_llm, pre_generated_feedback
    print("Cleaning up GPU memory...")
    
    # Properly shutdown vLLM engines before deleting
    if llm is not None:
        try:
            # vLLM has a shutdown method that properly terminates child processes
            if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'shutdown'):
                llm.llm_engine.shutdown()
        except Exception as e:
            print(f"Warning: Error shutting down main LLM engine: {e}")
        del llm
        llm = None
    
    if external_llm is not None:
        try:
            if hasattr(external_llm, 'llm_engine') and hasattr(external_llm.llm_engine, 'shutdown'):
                external_llm.llm_engine.shutdown()
        except Exception as e:
            print(f"Warning: Error shutting down external LLM engine: {e}")
        del external_llm
        external_llm = None
    
    # Wait a bit for processes to fully terminate
    time.sleep(2)
    
    # Kill any remaining vLLM processes (EngineCore, Worker processes)
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'EngineCore|Worker_TP'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"Found {len(pids)} remaining vLLM processes, killing them...")
            for pid in pids:
                try:
                    subprocess.run(['kill', '-9', pid], check=False)
                except:
                    pass
            time.sleep(1)
    except Exception as e:
        print(f"Warning: Error killing vLLM processes: {e}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()
    gc.collect()
