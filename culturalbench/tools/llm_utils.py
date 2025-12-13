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
}

def get_llm():
    """Get or initialize the LLM instance."""
    global llm
    if llm is None:
        if MODEL_NAME == "Qwen/Qwen3-4B":
            llm = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,  # Optimize memory during loading
                use_cache=True,  # Enable KV cache for efficiency
            )
            # Move model to eval mode and optimize memory
            llm.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
            llm = LLM(model=MODEL_NAME, tensor_parallel_size=4, dtype='half')
    return llm


def cleanup():
    """Clean up GPU memory by deleting the LLM instance."""
    global llm
    if llm is not None:
        print("Cleaning up GPU memory for LLM instance...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared")
            del llm
            llm = None
            print("LLM instance deleted")
            gc.collect()
        except Exception as e:
            print(f"Error during cleanup: {e}")

