"""LLM utilities for model initialization and text generation."""

import os
import torch
import gc
from vllm import LLM, SamplingParams

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Global LLM instance
llm = None


def get_llm():
    """Get or initialize the LLM instance."""
    global llm
    if llm is None:
        llm = LLM(model=MODEL_NAME, tensor_parallel_size=2, dtype='half')
    return llm


def generate_text(chat_input, llm_instance, max_tokens=1024):
    """Generate text from chat input using the LLM.
    
    Args:
        chat_input: Chat messages in format [{"role": "system/user", "content": "..."}]
        llm_instance: The LLM instance to use for generation
    
    Returns:
        Generated text string
    """
    SAMPLING_PARAMS = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=max_tokens)
    output = llm_instance.chat(chat_input, SAMPLING_PARAMS)
    return output[0].outputs[0].text


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

