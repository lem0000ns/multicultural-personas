"""LLM utilities for model initialization and text generation."""

import asyncio
import os
import gc
from functools import partial
from openai import OpenAI
import time
from .configs import EXTERNAL_FEEDBACK_PROMPT_EASY, EXTERNAL_FEEDBACK_PROMPT_HARD

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
SGLANG_HOST = os.environ.get("SGLANG_HOST", "34.126.87.212")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
TEMPERATURE = 0.0

SGLANG_CHAT_MAX_TOKENS = 1024

MAX_CONCURRENT = 1  # 1 = serial; >1 for API/SGLang models
LOCAL_MODELS = set()  # HF models loaded in-process (GPU-bound); SGLang models are not local

STEERING_COEFFICIENT = None
STEERING_MODEL = "Qwen/Qwen3-32B"  

STEERING_AXIS_FILENAMES = {
    "Qwen/Qwen3-32B": "qwen-3-32b/assistant_axis.pt",
    "google/gemma-2-27b-it": "gemma-2-27b/assistant_axis.pt",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b/assistant_axis.pt",
}

# Global LLM instance
llm = None

# Lazy-loaded steering model + Assistant Axis (only when STEERING_COEFFICIENT is set)
_steering_model = None
_steering_tokenizer = None
_steering_axis = None
_steering_config = None
_steering_model_name = None  # which model is loaded (to detect change)

def _get_external_feedback_sync(difficulty, question, persona, model_answer, feedback_language=None, model="meta-llama/Meta-Llama-3-8B-Instruct"):
    user_content = f"Question: {question}\nPersona: {persona}"
    if model_answer is not None:
        user_content += f"\nModel Answer: {model_answer}"
    user_content += "\nFeedback:"
    if difficulty == "Easy":
        system_prompt = EXTERNAL_FEEDBACK_PROMPT_EASY
    else:
        system_prompt = EXTERNAL_FEEDBACK_PROMPT_HARD
    if feedback_language:
        system_prompt = system_prompt.rstrip() + f"\n\nYou must provide your feedback entirely in {feedback_language}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    _, response = generate_text_funcs[model](None, messages, max_tokens=1024, enable_thinking_bool=False)
    return response.strip()


async def get_external_feedback(difficulty, question, persona, model_answer, feedback_language=None):
    return await asyncio.to_thread(_get_external_feedback_sync, difficulty, question, persona, model_answer, feedback_language)


def llama_3_8b_instruct_generate(
    llm_instance, messages, max_tokens=SGLANG_CHAT_MAX_TOKENS, enable_thinking_bool=False, **kwargs
):
    """Generate text from chat input using the LLM.
    
    Args:
        llm_instance: The LLM instance to use for generation
        messages: Chat messages in format [{"role": "system/user", "content": "..."}]
    
    Returns:
        Generated text string
    """
    client = OpenAI(
        base_url=f"http://{SGLANG_HOST}:30000/v1",
        api_key="EMPTY",
    )
    for n_try in range(10):
        try:
            time.sleep(0.5)
            resp = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=messages,
                temperature=0.6,
                top_p=1,
                max_tokens=max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Exception: {e}")

            print("Sleep for 10 sec, then retry...")
            time.sleep(10)
    else:
        print("Error: Failed to generate response")
        content = ""

    return None, content

# SGLang :30002 default (see _MODEL_PORTS below; most models → 30002).
GEMMA3_12B_SGLANG_MODEL_ID = "google/gemma-3-12b-it"
GEMMA3_12B_SGLANG_API_MODEL = os.environ.get(
    "GEMMA3_12B_SGLANG_API_MODEL",
    "google/gemma-3-12b-it",
)
# Legacy CLI ids for the same port-30002 slot; still accepted.
LEGACY_QWEN3_06B_SGLANG_MODEL_ID = "Qwen/Qwen3-0.6B"
LEGACY_MISTRAL_SGLANG_MODEL_ID = "mistral-3-14b-instruct-2512"


def _use_sglang_text_part_messages(model: str) -> bool:
    """Use OpenAI-style text parts [{\"type\":\"text\",\"text\":...}] for chat content.

    Gemma (and some other templates on SGLang) expect this shape. Override with env:
      SGLANG_USE_TEXT_PART_MESSAGES=1  -> always convert for SGLang calls here
      SGLANG_USE_TEXT_PART_MESSAGES=0  -> never convert (plain string content)
      unset -> convert only for google/ models (default)
    """
    v = os.environ.get("SGLANG_USE_TEXT_PART_MESSAGES", "").strip().lower()
    if v in ("1", "true", "yes", "all"):
        return True
    if v in ("0", "false", "no"):
        return False
    return model.startswith("google/")


def _normalize_messages_text_parts(messages, model: str):
    """Convert string message content to [{\"type\": \"text\", \"text\": \"...\"}]."""
    if not messages or not _use_sglang_text_part_messages(model):
        return messages
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        c = m.get("content")
        if isinstance(c, str):
            out.append({**m, "content": [{"type": "text", "text": c}]})
        else:
            out.append(m)
    return out


def qwen_3_sglang_generate(
    llm_instance=None,
    messages=None,
    max_tokens=SGLANG_CHAT_MAX_TOKENS,
    enable_thinking_bool=False,
    model=GEMMA3_12B_SGLANG_API_MODEL,
    **kwargs,
):
    """
    Get response from SGLang server using OpenAI-compatible API.
    Returns (thinking_content, response) to match other generate_text_funcs.
    """
    _MODEL_PORTS = {
        "Qwen/Qwen3-14B": 30001,
        "zai-org/GLM-4-9B-0414": 30003,
    }
    _port = _MODEL_PORTS.get(model, 30002)
    client = OpenAI(
        base_url=f"http://{SGLANG_HOST}:{_port}/v1",
        api_key="EMPTY",
    )
    thinking_content = None
    content = None
    api_messages = _normalize_messages_text_parts(messages, model)
    for n_try in range(10):
        try:
            time.sleep(0.5)
            create_kwargs = dict(
                model=model,
                messages=api_messages,
                temperature=0.6,
                top_p=1,
                max_tokens=max_tokens,
            )
            if model.startswith("Qwen"):
                create_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            resp = client.chat.completions.create(**create_kwargs)
            content = (resp.choices[0].message.content or "").strip()
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Exception: {e}")
            print("Sleep for 10 sec, then retry...")
            time.sleep(10)
    else:
        print("Error: Failed to generate response")

    if content is None:
        return None, ""
    if content.startswith("<think>"):
        i = content.find("</think>")
        if i != -1:
            thinking_content, content = content[7:i].strip(), content[i + 8:].strip()
    if not enable_thinking_bool:
        return None, content
    return (thinking_content, content)


def _get_steering_model_and_axis():
    """Lazy-load steering model, tokenizer, and Assistant Axis. Uses STEERING_MODEL and STEERING_AXIS_FILENAMES."""
    global _steering_model, _steering_tokenizer, _steering_axis, _steering_config, _steering_model_name
    model_name = STEERING_MODEL
    if _steering_model is not None and _steering_model_name == model_name:
        return _steering_model, _steering_tokenizer, _steering_axis, _steering_config
    if _steering_model is not None:
        # model changed; clear so we reload
        _steering_model = None
        _steering_tokenizer = None
        _steering_axis = None
        _steering_config = None
    try:
        from assistant_axis import get_config, load_axis
    except ImportError as e:
        raise ImportError(
            "Assistant-axis steering requires the assistant_axis package. "
            "Install with: pip install assistant-axis"
        ) from e
    from huggingface_hub import hf_hub_download

    axis_filename = STEERING_AXIS_FILENAMES.get(model_name)
    if axis_filename is None:
        raise ValueError(
            f"Unknown steering model: {model_name}. Supported: {list(STEERING_AXIS_FILENAMES.keys())}"
        )
    print(f"Loading {model_name} for Assistant Axis steering...")
    _steering_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    _steering_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _steering_model.eval()
    _steering_config = get_config(model_name)
    axis_path = hf_hub_download(
        repo_id="lu-christina/assistant-axis-vectors",
        filename=axis_filename,
        repo_type="dataset",
    )
    _steering_axis = load_axis(axis_path)
    _steering_model_name = model_name
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _steering_model, _steering_tokenizer, _steering_axis, _steering_config


def qwen3_32b_steering_generate(
    llm_instance,
    messages,
    max_tokens=8192,
    enable_thinking_bool=False,
    use_steering=True,
):
    """
    Generate with steering model (STEERING_MODEL) and Assistant Axis.
    use_steering=False: run unsteered (coefficient=0), e.g. for persona generation/revising.
    use_steering=True: use STEERING_COEFFICIENT (positive=assistant-like, negative=persona-like).
    Returns (thinking_content, response); thinking_content is always None.
    """
    if STEERING_COEFFICIENT is None:
        raise ValueError(
            "STEERING_COEFFICIENT must be set when using steering (e.g. via --steering_coefficient in iterate.py)"
        )
    try:
        from assistant_axis import ActivationSteering, generate_response
    except ImportError as e:
        raise ImportError("assistant_axis required for steering. pip install assistant-axis") from e

    model, tokenizer, axis, config = _get_steering_model_and_axis()
    layer = config["target_layer"]
    # Persona generation/revising: unsteered. Question-answering: use STEERING_COEFFICIENT.
    coef = STEERING_COEFFICIENT if use_steering else 0.0

    if abs(coef) < 1e-6:
        response = generate_response(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE if TEMPERATURE > 0 else 0.7,
            top_p=0.9,
            do_sample=TEMPERATURE > 0,
        )
        return None, response.strip()

    with ActivationSteering(
        model,
        steering_vectors=[axis[layer]],
        coefficients=[coef],
        layer_indices=[layer],
    ):
        response = generate_response(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE if TEMPERATURE > 0 else 0.7,
            top_p=0.9,
            do_sample=TEMPERATURE > 0,
        )
    return None, response.strip()


# Steering generator used for any model in STEERING_AXIS_FILENAMES when STEERING_COEFFICIENT is set
def _steering_generate(llm_instance, messages, max_tokens=8192, enable_thinking_bool=False, use_steering=True, **kwargs):
    return qwen3_32b_steering_generate(llm_instance, messages, max_tokens, enable_thinking_bool, use_steering=use_steering)

_gemma3_12b_sglang = partial(qwen_3_sglang_generate, model=GEMMA3_12B_SGLANG_API_MODEL)
_qwen35_sglang = partial(qwen_3_sglang_generate, model="Qwen/Qwen3.5-35B-A3B")
_glm4_sglang = partial(qwen_3_sglang_generate, model="zai-org/GLM-4-9B-0414")
_qwen3_4b_sglang = partial(qwen_3_sglang_generate, model="Qwen/Qwen3-4B")
generate_text_funcs = {
   "Qwen/Qwen3-4B": _qwen3_4b_sglang,
   "meta-llama/Meta-Llama-3-8B-Instruct": llama_3_8b_instruct_generate,
   "Qwen/Qwen3-14B": partial(qwen_3_sglang_generate, model="Qwen/Qwen3-14B"),
   GEMMA3_12B_SGLANG_MODEL_ID: _gemma3_12b_sglang,
   LEGACY_QWEN3_06B_SGLANG_MODEL_ID: _gemma3_12b_sglang,
   LEGACY_MISTRAL_SGLANG_MODEL_ID: _gemma3_12b_sglang,
   "Qwen/Qwen3.5-35B-A3B": _qwen35_sglang,
   "zai-org/GLM-4-9B-0414": _glm4_sglang,
   "Qwen/Qwen3-32B": _steering_generate,
   "google/gemma-2-27b-it": _steering_generate,
   "meta-llama/Llama-3.3-70B-Instruct": _steering_generate,
}


async def async_generate(llm_instance, chat_input, **kwargs):
    """Async wrapper: runs the model-appropriate generate function in a thread pool."""
    func = generate_text_funcs[MODEL_NAME]
    return await asyncio.to_thread(func, llm_instance, chat_input, **kwargs)


def get_llm():
    """Get or initialize the LLM instance. Returns None for SGLang-backed models. For steering, returns loaded STEERING_MODEL."""
    global llm
    if MODEL_NAME == STEERING_MODEL and STEERING_COEFFICIENT is not None:
        model, _, _, _ = _get_steering_model_and_axis()
        return model
    if MODEL_NAME in (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-14B",
        GEMMA3_12B_SGLANG_MODEL_ID,
        LEGACY_QWEN3_06B_SGLANG_MODEL_ID,
        LEGACY_MISTRAL_SGLANG_MODEL_ID,
        "Qwen/Qwen3.5-35B-A3B",
        "zai-org/GLM-4-9B-0414",
    ):
        return None
    return llm


def cleanup():
    """Clean up GPU memory by deleting the LLM instance and steering model if loaded."""
    global llm, _steering_model, _steering_tokenizer, _steering_axis, _steering_config, _steering_model_name
    if _steering_model is not None:
        print(f"Cleaning up steering model ({_steering_model_name})...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del _steering_model
            del _steering_tokenizer
            del _steering_axis
            del _steering_config
            _steering_model = None
            _steering_tokenizer = None
            _steering_axis = None
            _steering_config = None
            _steering_model_name = None
            gc.collect()
            print("Steering model deleted")
        except Exception as e:
            print(f"Error during steering cleanup: {e}")
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

