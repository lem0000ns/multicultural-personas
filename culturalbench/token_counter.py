import json
import os
import threading

_MODEL_TO_FOLDER = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b-instruct",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-14B": "qwen3-14b",
    "google/gemma-3-12b-it": "gemma-3-12b-it",
    "Qwen/Qwen3-0.6B": "gemma-3-12b-it",
    "mistral-3-14b-instruct-2512": "gemma-3-12b-it",
    "Qwen/Qwen3.5-35B-A3B": "qwen3.5-35b",
    "Qwen/Qwen3-32B": "qwen3-32b",
    "google/gemma-2-27b-it": "gemma-2-27b-it",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b-instruct",
    "zai-org/GLM-4-9B-0414": "glm4-9b",
}

def get_model_folder(model_name):
    """Return results subfolder name for model (e.g. llama3-8b-instruct)."""
    return _MODEL_TO_FOLDER.get(model_name) or model_name.replace("/", "-").lower().replace(" ", "-")

_enc = None

def _get_enc():
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc

def _chat_to_text(chat_input):
    if not isinstance(chat_input, list):
        return str(chat_input)
    parts = []
    for m in chat_input:
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)

def count_tokens_text(text):
    if text is None:
        return 0
    return len(_get_enc().encode(str(text)))

def count_tokens_chat(chat_input):
    return count_tokens_text(_chat_to_text(chat_input))

_totals = {}
_totals_lock = threading.Lock()

def _key(difficulty, mode):
    return f"{difficulty}_{mode}"

def add_input_tokens(difficulty, mode, chat_input):
    k = _key(difficulty, mode)
    n = count_tokens_chat(chat_input)
    with _totals_lock:
        if k not in _totals:
            _totals[k] = {"input_tokens": 0, "output_tokens": 0}
        _totals[k]["input_tokens"] += n
    return n

def add_output_tokens(difficulty, mode, output_text):
    k = _key(difficulty, mode)
    n = count_tokens_text(output_text)
    with _totals_lock:
        if k not in _totals:
            _totals[k] = {"input_tokens": 0, "output_tokens": 0}
        _totals[k]["output_tokens"] += n
    return n

def get_totals():
    return dict(_totals)

def _token_counts_dir():
    from tools import llm_utils
    name = getattr(llm_utils, "MODEL_NAME", None) or ""
    folder = _MODEL_TO_FOLDER.get(name) or name.replace("/", "-").lower().replace(" ", "-")
    return os.path.join(os.path.dirname(__file__), "token_counts", folder)

def write_to_json(filepath=None, totals_dict=None):
    out = totals_dict if totals_dict is not None else get_totals()
    if not out:
        return None
    base_dir = _token_counts_dir()
    os.makedirs(base_dir, exist_ok=True)
    saved = []
    for k, v in out.items():
        path = os.path.join(base_dir, f"{k}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(v, f, indent=2, ensure_ascii=False)
        saved.append(path)
    return saved[0] if len(saved) == 1 else saved

def reset():
    global _totals
    _totals = {}
