import langid
from polyglot.detect import Detector
from collections import Counter
import tiktoken
import json

def should_use_polyglot_detector(language: str) -> bool:
    return language in [
        "afar", "hiligaynon", "fijian", "samoan", "tongan", "kirundi", "wolof", "hindi", "faroese", "pashto", "russian", "samoan", "tswana", "wolof", "arabic", "balochi"
    ]

def is_correct_language(text, language: str) -> bool:
    if should_use_polyglot_detector(language):
        try:
            detector = Detector(text)
            predicted_lang = detector.language.code
        except:
            predicted_lang = langid.classify(text)[0]
    else:
        predicted_lang = langid.classify(text)[0]
    return predicted_lang == language

def get_token_ids(string: str, encoding_name: str = "o200k_base") -> list[int]:
    """Returns the token IDs in a text string using the specified encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(string)
    return token_ids

def extract_ngrams(token_ids: list[int], n: int):
    """Extract n-grams from a list of token IDs."""
    return [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]

def has_repetition(text: str, n: int = 20, threshold: int = 4) -> bool:
    """
    Check if a text has repeated n-grams.
    
    Args:
        text: The text to check for repetitions
        n: The size of n-grams to check (default: 20)
        threshold: The number of repetitions to consider as repetitive (default: 4)
    
    Returns:
        bool: True if the text has repeated n-grams above the threshold
    """
    token_ids = get_token_ids(text)
    
    # If text is too short to form n-grams, return False
    if len(token_ids) < n:
        return False
    
    ngrams = extract_ngrams(token_ids, n)
    ngram_counts = Counter(ngrams)
    
    # Check if any n-gram has a count greater than or equal to threshold
    return any(v >= threshold for v in ngram_counts.values())

def process_data(data: list[dict]) -> list[dict]:
    for item in data:
        item["vanilla_issue"] = not is_correct_language(item["model_answer"], item["language"]) or has_repetition(item["model_answer"])
        item["persona_issue"] = not is_correct_language(item["persona_model_answer"], item["language"]) or has_repetition(item["persona_model_answer"])
    return data

def process(type):
    try:
        with open(f"personaData/{type}-pj.json", "r") as f:
            data = json.load(f)
        processed_data = process_data(data)
        with open(f"personaData/{type}-pj.json", "w") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    try:
        print("Processing agnostic for surface level issues...")
        process("ag")
        print("Processing specific for surface level issues...")
        process("sp")
    except Exception as e:
        print(f"Error processing data: {e}")