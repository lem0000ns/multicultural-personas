import langid
from polyglot.detect import Detector
from collections import Counter
import tiktoken
import json

language_codes = {
    "afar": "aa",            # ISO 639-1
    "fijian": "fj",          # ISO 639-1
    "samoan": "sm",          # ISO 639-1
    "tongan": "to",          # ISO 639-1
    "wolof": "wo",           # ISO 639-1
    "hindi": "hi",           # ISO 639-1
    "faroese": "fo",         # ISO 639-1
    "pashto": "ps",          # ISO 639-1
    "russian": "ru",         # ISO 639-1
    "tswana": "tn",          # ISO 639-1
    "arabic": "ar",          # ISO 639-1
    "english": "en",         # ISO 639-1
    "spanish": "es",         # ISO 639-1
    "korean": "ko",          # ISO 639-1
    "japanese": "ja",        # ISO 639-1
    "chinese": "zh",         # ISO 639-1
    "german": "de",          # ISO 639-1
    "hebrew": "he",          # ISO 639-1
    "hungarian": "hu",       # ISO 639-1
}


def should_use_polyglot_detector(language: str) -> bool:
    return language in [
        "afar", "fijian", "samoan", "tongan", "wolof", "hindi", "faroese", "pashto", "russian","tswana", "arabic"
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
    
    # if low-accuracy detection language, check if predicted language is in high-accuracy detection language, else return True
    if (language == "papiamento" or language == "balochi" or language == "hiligaynon" or language == "kirundi"):
        if predicted_lang in [language_codes[lang] for lang in ["afar", "arabic", "chinese", "english", "faroese", "german", "hebrew", "hindi", "hungarian", "japanese", "korean", "pashto", "spanish", "tongan"]]:
            return False
        return True
    print("Predicted language:", predicted_lang)
    print("Language code:", language_codes[language])
    return predicted_lang == language_codes[language]

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

def process(type, data):
    try:
        processed_data = process_data(data)
        return processed_data
    except Exception as e:
        print(f"Error processing data: {e}")