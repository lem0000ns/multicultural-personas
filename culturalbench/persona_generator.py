"""Persona generation and refinement functions."""

from tools.utils import country_to_language, language_to_code, questions_translated, countries_translated, persona_descriptions_translated, persona_translated, predicted_answers_translated, reasonings_translated, lang_to_spp, feedback_translated
from tools.configs import (
    system_prompts,
    self_refine_prompt_easy,
    self_refine_prompt_hard,
    self_refine_prompt_hard_qwen35,
    PERSONA_REFINE_MAX_TOKENS_QWEN35_HARD,
)
from tools.llm_utils import get_llm, generate_text_funcs
from tools import llm_utils
from token_counter import add_input_tokens, add_output_tokens
import googletrans
from langdetect import detect
import json
import json_repair

def is_english(text):
    """Check if text is in English."""
    return detect(text) == "en"

async def translate_text_chunk(translator, text, language, max_retries=3):
    """Translate a single chunk of text with retry logic."""
    import asyncio
    
    for attempt in range(max_retries):
        try:
            translated = await translator.translate(text, dest=language)
            return translated.text
        except Exception as e:
            error_name = type(e).__name__
            if "Timeout" in error_name or "Timeout" in str(e):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Translation timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"Translation failed after {max_retries} attempts")
                    raise
            else:
                raise
    return text

async def translate_long_text(text, language, max_chunk_size=400, max_retries=3):
    """Translate long text by chunking it into smaller pieces."""
    translator = googletrans.Translator()
    
    # If text is short enough, translate directly
    if len(text) <= max_chunk_size:
        return await translate_text_chunk(translator, text, language, max_retries)
    
    # Split text into sentences to preserve meaning
    import re
    # Split by sentence boundaries (., !, ?) followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence is too long, split it by words
        if len(sentence) > max_chunk_size:
            # Save current chunk if not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by words
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    current_chunk += (" " if current_chunk else "") + word
            continue
        
        # If adding this sentence exceeds chunk size and current chunk is not empty, save it
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Translate each chunk
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        translated_chunk = await translate_text_chunk(translator, chunk, language, max_retries)
        translated_chunks.append(translated_chunk)
    
    return " ".join(translated_chunks)

async def translate_text(response, language, parse, max_retries=3):
    """Translate text to a given language. If parse is True, parse the response as json and return updated json object with translated revised_persona. If parse is False, return direct string translation."""
    import asyncio
    import json_repair
    
    # Retry logic for network timeouts
    for attempt in range(max_retries):
        try:
            # update json object with translated revised_persona (self-refinement)
            if parse:
                try:
                    resp_obj = json_repair.loads(response) if isinstance(response, str) else response
                    translated = await translate_long_text(resp_obj["revised_persona"], language, max_retries=max_retries)
                    resp_obj["revised_persona"] = translated
                    return json.dumps(resp_obj, ensure_ascii=False)
                except Exception:
                    return response
            # directly translate persona description (first iteration)
            translated = await translate_long_text(response, language, max_retries=max_retries)
            return translated
            
        except Exception as e:
            error_name = type(e).__name__
            if "Timeout" in error_name or "Timeout" in str(e):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"Translation timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"Translation failed after {max_retries} attempts, returning original text")
                    # Return original text if translation fails
                    if parse:
                        try:
                            resp_dict = json.loads(response) if isinstance(response, str) else response
                            return json.dumps(response, ensure_ascii=False)
                        except:
                            return response
                    return response
            else:
                # For other exceptions, re-raise
                raise


def cap(country):
    """Capitalize each word in country name."""
    country_words = country.split(" ")
    for i in range(len(country_words)):
        country_words[i] = country_words[i].capitalize()
    return " ".join(country_words)


async def generate_persona_description(question, country, mode, difficulty="Easy"):
    llm_instance = get_llm()
    if "eng" in mode or "e2l" in mode:
        language = "English"
    else:
        language = country_to_language[cap(country)].lower()
    
    # Get system prompt - call lambda for ling mode
    if "eng" in mode or "e2l" in mode:
        system_prompt = system_prompts[mode]
    else:
        system_prompt = system_prompts[mode](language)
    
    question_t = questions_translated[language.capitalize()]
    country_t = countries_translated[country.lower()]
    persona_description_t = persona_descriptions_translated[language.capitalize()]
    chat_input = [
        {"role": "system",
        "content": system_prompt
        },
        {"role": "user",
        "content": f"{question_t}: " + question + "\n\n" + f"{country_t}: " + country + f"\n\n{persona_description_t}: "}
    ]

    add_input_tokens(difficulty, mode, chat_input)
    attempts = 3
    response = ""
    while attempts > 0:
        _, response = generate_text_funcs[llm_utils.MODEL_NAME](llm_instance, chat_input, use_steering=False)
        # check if response is in correct language
        if ("e2l" in mode or "eng" in mode) and not is_english(response):
            attempts -= 1
        elif ("l2e" in mode or "ling" in mode) and is_english(response):
            attempts -= 1
        else:
            break

    add_output_tokens(difficulty, mode, response)
    translated_response = None
    if "e2l" in mode:
        translated_response = await translate_text(response, language_to_code[country_to_language[cap(country)]], parse=False)
    elif "l2e" in mode:
        translated_response = await translate_text(response, language_to_code["English"], parse=False)

    # outputs direct string response, no json, of persona description
    return response, translated_response


async def generate_new_persona(difficulty, question, previous_personas_data, mode, country, feedback=None):
    """Generate new persona description through self-refinement.
    
    For eng mode, generate english persona description. For ling mode, generate persona 
    description in the language of the country. For e2l mode, generate persona description 
    in english, then translate it to appropriate language with same LLM.
    
    Args:
        difficulty: Difficulty level ("Easy" or "Hard")
        question: The question text
        previous_personas_data: list of dicts with previous persona info.
            For Easy mode, each dict should have:
                - 'persona': persona description
                - 'model_answer': model's answer
                - 'reasoning': reasoning for the answer
                - 'iteration': iteration number
            For Hard mode, each dict should have:
                - 'persona': persona description
                - 'reasoning': reasoning for the answer
                - 'iteration': iteration number
        mode: The mode (eng, ling, l2e, or e2l)
        country: The country name
        feedback: Feedback from external model
    
    Returns:
        New persona description (no JSON)
    """
    llm_instance = get_llm()
    if "eng" in mode or "e2l" in mode:
        language = "English"
    else:
        language = country_to_language[cap(country)]
    
    # self-refinement
    question_t = questions_translated[language]
    persona_t = persona_translated[language]
    predicted_answer_t = predicted_answers_translated[language]
    feedback_t = feedback_translated[language]

    # Build content with all previous personas or just the previous one
    if difficulty == "Easy":
        iterations_description = "You will be provided with a question, its corresponding persona description, and the model's predicted answer among the 4 options."
        feedback_tip = ""
        if feedback:
            iterations_description = iterations_description [:len(iterations_description)-1] + " and the feedback on how it can be improved."
            feedback_tip = "based on the feedback provided."
        
        self_refine_prompt = self_refine_prompt_easy.format(
            language=language, 
            second_person_pronoun=lang_to_spp[language],
            iterations_description=iterations_description,
            feedback_tip=feedback_tip
        )
        
        prev_data = previous_personas_data
        persona = prev_data.get('persona', '')
        model_answer = prev_data.get('model_answer', '')
        user_content = (
            f"{question_t}: " + question + "\n\n"
            + f"{persona_t}: " + persona + "\n\n"
            + f"{predicted_answer_t}: " + model_answer
        )
        if feedback:
            user_content += "\n\n" + f"{feedback_t}: " + feedback
        user_content += "\n\n"

    # Hard mode
    else:
        iterations_description = "You will be provided with a question and its corresponding persona description."

        feedback_tip = ""
        if feedback:
            iterations_description = iterations_description [:len(iterations_description)-1] + " and the feedback on how it can be improved."
            feedback_tip = "based on the feedback provided."

        use_qwen35_hard_prompt = llm_utils.MODEL_NAME == "Qwen/Qwen3.5-35B-A3B"

        if use_qwen35_hard_prompt:
            self_refine_prompt = self_refine_prompt_hard_qwen35.format(
                language=language,
                second_person_pronoun=lang_to_spp[language],
                iterations_description=iterations_description,
                feedback_tip=feedback_tip,
            )
        else:
            self_refine_prompt = self_refine_prompt_hard.format(
                language=language,
                second_person_pronoun=lang_to_spp[language],
                iterations_description=iterations_description,
                feedback_tip=feedback_tip,
            )

        prev_data = previous_personas_data
        persona = prev_data.get('persona', '')
        user_content = (
            f"{question_t}: " + question + "\n\n"
            + f"{persona_t}: " + persona
        )
        if feedback:
            user_content += "\n\n" + f"{feedback_t}: " + feedback
        user_content += "\n\n"
        
    chat_input = [
        {"role": "system",
        "content": self_refine_prompt},
        {"role": "user",
        "content": user_content}
    ]

    add_input_tokens(difficulty, mode, chat_input)
    max_tokens_kw = {}
    if difficulty == "Hard" and llm_utils.MODEL_NAME == "Qwen/Qwen3.5-35B-A3B":
        max_tokens_kw["max_tokens"] = PERSONA_REFINE_MAX_TOKENS_QWEN35_HARD
    attempts = 3
    while attempts > 0:
        # _ is thinking content (not relevant)
        _, response = generate_text_funcs[llm_utils.MODEL_NAME](
            llm_instance, chat_input, use_steering=False, **max_tokens_kw
        )
        # sanitize json response
        try:
            response_json = json_repair.loads(response)
            if ("e2l" in mode or "eng" in mode) and not is_english(response_json["revised_persona"]):
                attempts -= 1
            elif ("l2e" in mode or "ling" in mode) and is_english(response_json["revised_persona"]):
                attempts -= 1
            else:
                break
        except Exception as e:
            print("Error parsing response: " + response + " " + str(e))
            attempts -= 1

    add_output_tokens(difficulty, mode, response)
    translated_response = None
    fixed_json_string = json.dumps(json_repair.loads(response), ensure_ascii=False)
    # translate english revised_persona to appropriate language if e2l mode
    if "e2l" in mode:
        translated_response = await translate_text(fixed_json_string, language_to_code[country_to_language[cap(country)]], parse=True)
    # translate ling mode revised_persona to english
    elif "l2e" in mode:
        translated_response = await translate_text(fixed_json_string, language_to_code["English"], parse=True)
        
    return response, translated_response

if __name__ == "__main__":
    response = "{\"reasoning\": \"The previous persona is knowledgeable about Japanese food culture, but the predicted answer is not entirely accurate. Leaving a little bit of food on the plate is not a common practice in Japan, especially when eating ramen. A more accurate approach would be to finish the entire bowl and then show appreciation for the chef's effort. This revised persona will focus on the importance of finishing the meal and expressing gratitude to the chef. By doing so, the model will be more likely to select the correct answer.\",   \"revised_persona\": あなたは、ラーメンを食べ終わった後、シェフに対して感謝の言葉を述べる日本人です。}"
    print(response)
    print("--------------------------------")
    print(json_repair.loads(response))

