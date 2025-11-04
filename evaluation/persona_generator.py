"""Persona generation and refinement functions."""

from tools.utils import country_to_language, language_to_code, questions_translated, countries_translated, persona_descriptions_translated, previous_persona_translated, predicted_answers_translated, reasonings_translated, lang_to_spp
from tools.configs import system_prompts, self_refine_prompt_easy, self_refine_prompt_hard
from tools.llm_utils import get_llm, generate_text
import googletrans
from langdetect import detect
import json

def is_english(text):
    """Check if text is in English."""
    return detect(text) == "en"

# used to sanitize json response for self-refinement and also easy mode answer + CoT
def sanitize_json(response, question_type):
    """Sanitize response to be a valid json object. If response is not a valid json object, add closing brace and remove newlines."""
    try:
        json.loads(response)
        return response
    except json.JSONDecodeError:
        pass

    if question_type == "easy":
        first_key_str = "answer"
        second_key_str = "reasoning"
    elif question_type == "refine":
        first_key_str = "reasoning"
        second_key_str = "revised_persona"
    elif question_type == "hard":
        first_key_str = "correct"
        second_key_str = "reasoning"

    response = response.replace("\n", "")
    if response[-1] != "}":
        response = response + "}"
    if response[0] != "{":
        response = "{" + response
    
    # deal with quotes in first key
    second_key_pos = response.find(second_key_str)
    comma_before_second_key = response.rfind(",", 0, second_key_pos)
    first_key_val = response[response.find(":") + 1: comma_before_second_key].strip()
    # replace double quotes with single quotes
    first_key_val = first_key_val.replace("\"", "'")
    # wrap first_key in double quotes again
    if first_key_val[0] == "'":
        first_key_val = "\"" + first_key_val[1:]
    else:
        first_key_val = "\"" + first_key_val
    if first_key_val[-1] == "'":
        first_key_val = first_key_val[:-1] + "\""
    else:
        first_key_val = first_key_val + "\""

    # deal with quotes in revised_persona
    second_key_val = response[response.rfind(":") + 1:response.rfind("}")].strip()
    # replace double quotes with single quotes
    second_key_val = second_key_val.replace("\"", "'")
    # wrap second_key in double quotes again
    if second_key_val[0] == "'":
        second_key_val = "\"" + second_key_val[1:]
    else:
        second_key_val = "\"" + second_key_val
    if second_key_val[-1] == "'":
        second_key_val = second_key_val[:-1] + "\""
    else:
        second_key_val = second_key_val + "\""

    response = f"{{\"{first_key_str}\": {first_key_val}, \"{second_key_str}\": {second_key_val}}}"

    return response

async def translate_text(response, language, parse):
    """Translate text to a given language. If parse is True, parse the response as json and return updated json object with translated revised_persona. If parse is False, return direct string translation."""
    translator = googletrans.Translator()
    # update json object with translated revised_persona (self-refinement)
    if parse:
        try:
            response = json.loads(response)
            translated = await translator.translate(response["revised_persona"], dest=language)
            response["revised_persona"] = translated.text
            return json.dumps(response, ensure_ascii=False)
        except json.JSONDecodeError:
            return response
    # directly translate persona description (first iteration)
    translated = await translator.translate(response, dest=language)
    return translated.text


def cap(country):
    """Capitalize each word in country name."""
    country_words = country.split(" ")
    for i in range(len(country_words)):
        country_words[i] = country_words[i].capitalize()
    return " ".join(country_words)


async def generate_persona_description(question, country, mode):
    """Generate initial persona description for a given question and country.
    
    Args:
        question: The question text
        country: The country name
        mode: The mode (eng_*, ling_*, e2l_*, or l2e_*)
    
    Returns:
        Generated persona description
    """
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
    # outputs direct string response, no json, of persona description
    attempts = 3
    while attempts > 0:
        response = generate_text(chat_input, llm_instance)
        # check if english
        if ("e2l" in mode or "eng" in mode) and not is_english(response):
            attempts -= 1
        elif ("l2e" in mode or "ling" in mode) and is_english(response):
            attempts -= 1
        else:
            break

    if attempts == 0:
        return None, None

    translated_response = None
    if "e2l" in mode:
        translated_response = await translate_text(response, language_to_code[country_to_language[cap(country)]], parse=False)
    elif "l2e" in mode:
        translated_response = await translate_text(response, language_to_code["English"], parse=False)

    return response, translated_response


async def generate_new_persona(difficulty, question, old_persona, pred_ans, reasoning, mode, country, iteration):
    """Generate new persona description through self-refinement.
    
    For eng mode, generate english persona description. For ling mode, generate persona 
    description in the language of the country. For e2l mode, generate persona description 
    in english, then translate it to appropriate language with same LLM.
    
    Args:
        difficulty: Difficulty level ("Easy" or "Hard")
        question: The question text
        old_persona: Previous persona description
        pred_ans: Predicted answer
        reasoning: Reasoning for the prediction
        mode: The mode (eng_*, ling_*, or e2l_*)
        country: The country name
        iteration: Current iteration number
    
    Returns:
        New persona description
    """
    llm_instance = get_llm()
    if "eng" in mode or "e2l" in mode:
        language = "English"
    else:
        language = country_to_language[cap(country)]
    
    # self-refinement
    question_t = questions_translated[language]
    previous_persona_t = previous_persona_translated[language]
    predicted_answer_t = predicted_answers_translated[language]
    reasoning_t = reasonings_translated[language]
    if difficulty == "Easy":
        self_refine_prompt = self_refine_prompt_easy.format(language=language, second_person_pronoun=lang_to_spp[language])
    else:
        self_refine_prompt = self_refine_prompt_hard.format(language=language, second_person_pronoun=lang_to_spp[language])
    chat_input = [
        {"role": "system",
        "content": self_refine_prompt},
        {"role": "user",
        "content": f"{question_t}: " + question + "\n\n" + f"{previous_persona_t}: " + old_persona + "\n\n" + f"{predicted_answer_t}: " + pred_ans + "\n\n" + f"{reasoning_t}: " + reasoning}
    ]

    # outputs json object with two keys: reasoning and revised_persona

    # 3 attempts to generate response in correct language
    attempts = 3
    while attempts > 0:
        response = generate_text(chat_input, llm_instance)
        # sanitize json response
        response = sanitize_json(response, "refine")
        try:
            response_json = json.loads(response)
            if ("e2l" in mode or "eng" in mode) and not is_english(response_json["revised_persona"]):
                attempts -= 1
            elif ("l2e" in mode or "ling" in mode) and is_english(response_json["revised_persona"]):
                attempts -= 1
            else:
                break
        except json.JSONDecodeError:
            attempts -= 1
    if attempts == 0:
        print("Unable to generate response in correct language", response)
        return None, None

    translated_response = None
    # translate english revised_persona to appropriate language if e2l mode
    if "e2l" in mode:
        translated_response = await translate_text(response, language_to_code[country_to_language[cap(country)]], parse=True)
    # translate ling mode revised_persona to english
    elif "l2e" in mode:
        translated_response = await translate_text(response, language_to_code["English"], parse=True)
        
    return response, translated_response

if __name__ == "__main__":
    response = "{\"reasoning\": \"The previous persona is knowledgeable about Japanese food culture, but the predicted answer is not entirely accurate. Leaving a little bit of food on the plate is not a common practice in Japan, especially when eating ramen. A more accurate approach would be to finish the entire bowl and then show appreciation for the chef's effort. This revised persona will focus on the importance of finishing the meal and expressing gratitude to the chef. By doing so, the model will be more likely to select the correct answer.\",   \"revised_persona\": あなたは、ラーメンを食べ終わった後、シェフに対して感謝の言葉を述べる日本人です。}"
    print(response)
    print("--------------------------------")
    print(sanitize_json(response, "refine"))

