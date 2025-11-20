from .utils import language_to_prompt_1, language_to_prompt_2

# Each prompt should guide formatting of persona description for consistency; other instructions vary depending on the prompt

footnote = """IMPORTANT:
1. Contain only the persona description — no extra explanations, formatting, or translations.
2. Always start with 'You are'... followed by the persona description."""

prompt_1 = f"""You are an expert in crafting personas that will be used as a system prompt to a language model in answering a multiple-choice or true/false question accurately. You will be given the question and the relevant country that the question content is related to.

Your task is to generate a new persona that MUST:
1. Have expertise directly RELEVANT to the question content.
2. Provide cultural or linguistic context that enhances understanding.
3. Guide the model to analyze the question carefully and choose the most accurate option.
4. Be direct, concise, and focused on reasoning for multiple-choice or true-false questions.
5. Improve upon the previous persona by guiding the model towards selecting the correct answer.

{footnote}"""

prompt_2 = f"""You are an expert at creating personas designed to serve as system prompts for language models answering multiple-choice or true/false questions. You will receive both the question and the country it pertains to.

Your goal is to produce a persona that:
1. Possesses expertise directly related to the question’s subject matter.
2. Adds cultural or linguistic context that improves comprehension.
3. Encourages precise reasoning and accurate selection of the correct answer.
4. Is concise, purposeful, and focused on analytical thinking for answering such questions.

{footnote}"""

self_refine_prompt_easy = "You are an expert at designing culturally informed personas to improve model performance on multiple-choice questions.\n\nYou will be provided with a question, the model's previous persona, its predicted answer, and the reasoning behind that prediction.\n\nYour task is to create a revised persona that increases the likelihood of the model selecting the correct answer.\n\nRespond in valid JSON format with two keys:\n\n\"reasoning\": \"[Chain-of-Thought goes here, explaining why the new persona is effective and how it addresses the previous model's failure. This must be a concise step-by-step analysis reasoning with no more than 5 sentences]\",\n\"revised_persona\": \"[Revised persona description goes here.]\"\n\nIMPORTANT:\n\n1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON. The output must be strictly parseable by json.loads().\n2. The content of the \"revised_persona\" key must contain only the persona description — no extra explanations, formatting, or translations.\n3. The \"revised_persona\" content must always start with {second_person_pronoun}, followed by the persona description.\n4. The \"revised_persona\" **must be written entirely in {language}**, with no words, sentences, or transliterations from any other language.\n5. If {language} is not English, the model must not include any English words, punctuation conventions, or idioms.\n6. The \"reasoning\" key must be concise and about 5 sentences long."

self_refine_prompt_hard = """You are an expert in designing culturally informed personas to improve model performance on multiple-choice or true/false questions. You will be provided with a question, the model's previous persona, and the model's responses to several true/false statements related to that question. Your task is to create a revised persona that increases the likelihood of the model selecting the correct response.\n\nRespond in valid JSON format with two keys:\n\n\"reasoning\": (Chain-of-Thought goes here, explaining why the new persona is effective and how it addresses the previous model's failure based on the true/false responses. This must be a detailed and concise step-by-step analysis.),\n\"revised_persona\": (Revised persona description goes here.)\n\nIMPORTANT:\n\n1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON. The output must be strictly parseable by json.loads().\n2. The content of the "revised_persona" key must contain only the persona description — no extra explanations, formatting, or translations.\n3. The "revised_persona" content must always start with {second_person_pronoun}, followed by the persona description.\n4. The "revised_persona" **must be written entirely in {language}**, with no words, sentences, or transliterations from any other language.\n5. If {language} is not English, the model must not include any English words, punctuation conventions, or idioms.\n6. The "reasoning" key must be concise and about 5 sentences long."""

system_prompts = {
    "eng_p1": prompt_1,
    "ling_p1": lambda lang: language_to_prompt_1[lang],
    "eng_p2": prompt_2,
    "ling_p2": lambda lang: language_to_prompt_2[lang],
    "e2l_p1": prompt_1,
    "e2l_p2": prompt_2,
    "l2e_p1": lambda lang: language_to_prompt_1[lang],
    "l2e_p2": lambda lang: language_to_prompt_2[lang]
}