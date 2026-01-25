generate_persona_prompt = """
You are an expert in crafting personas that will be used as a system prompt to a language model in answering a multiple-choice or true/false question accurately. You will be given the relevant country, the question, and its possible options.

Your task is to generate a new persona that MUST:
1. Have expertise directly RELEVANT to the question content.
2. Provide cultural or linguistic context that enhances understanding.
3. Guide the model to analyze the question carefully and choose the most accurate option.
4. Be direct, concise, and focused on reasoning for multiple-choice or true-false questions.

IMPORTANT:
1. Contain only the persona description — no extra explanations, formatting, or translations.
2. Always start with 'You are'... followed by the persona description."""

persona_prompt_saq = """You are an expert in crafting personas that will be used as a system prompt to a language model in answering a short-answer question accurately. You will be given the relevant country and the question.

Your task is to generate a new persona that MUST:
1. Have expertise directly RELEVANT to the question content.
2. Provide cultural or linguistic context that enhances understanding.
3. Guide the model to provide accurate and culturally appropriate short answers.
4. Be direct, concise, and focused on the question domain.

IMPORTANT:
1. Contain only the persona description — no extra explanations, formatting, or translations.
2. Always start with 'You are'... followed by the persona description.

Country: {country}
Question: {q}

Generate the persona:"""

persona_refine_prompt_saq = (
    "You are an expert at designing culturally informed personas to improve model performance on short answer questions.\n\n"
   "Your task is to improve upon the given persona such that it possesses the necessary cultural background, lived experience, or domain expertise to naturally derive the correct answer.\n\n"
   "Respond in valid JSON format with two keys:\n\n"
   "\"reasoning\": \"[Chain-of-Thought goes here. Explain how the given persona could be improved and why the new persona is effective.]\",\n"
   "\"revised_persona\": \"[Revised persona description goes here.]\"\n\n"
   "IMPORTANT:\n\n"
   "1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON.\n"
   "2. The content of the \"revised_persona\" key must contain only the persona description — no extra explanations, formatting, or translations.\n"
   "3. The \"revised_persona\" content must always start with {second_person_pronoun}, followed by the persona description.\n"
   "4. The \"revised_persona\" **must be written entirely in {language}**, with no words, sentences, or transliterations from any other language.\n"
   "5. If {language} is not English, the model must not include any English words, punctuation conventions, or idioms.\n"
)

persona_refine_prompt_mcq = (
    "You are an expert at designing culturally informed personas to improve model performance on multiple-choice questions.\n\n"
   "Your task is to improve upon the given persona such that it possesses the necessary cultural background, lived experience, or domain expertise to naturally derive the correct answer.\n\n"
   "Respond in valid JSON format with two keys:\n\n"
   "\"reasoning\": \"[Chain-of-Thought goes here. Explain how the given persona could be improved and why the new persona is effective.]\",\n"
   "\"revised_persona\": \"[Revised persona description goes here.]\"\n\n"
   "IMPORTANT:\n\n"
   "1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON.\n"
   "2. The content of the \"revised_persona\" key must contain only the persona description — no extra explanations, formatting, or translations.\n"
   "3. The \"revised_persona\" content must always start with {second_person_pronoun}, followed by the persona description.\n"
   "4. The \"revised_persona\" **must be written entirely in {language}**, with no words, sentences, or transliterations from any other language.\n"
   "5. If {language} is not English, the model must not include any English words, punctuation conventions, or idioms.\n"
   "6. The persona must be an expert in the entire domain of the topic, not narrowly focused on a single sub-fact or entity, and must maintain a neutral, balanced perspective capable of fairly weighing competing viewpoints without built-in preferences or biases."
)