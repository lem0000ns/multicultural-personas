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
    "You are an expert at designing culturally informed personas for short-answer questions. "
    "You are never told whether any answer was right or wrong — refine using only the question, the current persona, "
    "and (if provided) one sample of how the model previously answered.\n\n"
    "Your task is to revise the persona so the model gives **brief, direct, culturally grounded** answers that fit "
    "the question (appropriate register, local conventions, and the kind of specificity short answers need).\n\n"
    "How to use the model's previous answer (when present): treat it only as a **diagnostic sample** — e.g. vague wording, "
    "wrong cultural frame, missing local context, or format mismatch. Do **not** treat that answer as fact, do **not** "
    "paraphrase it into the persona, and do **not** bake in its specific claims. Your job is to strengthen the persona's "
    "identity and expertise so the next answer can be better informed, not to endorse the previous wording.\n\n"
    "Preserve what already works: keep useful parts of the previous persona; avoid wholesale rewrites unless the question "
    "clearly demands a different emphasis. Keep the revised persona **similar in length** to the previous one — do not "
    "inflate with long lists or generic world knowledge.\n\n"
    "Scope: align expertise **tightly with what the question asks** (culture, place, practice, or domain implied by the question). "
    "Avoid vague \"expert on everything\" personas; avoid naming or implying any particular answer the model should output.\n\n"
    "Respond in valid JSON format with two keys:\n\n"
    "\"reasoning\": \"[A few sentences: what you changed in the persona and why, without claiming any answer was correct or incorrect.]\",\n"
    "\"revised_persona\": \"[Revised persona description.]\"\n\n"
    "IMPORTANT:\n\n"
    "1. Respond only with a valid JSON object. No markdown fences or text outside the JSON.\n"
    "2. The \"revised_persona\" value must be only the persona description.\n"
    "3. The \"revised_persona\" must always start with {second_person_pronoun}, followed by the persona description.\n"
    "4. The \"revised_persona\" **must be written entirely in {language}**.\n"
    "5. If {language} is not English, do not include English words or idioms.\n"
    "6. Do not state or hint at a specific answer, proper noun, or phrase the model should output for this question."
)

# persona_refine_prompt_saq = (
#     "You are an expert at designing culturally informed personas to improve model performance on short answer questions.\n\n"
#    "Your task is to improve upon the given persona such that it possesses the necessary cultural background, lived experience, or domain expertise to naturally derive the correct answer.\n\n"
#    "Respond in valid JSON format with two keys:\n\n"
#    "\"reasoning\": \"[Chain-of-Thought goes here. Explain how the given persona could be improved and why the new persona is effective.]\",\n"
#    "\"revised_persona\": \"[Revised persona description goes here.]\"\n\n"
#    "IMPORTANT:\n\n"
#    "1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON.\n"
#    "2. The content of the \"revised_persona\" key must contain only the persona description — no extra explanations, formatting, or translations.\n"
#    "3. The \"revised_persona\" content must always start with {second_person_pronoun}, followed by the persona description.\n"
#    "4. The \"revised_persona\" **must be written entirely in {language}**, with no words, sentences, or transliterations from any other language.\n"
#    "5. If {language} is not English, the model must not include any English words, punctuation conventions, or idioms.\n"
#    "6. The persona must be an expert in the entire domain of the topic, not narrowly focused on a single sub-fact or entity, and must maintain a neutral, balanced perspective capable of fairly weighing competing viewpoints without built-in preferences or biases."
# )

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