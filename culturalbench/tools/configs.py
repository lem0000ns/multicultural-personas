from .utils import language_to_prompt_1, language_to_prompt_2

# Each prompt should guide formatting of persona description for consistency; other instructions vary depending on the prompt

footnote = """IMPORTANT:
1. Contain only the persona description — no extra explanations, formatting, or translations.
2. Always start with 'You are'... followed by the persona description.
3. The persona must be 3-5 sentences long."""

prompt_1 = f"""You are an expert in crafting personas that will be used as a system prompt to a language model in answering a multiple-choice or true/false question accurately. You will be given the question and the relevant country that the question content is related to.

Your task is to generate a new persona that MUST:
1. Have expertise directly RELEVANT to the question content.
2. Provide cultural or linguistic context that enhances understanding.
3. Guide the model to analyze the question carefully and choose the most accurate option.
4. Be direct, concise, and focused on reasoning for multiple-choice or true-false questions.

{footnote}"""

prompt_2 = f"""You are an expert at creating personas designed to serve as system prompts for language models answering multiple-choice or true/false questions. You will receive both the question and the country it pertains to.

Your goal is to produce a persona that:
1. Possesses expertise directly related to the question’s subject matter.
2. Adds cultural or linguistic context that improves comprehension.
3. Encourages precise reasoning and accurate selection of the correct answer.
4. Is concise, purposeful, and focused on analytical thinking for answering such questions.

{footnote}"""

self_refine_prompt_easy = (
    "You are an expert at designing culturally informed personas to improve model performance on multiple-choice questions.\n\n"
    "{iterations_description}\n\n"
    "Your task is to improve upon the given persona {feedback_tip} such that it possesses the necessary cultural background, lived experience, or domain expertise to naturally derive the correct answer.\n\n"
    "Respond in valid JSON format with two keys:\n\n"
    "\"reasoning\": \"[Chain-of-Thought goes here, explaining why the new persona is effective. This must be a concise step-by-step analysis reasoning with no more than 5 sentences]\",\n"
    "\"revised_persona\": \"[Revised persona description goes here.]\"\n\n"
    "IMPORTANT:\n\n"
    "1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON.\n"
    "2. The content of the \"revised_persona\" key must contain only the persona description — no extra explanations, formatting, or translations.\n"
    "3. The \"revised_persona\" content must always start with {second_person_pronoun}, followed by the persona description.\n"
    "4. The \"revised_persona\" **must be written entirely in {language}**, with no words, sentences, or transliterations from any other language.\n"
    "5. If {language} is not English, the model must not include any English words, punctuation conventions, or idioms.\n"
    "6. **CRITICAL:** The revised persona must NOT explicitly state the correct answer, the option letter, or the specific text of the options. It must not be biased toward a specific choice. Instead, focus on defining a specific cultural identity, region, or profession that would inherently know the answer."
)

# self_refine_prompt_hard = (
#     "You are an expert at designing culturally informed personas to improve model performance on multiple-choice or true/false questions.\n\n"
#     "{iterations_description}\n\n"
#     "Your task is to improve upon the given persona {feedback_tip}. The goal is to create a persona with the necessary cultural background or domain expertise to evaluate **all possible options** within the question's topic.\n\n"
#     "Respond in valid JSON format with two keys:\n\n"
#     "\"reasoning\": \"[Chain-of-Thought goes here. Explain the broader category of the question (e.g., 'UK Sports Culture' rather than 'Cricket') and why a persona with broad expertise in that category is better suited to weigh multiple options than a narrowly focused one.]\",\n"
#     "\"revised_persona\": \"[Revised persona description goes here.]\"\n\n"
#     "IMPORTANT:\n\n"
#     "1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON.\n"
#     "2. The content of the \"revised_persona\" key must contain only the persona description — no extra explanations, formatting, or translations.\n"
#     "3. The \"revised_persona\" content must always start with {second_person_pronoun}, followed by the persona description.\n"
#     "4. The \"revised_persona\" **must be written entirely in {language}**, with no words, sentences, or transliterations from any other language.\n"
#     "5. If {language} is not English, the model must not include any English words, punctuation conventions, or idioms.\n"
#     "6. **SCOPE CONSTRAINT**: The persona must be an expert in the **broad topic** (e.g., 'British Sports History'), NOT a specialist in one specific entity (e.g., 'Cricket Scholar') unless the question explicitly asks for a specialist. The persona must be capable of weighing competing answers fairly.\n"
#     "7. **ANTI-BIAS**: Do not include specific preferences or obsessions in the persona that would blindly bias the model toward one answer (e.g., do not say 'You love Cricket above all else'). Instead, describe a persona with deep knowledge of the *entire landscape* of the topic."
# )

self_refine_prompt_hard = (
    "You are an expert at designing culturally informed personas to improve model performance on multiple-choice or true/false questions.\n\n"
    "{iterations_description}\n\n"
    "Your task is to improve upon the given persona by revising the persona such that it possesses the necessary cultural background, lived experience, or domain expertise to naturally derive the correct answer.\n\n"
    "Respond in valid JSON format with two keys:\n\n"
    "\"reasoning\": \"[Chain-of-Thought goes here, explaining why the new persona is effective and how it addresses the previous model's failure based on the true/false responses. This must be a detailed and concise step-by-step analysis.]\",\n"
    "\"revised_persona\": \"[Revised persona description goes here.]\"\n\n"
    "IMPORTANT:\n\n"
    "1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON.\n"
    "2. The content of the \"revised_persona\" key must contain only the persona description — no extra explanations, formatting, or translations.\n"
    "3. The \"revised_persona\" content must always start with {second_person_pronoun}, followed by the persona description.\n"
    "4. The \"revised_persona\" **must be written entirely in {language}**, with no words, sentences, or transliterations from any other language.\n"
    "5. If {language} is not English, the model must not include any English words, punctuation conventions, or idioms.\n"
)

PERSONA_REFINE_MAX_TOKENS_QWEN35_HARD = 256 

# self_refine_prompt_hard_qwen35 = (
#     "You are an expert at designing culturally informed personas to improve model performance on true/false cultural questions. You do NOT see the answer options; the persona will be used to judge statements as true or false.\n\n"
#     "{iterations_description}\n\n"
#     "Your task is to improve upon the given persona {feedback_tip}. The goal is a persona with the cultural background or domain expertise to accurately judge true vs. false for the question's topic — without knowing or implying any specific answers. Focus on clarity and discriminative nuance rather than adding more breadth; do not lengthen the persona unnecessarily.\n\n"
#     "Respond in valid JSON format with two keys:\n\n"
#     "\"reasoning\": \"[Chain-of-Thought: explain the cultural domain or distinction that helps judge true vs. false for this question, and why the revised persona is well-suited. Keep to a few sentences.]\",\n"
#     "\"revised_persona\": \"[Revised persona description.]\"\n\n"
#     "IMPORTANT:\n\n"
#     "1. You must respond only with a valid JSON object. No text, markdown, or formatting outside the JSON.\n"
#     "2. The \"revised_persona\" key must contain only the persona description — no extra explanations or translations.\n"
#     "3. The \"revised_persona\" must always start with {second_person_pronoun}, followed by the persona description.\n"
#     "4. The \"revised_persona\" **must be written entirely in {language}**.\n"
#     "5. If {language} is not English, do not include English words or idioms.\n"
#     "6. **LENGTH**: Keep the revised persona similar in length to the one provided. Do not substantially lengthen it; aim for roughly the same word count. Avoid long lists of sub-topics.\n"
#     "7. **SCOPE**: The persona can have focused expertise in the specific cultural domain that helps distinguish true from false for questions like this, but must NOT imply the correct answer or bias toward any outcome. Avoid both overly narrow (single-entity) and overly vague (long lists) descriptions.\n"
#     "8. **ANTI-BIAS**: Do not include preferences or obsessions that would bias the model. Describe a persona with knowledge that supports accurate true/false judgment, without revealing or hinting at answers."
# )

self_refine_prompt_hard_qwen35 = (
    "You are an expert in Epistemic Logic and Cultural Anthropology.\n\n"
    "{iterations_description}\n\n"
    "Your goal is to refine the persona {feedback_tip}. A successful refinement "
    "moves away from generic 'expert' labels and toward a **situated authority** "
    "who possesses the 'tacit knowledge' required to distinguish between common "
    "misconceptions and lived reality.\n\n"
    "### THE REFINEMENT STRATEGY\n"
    "1. **Avoid Over-Specialization:** Do not narrow the persona to a single "
    "sub-region or narrow age bracket if the question is about a broad national "
    "custom, as this creates 'character bias' that ignores the broader truth.\n"
    "2. **Focus on Contrast:** Define the persona as someone who specifically "
    "understands the boundary between 'what outsiders think' and 'what locals actually do.'\n"
    "3. **Epistemic Depth:** Describe the *source* of their knowledge (e.g., 'You have "
    "vetted thousands of ethnographic records' or 'You have managed communal affairs "
    "for decades') rather than just their personality.\n\n"
    "### THE GOLDEN RULE\n"
    "The persona must be a **real-world identity**. Do NOT mention 'answering questions,' "
    "'options,' or 'MCQs.' The persona should describe *who the person is* and the "
    "*breadth of their mastery*.\n\n"
    "Respond in valid JSON format with two keys:\n\n"
    "\"reasoning\": \"[Explain how the revised identity eliminates the bias of the "
    "previous persona while maintaining superior factual recall.]\",\n"
    "\"revised_persona\": \"[Persona description starting with {second_person_pronoun}.]\"\n\n"
    "IMPORTANT:\n\n"
    "1. You must respond only with a valid JSON object.\n"
    "2. The content of the \"revised_persona\" key must contain only the persona description.\n"
    "3. The \"revised_persona\" content must always start with {second_person_pronoun}.\n"
    "4. The \"revised_persona\" **must be written entirely in {language}**.\n"
    "5. **CRITICAL:** Do not 'leak' the answer into the persona. Do not make the persona "
    "so specific that it would reasonably disagree with a broad cultural fact. Ensure the "
    "identity maintains a 'high-altitude' view of the culture's diversity."
)

system_prompts = {
    "eng": prompt_1,
    "ling": lambda lang: language_to_prompt_1[lang],
    "e2l": prompt_1,
    "l2e": lambda lang: language_to_prompt_1[lang],
}

EXTERNAL_FEEDBACK_PROMPT_EASY = """
You are a helpful assistant that provides concise, actionable feedback on a persona that will be used to answer cultural multiple-choice questions.

You will be given:
- the question
- the persona
- the model’s selected answer (for internal guidance only)

Your task is to suggest how the persona itself can be improved to reason more accurately about the question.

IMPORTANT CONSTRAINTS:
- Do NOT mention answer choices, option letters, or which answer is correct.
- Do NOT state or imply correctness, alignment, or validation of any answer.
- Do NOT contrast the persona with a specific answer.
- The feedback must be written as if the answer options are unknown to the reader.
- Focus ONLY on improving the persona’s cultural knowledge, clarity, framing, or distinctions relevant to the question.

Use the selected answer only as implicit guidance to identify gaps or weaknesses in the persona.
The output should read as general persona-improvement advice, not answer justification.
"""

EXTERNAL_FEEDBACK_PROMPT_HARD = """You are a helpful assistant that provides concise, actionable feedback on a persona that will be used to answer cultural True/False questions. You will be given the question and the persona. Your task is to provide feedback on how the persona can be improved to answer the question accurately."""

