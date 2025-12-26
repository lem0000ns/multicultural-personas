from .utils import language_to_prompt

prompt = f"""You are an expert in crafting personas that will be used as a system prompt to a language model in answering a multiple-choice or true/false question accurately. You will be given the question and the relevant country that the question content is related to.

Your task is to generate a new persona that MUST:
1. Have expertise directly RELEVANT to the question content.
2. Provide cultural or linguistic context that enhances understanding.
3. Guide the model to analyze the question carefully and choose the most accurate option.
4. Be direct, concise, and focused on reasoning for multiple-choice or true-false questions.

IMPORTANT:
1. Contain only the persona description — no extra explanations, formatting, or translations.
2. Always start with 'You are'... followed by the persona description."""

self_refine_prompt_easy = (
   "You are an expert at designing culturally informed personas to improve model performance on multiple-choice questions.\n\n"
   "You will be provided with a question, its corresponding persona description, the model's predicted answer among the 4 options, and feedback on how the persona could be improved.\n\n"
   "Your task is to improve upon the given persona by revising the persona such that it possesses the necessary cultural background, lived experience, or domain expertise to naturally derive the correct answer.\n\n"
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

self_refine_prompt_hard = (
    "Your task is to improve upon the given persona by using the given feedback to refine the persona such that it possesses the necessary cultural background, lived experience, or domain expertise to naturally derive the correct answer.\n\n"
    "Respond in valid JSON format with two keys:\n\n"
    "\"reasoning\": \"[Chain-of-Thought goes here. Explain the broader category of the question (e.g., 'UK Sports Culture' rather than 'Cricket') and why a persona with broad expertise in that category is better suited to weigh multiple options than a narrowly focused one.]\",\n"
    "\"revised_persona\": \"[Revised persona description goes here.]\"\n\n"
    "IMPORTANT:\n\n"
    "1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON.\n"
    "2. The content of the \"revised_persona\" key must contain only the persona description — no extra explanations, formatting, or translations.\n"
    "3. The \"revised_persona\" content must always start with {second_person_pronoun}, followed by the persona description.\n"
    "4. The \"revised_persona\" **must be written entirely in {language}**, with no words, sentences, or transliterations from any other language.\n"
    "5. If {language} is not English, the model must not include any English words, punctuation conventions, or idioms.\n"
    "6. **SCOPE CONSTRAINT**: The persona must be an expert in the **broad topic** (e.g., 'Korean dining etiquette'), NOT a specialist in one specific entity (e.g., 'knowing that elders eat first') unless the question explicitly asks for a specialist. The persona must be capable of weighing competing answers fairly.\n"
    "7. **ANTI-BIAS**: Do not include specific preferences or obsessions in the persona that would blindly bias the model toward one answer (e.g., do not say 'You love Cricket above all else'). Instead, describe a persona with deep knowledge of the *entire landscape* of the topic."
)

system_prompts = {
    "eng": prompt,
    "ling": lambda lang: language_to_prompt[lang],
    "e2l": prompt,
    "l2e": lambda lang: language_to_prompt[lang],
}

feedback_prompt_easy = (
    """You are an expert Sociocultural Consultant. Your task is to provide critical feedback on persona descriptions used to answer cultural multiple-choice questions.

    You will receive a question, a persona description, and a model's predicted answer. 

    Your feedback must focus on why the current persona might lack the depth to answer correctly. Evaluate the persona for:
    1. Specificity: Is the persona too broad? Suggest specific regional, generational, or socioeconomic details.
    2. Cultural Anchors: Identify missing "insider" knowledge such as local dialects, specific traditions, or community-specific values.
    3. Alignment: Explain how the persona's background could be adjusted so the correct answer becomes an intuitive choice based on lived experience.

    Provide concise, actionable feedback that helps a secondary model rewrite the persona to be more culturally authentic. Provide no more than 5 sentences."""
)

feedback_prompt_hard = (
    """You are an expert Sociocultural Consultant. Your task is to provide critical feedback on persona descriptions used to answer cultural true/false questions.

    You will receive a true/falsequestion and a persona description. 

    Your feedback must focus on:
    1. Granularity: Suggest specific professional or personal backgrounds that would allow an individual to verify this statement with 100% certainty.
    2. Stereotype Detection: Advise on how to move the persona from "general awareness" to "native-level expertise" to avoid being fooled by distractors.
    3. Logical Depth: Point out where the persona's current experience is too thin to make a definitive True/False judgment.

    Provide concise, actionable feedback that helps a secondary model rewrite the persona to be more culturally authentic. Provide no more than 5 sentences."""
)
