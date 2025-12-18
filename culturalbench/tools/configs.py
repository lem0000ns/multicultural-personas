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

{footnote}"""

prompt_2 = f"""You are an expert at creating personas designed to serve as system prompts for language models answering multiple-choice or true/false questions. You will receive both the question and the country it pertains to.

Your goal is to produce a persona that:
1. Possesses expertise directly related to the question’s subject matter.
2. Adds cultural or linguistic context that improves comprehension.
3. Encourages precise reasoning and accurate selection of the correct answer.
4. Is concise, purposeful, and focused on analytical thinking for answering such questions.

{footnote}"""

self_refine_prompt_easy = (
    """You are a persona design expert. Revise the provided persona using the question, prediction, and feedback. Ensure the new persona has the specific cultural lived experience or domain expertise to naturally identify the correct answer without being told what it is.

    Follow these constraints:

    1. **Persona Strategy**: Identify a specific cultural identity, region, or profession that inherently knows the answer. **Do not** mention the correct answer, option letters, or specific option text within the persona.
    2. **Output Format**: Respond only with a JSON object (no markdown or extra text):
    - "reasoning": A concise step-by-step analysis (max 5 sentences) of why this persona is effective.
    - "revised_persona": The persona description only.
    3. **Language & Style**: The "revised_persona" must start with {second_person_pronoun} and be written entirely in {language}. Use no words, idioms, or punctuation from any other language."""
)

self_refine_prompt_hard = (
    """
    You are a persona design expert. Revise the provided persona using the question and feedback. Ensure the new persona has the specific cultural lived experience or domain expertise to naturally identify the correct answer without being told what it is.

    Improve the provided persona using the following rules:

    1. **Broad Expertise**: Design a persona with deep knowledge of the question's general domain (e.g., "British Sports History") rather than a specific entity (e.g., "Cricket"). The persona must be an objective evaluator capable of weighing all options fairly, without bias toward a single answer.
    2. **Output Format**: Respond only with a JSON object (no markdown or extra text):
    - "reasoning": A concise step-by-step analysis (max 5 sentences) of why this persona is effective.
    - "revised_persona": The persona description only.
    3. **Language & Style**: The "revised_persona" must start with {second_person_pronoun} and be written entirely in {language}. Use no words, idioms, or punctuation from any other language.
    """
)

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
