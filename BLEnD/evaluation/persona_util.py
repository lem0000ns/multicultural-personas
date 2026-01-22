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

persona_refine_prompt_saq = """You are an expert at designing culturally informed personas to improve model performance on short-answer questions.

You will be provided with a question and its corresponding persona description from a previous iteration.

Your task is to improve upon the given persona by revising it such that it possesses the necessary cultural background, lived experience, or domain expertise to naturally derive the correct answer.

Respond in valid JSON format with two keys:

"reasoning": "[Chain-of-Thought goes here, explaining why the new persona is effective. This must be a concise step-by-step analysis reasoning with no more than 5 sentences]",
"revised_persona": "[Revised persona description goes here.]"

IMPORTANT:

1. You must respond only with a valid JSON object. Do not include any text, explanation, markdown code fences, or formatting outside the JSON.
2. The content of the "revised_persona" key must contain only the persona description — no extra explanations, formatting, or translations.
3. The "revised_persona" content must always start with "You are", followed by the persona description.
4. The "revised_persona" **must be written entirely in English**, with no words, sentences, or transliterations from any other language.
5. **CRITICAL:** The revised persona must NOT explicitly state the correct answer. Instead, focus on defining a specific cultural identity, region, or profession that would inherently know the answer.

Question: {q}

Previous persona: {prev_persona}

Generate the improved persona:"""