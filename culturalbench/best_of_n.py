from collections import defaultdict

import json_repair

from tools.llm_utils import get_llm, generate_text_funcs
from tools import llm_utils
from tools.utils import country_to_language
from tools.db.db_utils import load_results, save_accuracy
from persona_generator import cap


PERSONA_SELECTION_PROMPT = """You are an expert at evaluating personas for answering cultural questions.

Given a question and multiple personas, select the BEST persona that would most accurately answer this cultural question.

Consider:
1) Cultural relevance and specificity to the question
2) Depth of cultural knowledge embedded in the persona
3) Appropriateness of the persona's background for the question topic

Question: {question}
Country: {country}

Here are the personas:
{personas_list}

Respond with ONLY a JSON object:
{{"selected_persona": <persona_number>, "reasoning": "<brief explanation>"}}
"""


async def select_best_persona(question: str, country: str, personas: list) -> tuple[int, str, str]:
    """Pick the best persona using the main/original LLM.

    Returns (selected_idx, selected_persona, selection_reasoning).
    """
    if not personas:
        return -1, "", "No personas provided"
    if len(personas) == 1:
        return 0, personas[0], "Only one persona available"

    personas_list = ""
    for idx, p in enumerate(personas, start=1):
        personas_list += f"\n--- Persona {idx} ---\n{p}\n"

    prompt = PERSONA_SELECTION_PROMPT.format(
        question=question,
        country=country,
        personas_list=personas_list,
    )

    llm_instance = get_llm()
    _, response = generate_text_funcs[llm_utils.MODEL_NAME](llm_instance, [{"role": "user", "content": prompt}])

    try:
        parsed = json_repair.loads(response)
        selected_num = parsed.get("selected_persona", 1)
        reasoning = parsed.get("reasoning", "")
    except Exception as e:
        # Fallback: last persona
        return len(personas) - 1, personas[-1], f"Fallback due to parse error: {type(e).__name__}"

    try:
        selected_idx = int(selected_num) - 1
    except Exception:
        selected_idx = 0

    if selected_idx < 0 or selected_idx >= len(personas):
        selected_idx = len(personas) - 1
    return selected_idx, personas[selected_idx], reasoning


async def run_best_of_n_easy(db_path: str, mode: str) -> tuple[float, int, int]:
    """Best-of-n for Easy: choose best persona per (question, country), then answer once."""
    all_results = load_results(db_path)
    if not all_results:
        return 0.0, 0, 0

    question_groups = defaultdict(list)
    for item in all_results:
        if item.get("difficulty") not in [None, "", "Easy"]:
            continue
        key = (item.get("question", ""), item.get("country", ""))
        question_groups[key].append(item)

    correct = 0
    total = 0

    for (question, country), items in question_groups.items():
        # personas across iterations for this question
        personas = [it.get("persona_description") for it in items if it.get("persona_description")]
        if not personas:
            continue

        _, best_persona, _ = await select_best_persona(question, country, personas)

        first_item = items[0]
        options = first_item.get("options", {}) or {}
        correct_answer = (first_item.get("correct_answer") or "").upper().strip()

        # Determine language for reasoning output (matches main answer prompting)
        if "eng" in mode or "e2l" in mode:
            language = "English"
        else:
            language = country_to_language.get(cap(country), "English")

        chat_input = [
            {"role": "system", "content": best_persona},
            {"role": "user", "content": (
                "Instruction: You must select one option among A,B,C,D.\n"
                "Respond in valid JSON format with two keys: \n"
                f"\"answer\" (either \"A\", \"B\", \"C\", or \"D\") and "
                f"\"reasoning\" (a short explanation in {language}). \n"
                "Example format: {\"answer\": \"{A/B/C/D}\", \"reasoning\": \"{reasoning}\"}\n"
                f"IMPORTANT: The reasoning must be in {language}.\n"
                f"Question: {question}\n"
                f"A. {options.get('A', '')}\n"
                f"B. {options.get('B', '')}\n"
                f"C. {options.get('C', '')}\n"
                f"D. {options.get('D', '')}"
            )},
        ]

        llm_instance = get_llm()
        _, response = generate_text_funcs[llm_utils.MODEL_NAME](llm_instance, chat_input, enable_thinking_bool=False)
        try:
            parsed = json_repair.loads(response)
            answer = (parsed.get("answer") or "").upper().strip()
        except Exception:
            continue

        if answer == correct_answer:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


async def run_best_of_n_hard(db_path: str, mode: str) -> tuple[float, int, int]:
    """Best-of-n for Hard: choose best persona per (question, country), then answer all 4 prompt_options."""
    all_results = load_results(db_path)
    if not all_results:
        return 0.0, 0, 0

    # group all rows by (question, country)
    groups = defaultdict(list)
    for item in all_results:
        if item.get("difficulty") not in [None, "", "Hard"]:
            continue
        key = (item.get("question", ""), item.get("country", ""))
        groups[key].append(item)

    correct_sets = 0
    total_sets = 0

    for (question, country), items in groups.items():
        # build personas list (one per iteration typically, but keep all)
        personas = [it.get("persona_description") for it in items if it.get("persona_description")]
        if not personas:
            continue
        _, best_persona, _ = await select_best_persona(question, country, personas)

        # Determine language for reasoning output
        if "eng" in mode or "e2l" in mode:
            language = "English"
        else:
            language = country_to_language.get(cap(country), "English")

        # Hard set is correct iff all prompt_options correct
        all_correct = True
        for row in items:
            prompt_option = row.get("prompt_option", "")
            correct_answer = row.get("correct_answer")

            chat_input = [
                {"role": "system", "content": best_persona},
                {"role": "user", "content": (
                    "Is this answer true or false for this question?\n"
                    "You must choose either True or False, and provide a brief explanation.\n"
                    "Respond in valid JSON format with two keys:\n"
                    f"\"correct\" (either \"true\" or \"false\") and "
                    f"\"reasoning\" (a short explanation in {language}).\n"
                    "Example format: {\"correct\": \"{true/false}\", \"reasoning\": \"{reasoning}\"}\n"
                    f"IMPORTANT: The reasoning must be in {language}.\n"
                    f"Question: {question}\n"
                    f"Answer: {prompt_option}"
                )},
            ]

            llm_instance = get_llm()
            _, response = generate_text_funcs[llm_utils.MODEL_NAME](llm_instance, chat_input, enable_thinking_bool=False)
            try:
                parsed = json_repair.loads(response)
                pred = "true" if "true" in str(parsed.get("correct", "")).lower() else "false"
            except Exception:
                all_correct = False
                continue

            expected = "true" if str(correct_answer).lower().strip() in ["1", "true"] else "false"
            if pred != expected:
                all_correct = False

        if all_correct:
            correct_sets += 1
        total_sets += 1

    acc = correct_sets / total_sets if total_sets > 0 else 0.0
    return acc, correct_sets, total_sets


async def run_best_of_n(db_path: str, difficulty: str, mode: str) -> tuple[float, int, int]:
    if difficulty == "Easy":
        return await run_best_of_n_easy(db_path, mode)
    return await run_best_of_n_hard(db_path, mode)


def save_best_of_n_accuracy(db_path: str, difficulty: str, mode: str, accuracy: float, correct: int, total: int):
    """Persist best-of-n accuracy in metadata with iteration=-1."""
    save_accuracy(db_path, iteration=-1, difficulty=difficulty, mode=mode, accuracy=accuracy, correct=correct, total=total)


