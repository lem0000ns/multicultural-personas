"""Evaluation functions for initial persona generation and testing."""

import asyncio
import os
from datasets import load_dataset
from tqdm.auto import tqdm
from persona_generator import generate_persona_description, cap
from tools.utils import country_to_language
from tools.llm_utils import (
    get_llm,
    generate_text_funcs,
    async_generate,
    GEMMA3_12B_SGLANG_MODEL_ID,
    LEGACY_QWEN3_06B_SGLANG_MODEL_ID,
    LEGACY_MISTRAL_SGLANG_MODEL_ID,
    MAX_CONCURRENT,
)
from tools import llm_utils
from tools.db.db_utils import save_results, save_accuracy
import json_repair
from token_counter import add_input_tokens, add_output_tokens, get_model_folder


def is_valid_set(ds, i):
    """Check if a set of 4 options is complete and valid.

    Args:
        ds: Dataset
        i: Starting index

    Returns:
        True if all 4 options are valid, False otherwise
    """
    for j in range(4):
        cur_row = ds[i + j]
        prompt_question = cur_row["prompt_question"]
        prompt_option = cur_row["prompt_option"]
        prompt_answer = cur_row["answer"]
        country = cur_row["country"]
        if prompt_option is None or prompt_answer is None or prompt_question is None or country is None:
            return False
    return True


async def _process_hard_set(i, ds, mode, difficulty, sem):
    """Process a single Hard-mode question set (4 sub-questions). Returns (set_data, is_correct) or None."""
    async with sem:
        if not is_valid_set(ds, i):
            return None

        isCorrect = True
        cur_set_data = []
        persona_description = None
        pretranslated = None

        for j in range(4):
            cur_row = ds[i + j]
            prompt_question = cur_row["prompt_question"]
            prompt_option = cur_row["prompt_option"]
            prompt_answer = cur_row["answer"]
            country = cur_row["country"]

            if "eng" in mode or "e2l" in mode:
                language = "English"
            else:
                language = country_to_language[cap(country)].capitalize()

            # use same persona description for same question (4 at a time)
            if j == 0:
                pretranslated, translated = await generate_persona_description(
                    prompt_question, country, mode, difficulty,
                )
                if "l2e" in mode or "e2l" in mode:
                    persona_description = translated
                else:
                    persona_description = pretranslated
                if persona_description is None:
                    return None

            thinking_instruction = ""
            if "ling" in mode and llm_utils.MODEL_NAME == "Qwen/Qwen3-4B":
                thinking_instruction = f"You MUST write internal reasoning inside <think>...</think> in {language}. If any part of <think>...</think> is not {language}, regenerate the reasoning.\n\n"

            chat_input = [
                {"role": "system", "content": persona_description},
                {"role": "user", "content": (
                    thinking_instruction +
                    "Is this answer true or false for this question?\n"
                    "You must choose either True or False, and provide a brief "
                    "explanation for your answer.\n"
                    "Respond in valid JSON format with two keys: \n"
                    f"\"correct\" (either \"true\" or \"false\") and "
                    f"\"reasoning\" (a short, brief explanation in {language}). \n"
                    "Example format: {\"correct\": \"{true/false}\", \"reasoning\": \"{reasoning}\"}\n"
                    f"IMPORTANT: The reasoning must be in {language}.\n"
                    f"IMPORTANT: DO NOT output any other text than the JSON response.\n"
                    f"Question: {prompt_question}\n"
                    f"Answer: {prompt_option}"
                )}
            ]

            add_input_tokens(difficulty, mode, chat_input)
            llm_instance = get_llm()
            thinking_content, response = await async_generate(llm_instance, chat_input, enable_thinking_bool=False)
            out_text = (thinking_content or "") + "\n" + (response or "")
            add_output_tokens(difficulty, mode, out_text)

            try:
                result = json_repair.loads(response)
                thinks_correct = (
                    "true"
                    if "true" in result["correct"].lower().strip()
                    else "false"
                )
                reasoning = result["reasoning"].strip()
            except Exception as e:
                print(f"Error parsing response for set {i//4} option {j}: {response} {e}")
                return None

            item_data = {
                "question": prompt_question,
                "prompt_option": prompt_option,
                "persona_description": persona_description,
                "correct_answer": prompt_answer,
                "model_answer": thinks_correct,
                "reasoning": reasoning,
                "country": country,
                "iteration": 1
            }

            if "l2e" in mode or "e2l" in mode:
                item_data["pretranslated_persona"] = pretranslated
            if thinking_content is not None:
                item_data["thinking_content"] = thinking_content

            cur_set_data.append(item_data)

            correct_str = str(prompt_answer).lower().strip()
            expected_answer = "true" if correct_str in ["1", "true"] else "false"
            if str(thinks_correct).lower() != expected_answer:
                isCorrect = False

        if len(cur_set_data) != 4:
            return None

        set_data = {i + j: cur_set_data[j] for j in range(4)}
        return (set_data, isCorrect)


async def evaluate_hard_initial(ds, mode, difficulty="Hard"):
    sem = asyncio.Semaphore(llm_utils.MAX_CONCURRENT)
    set_indices = [i for i in range(0, len(ds), 4)]

    tasks = [_process_hard_set(i, ds, mode, difficulty, sem) for i in set_indices]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Initial eval (Hard)", unit="set"):
        results.append(await coro)

    data = {}
    correct = total = 0
    for r in results:
        if r is None:
            continue
        set_data, is_correct = r
        data.update(set_data)
        if is_correct:
            correct += 1
        total += 1

    return data, correct, total


async def _process_easy_one(i, cur_row, mode, difficulty, sem):
    """Process a single Easy-mode question. Returns (index, item_data, is_correct) or None."""
    async with sem:
        prompt_question = cur_row["prompt_question"]
        option_a = cur_row["prompt_option_a"]
        option_b = cur_row["prompt_option_b"]
        option_c = cur_row["prompt_option_c"]
        option_d = cur_row["prompt_option_d"]
        answer = cur_row["answer"]
        country = cur_row["country"]

        if (option_a is None or option_b is None or option_c is None or
            option_d is None or answer is None or prompt_question is None or
            country is None):
            return None

        pretranslated, translated = await generate_persona_description(
            prompt_question, country, mode, difficulty,
        )
        if "l2e" in mode or "e2l" in mode:
            persona_description = translated
        else:
            persona_description = pretranslated

        if persona_description is None:
            return None

        if "eng" in mode or "e2l" in mode:
            language = "English"
        else:
            language = country_to_language[cap(country)].capitalize()

        chat_input = [
            {"role": "system", "content": persona_description},
            {"role": "user", "content": (
                "Instruction: You must select one option among A,B,C,D.\n"
                "Respond in valid JSON format with two keys: \n"
                f"\"answer\" (either \"A\", \"B\", \"C\", or \"D\") and "
                f"\"reasoning\" (a short explanation in {language}). \n"
                "Example format: {\"answer\": \"{A/B/C/D}\", \"reasoning\": \"{reasoning}\"}\n"
                f"IMPORTANT: The reasoning must be in {language}.\n"
                f"\nQuestion: {prompt_question}\n"
                f"A. {option_a}\n"
                f"B. {option_b}\n"
                f"C. {option_c}\n"
                f"D. {option_d}"
            )}
        ]

        add_input_tokens(difficulty, mode, chat_input)
        llm_instance = get_llm()
        thinking_content, response = await async_generate(llm_instance, chat_input, enable_thinking_bool=False)
        out_text = (thinking_content or "") + "\n" + (response or "")
        add_output_tokens(difficulty, mode, out_text)
        try:
            result = json_repair.loads(response)
            response_answer = result["answer"].upper().strip()
            reasoning = result["reasoning"].strip()

            options_dict = {"A": option_a, "B": option_b, "C": option_c, "D": option_d}

            item_data = {
                "question": prompt_question,
                "options": options_dict,
                "persona_description": persona_description,
                "correct_answer": answer,
                "model_answer": response_answer,
                "reasoning": reasoning,
                "country": country,
                "iteration": 1
            }

            if "l2e" in mode or "e2l" in mode:
                item_data["pretranslated_persona"] = pretranslated
            if thinking_content is not None:
                item_data["thinking_content"] = thinking_content

            is_correct = response_answer.upper() == answer.upper()
            return (i, item_data, is_correct)
        except Exception:
            print("Error parsing response: " + response)
            return None


async def evaluate_easy_initial(ds, mode, difficulty="Easy"):
    sem = asyncio.Semaphore(llm_utils.MAX_CONCURRENT)
    n_questions = len(ds)

    tasks = [_process_easy_one(i, ds[i], mode, difficulty, sem) for i in range(n_questions)]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Initial eval (Easy)", unit="q"):
        results.append(await coro)

    data = {}
    correct = total = 0
    for r in results:
        if r is None:
            continue
        idx, item_data, is_correct = r
        data[idx] = item_data
        if is_correct:
            correct += 1
        total += 1

    return data, correct, total


async def run_initial_eval(difficulty, mode, custom=None, max_questions=None):
    """Run initial evaluation (i1) for the given difficulty.

    Args:
        difficulty: "Easy" or "Hard"
        mode: Mode (eng, ling, l2e, or e2l)
        custom: Optional custom suffix to append to database path
        max_questions: If set, only the first N questions are evaluated. For Hard mode, one
            "question" is a full T/F set (4 rows in the dataset).

    Returns:
        Tuple of (accuracy, db_path)
    """
    print(f"Loading CulturalBench dataset ({difficulty})...")
    ds = load_dataset("kellycyy/CulturalBench", f"CulturalBench-{difficulty}", split="test")
    if max_questions is not None and max_questions > 0:
        if difficulty == "Hard":
            n_rows = min(max_questions * 4, len(ds))
        else:
            n_rows = min(max_questions, len(ds))
        ds = ds.select(range(n_rows))
        print(f"Subset: first {max_questions} question(s) ({len(ds)} rows). Starting evaluation...")
    else:
        print(f"Dataset loaded ({len(ds)} examples). Starting evaluation...")

    if difficulty == "Hard":
        data, correct, total = await evaluate_hard_initial(ds, mode, difficulty)
    else:
        data, correct, total = await evaluate_easy_initial(ds, mode, difficulty)

    model_to_save = {
        "Qwen/Qwen3-32B": "qwen3_32b",
        "google/gemma-2-27b-it": "gemma2_27b",
        "meta-llama/Llama-3.3-70B-Instruct": "llama33_70b",
        "Qwen/Qwen3-4B": "qwen3_4b",
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b",
        "Qwen/Qwen3-14B": "qwen3_14b",
        GEMMA3_12B_SGLANG_MODEL_ID: "gemma3_12b",
        LEGACY_QWEN3_06B_SGLANG_MODEL_ID: "gemma3_12b",
        LEGACY_MISTRAL_SGLANG_MODEL_ID: "gemma3_12b",
        "Qwen/Qwen3.5-35B-A3B": "qwen3.5_35b",
        "zai-org/GLM-4-9B-0414": "glm4_9b",
    }
    model_folder = get_model_folder(llm_utils.MODEL_NAME)
    # write results to database: results/{mode}/{model}/{file}
    db_path = f"../results/{mode}/{model_folder}/{difficulty.lower()}_t{llm_utils.TEMPERATURE}_{model_to_save[llm_utils.MODEL_NAME]}"
    if custom:
        db_path += f"_{custom}"
    db_path += ".db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    save_results(db_path, data, difficulty, mode)

    accuracy = correct / total if total > 0 else 0
    save_accuracy(db_path, 1, difficulty, mode, accuracy, correct, total)

    print(f"Initial Persona Accuracy for {difficulty}: {accuracy}")
    return accuracy, db_path
