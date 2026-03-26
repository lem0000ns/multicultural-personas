"""Functions for running iterative persona refinement."""

import asyncio
import json
from tqdm.auto import tqdm
from persona_generator import generate_new_persona, cap
from tools.utils import country_to_language
from tools.llm_utils import get_llm, generate_text_funcs, async_generate, get_external_feedback
from tools import llm_utils
from tools.db.db_utils import save_results, save_accuracy, load_previous_iteration
from token_counter import add_input_tokens, add_output_tokens
import json_repair


def _extract_revised_persona_text(value):
    """Best-effort: if value is a JSON string/dict with revised_persona, return that field; else return value."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("revised_persona") or value.get("persona_description") or value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and ("revised_persona" in s or "\"revised_persona\"" in s):
            try:
                obj = json_repair.loads(s)
                if isinstance(obj, dict) and "revised_persona" in obj:
                    return obj["revised_persona"]
            except Exception:
                return value
        return value
    return value


def _format_easy_options_and_answer(item):
    """Format Easy-mode options + model answer without revealing correct answer."""
    options = item.get("options") or {}
    lines = []
    for opt in ["A", "B", "C", "D"]:
        if opt in options:
            lines.append(f"{opt}: {options[opt]}")
    model_answer = item.get("model_answer")
    if model_answer is not None:
        lines.append(f"Model answer: {model_answer}")
    return "\n".join(lines).strip()


def append_to_db(db_path, new_data, correct, total, iteration, difficulty, mode):
    """Append iteration data to database."""
    save_results(db_path, new_data, difficulty, mode)
    accuracy = correct / total if total > 0 else 0
    save_accuracy(db_path, iteration, difficulty, mode, accuracy, correct, total)
    print(f"Iteration {iteration} Accuracy: {accuracy}")
    return accuracy


async def _process_easy_iter_one(i, item, mode, cur_iteration, is_translation_mode, external, sem):
    """Process a single Easy-mode question in an iteration. Returns (index, base_data, is_correct) or None."""
    async with sem:
        try:
            old_persona = (
                item["persona_description"]
                if "l2e" not in mode and "e2l" not in mode
                else _extract_revised_persona_text(item.get("pretranslated_persona"))
            )
            prev_answers = _format_easy_options_and_answer(item)
            previous_personas_data = {
                'persona': old_persona,
                'model_answer': prev_answers,
                'reasoning': item["reasoning"]
            }

            feedback = None
            if external:
                if "e2l" in mode:
                    feedback_language = "English"
                    persona_for_feedback = _extract_revised_persona_text(item.get("pretranslated_persona")) or old_persona
                elif "ling" in mode or "l2e" in mode:
                    feedback_language = country_to_language[cap(item["country"])].capitalize()
                    persona_for_feedback = old_persona
                else:
                    feedback_language = "English"
                    persona_for_feedback = old_persona
                feedback = await get_external_feedback("Easy", item["question"], persona_for_feedback, prev_answers, feedback_language=feedback_language)

            pretranslated, refine_response = await generate_new_persona(
                "Easy", item["question"], previous_personas_data, mode, item["country"], feedback,
            )
            if refine_response is None and is_translation_mode:
                return None

            result = None
            pretranslated_persona_text = None
            if is_translation_mode:
                result = json_repair.loads(refine_response)
                pretranslated_persona_text = _extract_revised_persona_text(pretranslated)
            else:
                result = json_repair.loads(pretranslated)

            new_persona = result["revised_persona"]
            refine_reasoning = result["reasoning"]

        except Exception as e:
            print(f"Error generating/parsing persona for question {i}: {type(e).__name__}: {str(e)}")
            return None

        prompt_question = item["question"]
        option_a = item["options"]["A"]
        option_b = item["options"]["B"]
        option_c = item["options"]["C"]
        option_d = item["options"]["D"]

        country = cap(item["country"])

        if "eng" in mode or "e2l" in mode:
            language = "English"
        else:
            language = country_to_language[country]

        chat_input = [
            {"role": "system", "content": new_persona},
            {"role": "user", "content": (
                "Instruction: You must select one option among A,B,C,D.\n"
                "Respond in valid JSON format with two keys: \n"
                f"\"answer\" (either \"A\", \"B\", \"C\", or \"D\") and "
                f"\"reasoning\" (a short explanation in {language}). \n"
                "Example format: {\"answer\": \"{A/B/C/D}\", \"reasoning\": \"{reasoning}\"}\n"
                f"IMPORTANT: The reasoning must be in {language}.\n"
                f"Question: {prompt_question}\n"
                f"A. {option_a}\n"
                f"B. {option_b}\n"
                f"C. {option_c}\n"
                f"D. {option_d}"
            )}
        ]

        add_input_tokens("Easy", mode, chat_input)
        llm_instance = get_llm()
        thinking_content, response = await async_generate(llm_instance, chat_input, enable_thinking_bool=False)
        out_text = (thinking_content or "") + "\n" + (response or "")
        add_output_tokens("Easy", mode, out_text)

        try:
            result = json_repair.loads(response)
        except Exception:
            print(f"Error sanitizing JSON for response: {response}")
            return None

        try:
            response_answer = result["answer"].upper().strip()
            reasoning = result["reasoning"].strip()
            correct_answer = item["correct_answer"]

            options_dict = {"A": option_a, "B": option_b, "C": option_c, "D": option_d}

            base_data = {
                "question": prompt_question,
                "options": options_dict,
                "persona_description": new_persona,
                "refine_reasoning": refine_reasoning,
                "correct_answer": correct_answer,
                "model_answer": response_answer,
                "reasoning": reasoning,
                "country": item["country"],
                "iteration": cur_iteration
            }

            if "l2e" in mode or "e2l" in mode:
                base_data["pretranslated_persona"] = pretranslated_persona_text
            if thinking_content is not None:
                base_data["thinking_content"] = thinking_content

            is_correct = response_answer.upper() == correct_answer.upper()
            return (i, base_data, is_correct)
        except Exception as e:
            print(f"Error generating answer for question {i}: {type(e).__name__}: {str(e)}")
            return None


async def run_easy_iterations(mode, num_iterations, db_path, start_iteration=2, external=False):
    """Run iterations for Easy difficulty."""
    accuracies = []
    is_translation_mode = "e2l" in mode or "l2e" in mode

    for cur_iteration in range(start_iteration, num_iterations + 1):
        data = load_previous_iteration(db_path, cur_iteration)
        print(f"Currently running iteration {cur_iteration}")
        sem = asyncio.Semaphore(llm_utils.MAX_CONCURRENT)

        tasks = [
            _process_easy_iter_one(i, item, mode, cur_iteration, is_translation_mode, external, sem)
            for i, item in enumerate(data)
        ]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Iter {cur_iteration} (Easy)", unit="q"):
            results.append(await coro)

        new_data = {}
        correct = total = 0
        for r in results:
            if r is None:
                continue
            idx, base_data, is_correct = r
            new_data[idx] = base_data
            if is_correct:
                correct += 1
            total += 1

        accuracy = append_to_db(db_path, new_data, correct, total, cur_iteration, "Easy", mode)
        accuracies.append(accuracy)

    return accuracies


async def _process_hard_iter_set(i, data, mode, cur_iteration, is_translation_mode, external, sem):
    """Process a single Hard-mode question set (4 sub-questions) in an iteration. Returns (set_data, is_correct) or None."""
    async with sem:
        prompt_question = data[i]["question"]

        try:
            old_persona = (
                data[i]["persona_description"]
                if "l2e" not in mode and "e2l" not in mode
                else _extract_revised_persona_text(data[i].get("pretranslated_persona"))
            )
            previous_personas_data = {
                'persona': old_persona,
                'reasoning': data[i]["reasoning"],
                'iteration': cur_iteration,
            }

            feedback = None
            if external:
                if "e2l" in mode:
                    feedback_language = "English"
                    persona_for_feedback = _extract_revised_persona_text(data[i].get("pretranslated_persona")) or old_persona
                elif "ling" in mode or "l2e" in mode:
                    feedback_language = country_to_language[cap(data[i]["country"])].capitalize()
                    persona_for_feedback = old_persona
                else:
                    feedback_language = "English"
                    persona_for_feedback = old_persona
                feedback = await get_external_feedback("Hard", prompt_question, persona_for_feedback, None, feedback_language=feedback_language)

            pretranslated, refine_response = await generate_new_persona(
                "Hard", prompt_question, previous_personas_data, mode, data[i]["country"], feedback,
            )
            if refine_response is None and is_translation_mode:
                return None

            result = None
            pretranslated_persona_text = None
            if is_translation_mode:
                result = json_repair.loads(refine_response)
                pretranslated_persona_text = _extract_revised_persona_text(pretranslated)
            else:
                result = json_repair.loads(pretranslated)

            new_persona = result["revised_persona"]
            refine_reasoning = result["reasoning"]
        except Exception as e:
            print(f"Error generating/parsing persona for question set {i//4}: {type(e).__name__}: {str(e)}")
            return None

        if "eng" in mode or "e2l" in mode:
            language = "English"
        else:
            language = country_to_language[cap(data[i]["country"])].capitalize()

        isCorrect = True
        cur_set_data = []
        for j in range(4):
            prompt_option = data[i + j]["prompt_option"]
            correct_answer = data[i + j]["correct_answer"]

            chat_input = [
                {"role": "system", "content": new_persona},
                {"role": "user", "content": (
                    "Is this answer true or false for this question?\n"
                    "You must choose either True or False, and provide a brief "
                    "explanation for your answer.\n"
                    "Respond in valid JSON format with two keys: \n"
                    f"\"correct\" (either \"true\" or \"false\") and "
                    f"\"reasoning\" (a short explanation in {language}). \n"
                    "Example format: {\"correct\": \"{true/false}\", \"reasoning\": \"{reasoning}\"}\n"
                    f"IMPORTANT: The reasoning must be in {language}.\n"
                    f"Question: {prompt_question}\n"
                    f"Answer: {prompt_option}"
                )}
            ]

            add_input_tokens("Hard", mode, chat_input)
            llm_instance = get_llm()
            thinking_content, response = await async_generate(llm_instance, chat_input, enable_thinking_bool=False)
            out_text = (thinking_content or "") + "\n" + (response or "")
            add_output_tokens("Hard", mode, out_text)
            try:
                result = json_repair.loads(response)
                thinks_correct = (
                    "true"
                    if "true" in result["correct"].lower().strip()
                    else "false"
                )
                reasoning = result["reasoning"].strip()
            except Exception as e:
                print(f"Error generating answer for option {j} in question set {i//4}: {type(e).__name__}: {str(e)}")
                return None

            item_data = {
                "question": prompt_question,
                "prompt_option": prompt_option,
                "persona_description": new_persona,
                "refine_reasoning": refine_reasoning,
                "correct_answer": correct_answer,
                "model_answer": thinks_correct,
                "reasoning": reasoning,
                "country": data[i + j]["country"],
                "iteration": cur_iteration
            }

            if "l2e" in mode or "e2l" in mode:
                item_data["pretranslated_persona"] = pretranslated_persona_text
            if thinking_content is not None:
                item_data["thinking_content"] = thinking_content

            cur_set_data.append(item_data)

            correct_str = str(correct_answer).lower().strip()
            expected_answer = "true" if correct_str in ["1", "true"] else "false"
            if str(thinks_correct).lower() != expected_answer:
                isCorrect = False

        if len(cur_set_data) != 4:
            return None

        set_data = {i + j: cur_set_data[j] for j in range(4)}
        return (set_data, isCorrect)


async def run_hard_iterations(mode, num_iterations, db_path, start_iteration=2, external=False):
    """Run iterations for Hard difficulty."""
    accuracies = []
    is_translation_mode = "e2l" in mode or "l2e" in mode

    for cur_iteration in range(start_iteration, num_iterations + 1):
        data = load_previous_iteration(db_path, cur_iteration)
        print(f"Currently running iteration {cur_iteration} (Hard)")
        sem = asyncio.Semaphore(llm_utils.MAX_CONCURRENT)
        set_indices = list(range(0, len(data), 4))

        tasks = [
            _process_hard_iter_set(i, data, mode, cur_iteration, is_translation_mode, external, sem)
            for i in set_indices
        ]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Iter {cur_iteration} (Hard)", unit="set"):
            results.append(await coro)

        new_data = {}
        correct = total = 0
        for r in results:
            if r is None:
                continue
            set_data, is_correct = r
            new_data.update(set_data)
            if is_correct:
                correct += 1
            total += 1

        accuracy = append_to_db(db_path, new_data, correct, total, cur_iteration, "Hard", mode)
        accuracies.append(accuracy)

    return accuracies


async def run_iterations(mode, num_iterations, difficulty, db_path, start_iteration=2, external=False):
    """Run iterations starting from iteration 2."""
    if difficulty == "Easy":
        return await run_easy_iterations(mode, num_iterations, db_path, start_iteration, external)
    else:
        return await run_hard_iterations(mode, num_iterations, db_path, start_iteration, external)
