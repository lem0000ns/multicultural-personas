"""
Run qwen3-32b (or another model) with the exact same personas used for qwen3-14b
in a given mode + difficulty. For each iteration 1..5, load that iteration's personas from the
source DB, run the answer model with them, and save accuracy for that iteration.
Output: one score per iteration (iter 1, 2, 3, 4, 5).

Usage (from culturalbench/):
  python run_foreign_personas.py --mode eng --difficulty Easy
  python run_foreign_personas.py --mode eng --difficulty Hard --persona_source_model qwen3_14b --answer_model qwen3_32b
"""
import argparse
import json
import os
import sys
import json_repair
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.db.db_utils import load_results, save_results, save_accuracy, init_db
from tools import llm_utils
from tools.llm_utils import get_llm, generate_text_funcs
from tools.utils import country_to_language

MODEL_ID_TO_FULL = {
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3_32b": "mistral-3-14b-instruct-2512",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}

MODEL_ID_TO_FOLDER = {
    "qwen3_4b": "qwen3-4b",
    "qwen3_14b": "qwen3-14b",
    "qwen3_32b": "qwen3-32b",
    "llama3_8b": "llama3-8b-instruct",
}

TEMPERATURE = 0.6
MAX_ITERATION = 5  # Run for iteration 1, 2, 3, 4, 5


def cap(s):
    if not s:
        return s
    return " ".join(w.capitalize() for w in str(s).split())


def run_easy(persona_rows, mode, answer_model_name, current_iteration):
    new_data = {}
    correct = total = 0
    for i, item in tqdm(enumerate(persona_rows), total=len(persona_rows), desc=f"Easy iter{current_iteration}", unit="q"):
        persona = item.get("persona_description") or ""
        question = item.get("question") or ""
        options = item.get("options")
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = {}
        options = options or {}
        option_a, option_b = options.get("A", ""), options.get("B", "")
        option_c, option_d = options.get("C", ""), options.get("D", "")
        correct_answer = (item.get("correct_answer") or "").strip()
        country = item.get("country") or ""
        language = "English" if ("eng" in mode or "e2l" in mode) else country_to_language.get(cap(country), "English")

        chat_input = [
            {"role": "system", "content": persona},
            {"role": "user", "content": (
                "Instruction: You must select one option among A,B,C,D.\n"
                "Respond in valid JSON format with two keys: \n"
                '"answer" (either "A", "B", "C", or "D") and "reasoning" (a short explanation in ' + language + "). \n"
                'Example format: {"answer": "A", "reasoning": "..."}\n'
                f"IMPORTANT: The reasoning must be in {language}.\n"
                f"\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}"
            )},
        ]
        try:
            llm_instance = get_llm()
            thinking_content, response = generate_text_funcs[answer_model_name](llm_instance, chat_input, enable_thinking_bool=False)
            result = json_repair.loads(response)
            response_answer = (result.get("answer") or "").upper().strip()
            reasoning = (result.get("reasoning") or "").strip()
        except Exception as e:
            print(f"Error Easy q {i}: {e}")
            response_answer, reasoning, thinking_content = "", "", None

        new_data[i] = {
            "iteration": current_iteration,
            "question": question,
            "persona_description": persona,
            "correct_answer": correct_answer,
            "model_answer": response_answer,
            "reasoning": reasoning,
            "country": country,
            "options": options,
            "thinking_content": thinking_content,
        }
        if response_answer and correct_answer and response_answer[0] == correct_answer[0]:
            correct += 1
        total += 1
    return new_data, correct, total


def run_hard(persona_rows, mode, answer_model_name, current_iteration):
    new_data = {}
    correct = total = 0
    for idx in tqdm(range(0, len(persona_rows), 4), total=len(persona_rows) // 4, desc=f"Hard iter{current_iteration}", unit="set"):
        if idx + 4 > len(persona_rows):
            break
        chunk = persona_rows[idx : idx + 4]
        persona = chunk[0].get("persona_description") or ""
        question = chunk[0].get("question") or ""
        country = chunk[0].get("country") or ""
        language = "English" if ("eng" in mode or "e2l" in mode) else country_to_language.get(cap(country), "English")

        set_correct = True
        for j, row in enumerate(chunk):
            prompt_option = row.get("prompt_option") or ""
            correct_answer = row.get("correct_answer")
            expected = "true" if str(correct_answer).lower().strip() in ("1", "true", "yes") else "false"
            chat_input = [
                {"role": "system", "content": persona},
                {"role": "user", "content": (
                    "Is this answer true or false for this question?\n"
                    "You must choose either True or False, and provide a brief explanation.\n"
                    "Respond in valid JSON format with two keys: \"correct\" (\"true\" or \"false\") and \"reasoning\".\n"
                    f"IMPORTANT: The reasoning must be in {language}.\n"
                    f"Question: {question}\nAnswer: {prompt_option}"
                )},
            ]
            try:
                llm_instance = get_llm()
                thinking_content, response = generate_text_funcs[answer_model_name](llm_instance, chat_input, enable_thinking_bool=False)
                result = json_repair.loads(response)
                thinks = "true" if "true" in (result.get("correct") or "").lower().strip() else "false"
                reasoning = (result.get("reasoning") or "").strip()
            except Exception as e:
                print(f"Error Hard set {idx//4} opt {j}: {e}")
                thinks, reasoning, thinking_content = "false", "", None
            if thinks != expected:
                set_correct = False
            new_data[idx + j] = {
                "iteration": current_iteration,
                "question": question,
                "persona_description": persona,
                "correct_answer": correct_answer,
                "model_answer": thinks,
                "reasoning": reasoning,
                "country": country,
                "prompt_option": prompt_option,
                "thinking_content": thinking_content,
            }
        if set_correct:
            correct += 1
        total += 1
    return new_data, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="eng, ling, l2e, or e2l")
    parser.add_argument("--difficulty", type=str, required=True, help="Easy or Hard")
    parser.add_argument("--persona_source_model", type=str, default="qwen3_14b")
    parser.add_argument("--answer_model", type=str, default="qwen3_32b")
    parser.add_argument("--results_root", type=str, default="../results")
    args = parser.parse_args()

    difficulty = args.difficulty.capitalize()
    diff_lower = difficulty.lower()
    t = TEMPERATURE
    results_root = os.path.abspath(args.results_root)

    source_folder = MODEL_ID_TO_FOLDER.get(args.persona_source_model) or args.persona_source_model.replace("_", "-")
    answer_folder = MODEL_ID_TO_FOLDER.get(args.answer_model) or args.answer_model.replace("_", "-")
    source_db = os.path.join(results_root, args.mode, source_folder, f"{diff_lower}_t{t}_{args.persona_source_model}.db")
    if not os.path.isfile(source_db):
        print("Source DB not found:", source_db)
        sys.exit(1)

    answer_model_full = MODEL_ID_TO_FULL.get(args.answer_model)
    if not answer_model_full or answer_model_full not in generate_text_funcs:
        print("Unknown answer model:", args.answer_model)
        sys.exit(1)

    out_dir = os.path.join(results_root, args.mode, answer_folder)
    os.makedirs(out_dir, exist_ok=True)
    out_db = os.path.join(out_dir, f"{diff_lower}_t{t}_{args.persona_source_model}_personas.db")

    llm_utils.MODEL_NAME = answer_model_full
    llm_utils.TEMPERATURE = t
    print("Source DB:", source_db)
    print("Answer model:", answer_model_full)
    print("Output DB:", out_db)
    print(f"Running iterations 1..{MAX_ITERATION} (load personas per iteration, run answer model)\n")

    init_db(out_db)
    for iteration in range(1, MAX_ITERATION + 1):
        persona_rows = load_results(source_db, iteration=iteration)
        if not persona_rows:
            print(f"Iteration {iteration}: no rows in source DB, skipping.")
            continue
        if difficulty == "Easy":
            new_data, correct, total = run_easy(persona_rows, args.mode, answer_model_full, iteration)
        else:
            new_data, correct, total = run_hard(persona_rows, args.mode, answer_model_full, iteration)
        save_results(out_db, new_data, difficulty, args.mode)
        acc = correct / total if total else 0
        save_accuracy(out_db, iteration, difficulty, args.mode, acc, correct, total)
        print(f"Iteration {iteration}: accuracy = {acc:.4f}  ({correct}/{total})")
    print("\nSaved to:", out_db)


if __name__ == "__main__":
    main()
