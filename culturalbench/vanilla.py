import argparse
import json
import re
from tools.llm_utils import get_llm, generate_text_funcs
from tools import llm_utils
from datasets import load_dataset
from tools.db.db_utils import save_results, save_accuracy
from tqdm.auto import tqdm

best_accuracy_easy = 0
best_accuracy_hard = 0

VANILLA_MAX_TOKENS = 1


def _strip_think(response):
    if not response or "<think>" not in str(response):
        return (response or "").strip()
    text = str(response).strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    start = text.lower().find("<think>")
    if start != -1:
        text = text[:start]
    return text.strip()


def run_vanilla(difficulty, run_num, model_name):
    """Run vanilla (no persona) evaluation for a given difficulty."""
    global best_accuracy_easy, best_accuracy_hard
    print(f"Running vanilla evaluation for {difficulty} (run {run_num+1})...")
    ds = load_dataset("kellycyy/CulturalBench", f"CulturalBench-{difficulty}", split="test")
    llm_instance = get_llm()
    total_correct = 0
    total = 0
    data = {}
    if difficulty.lower() == "hard":
        for i in tqdm(range(0, len(ds), 4)):
            # Reset for each set of 4 questions
            isCorrect = True
            isError = False
            cur_set_data = []
            
            # get basic fields
            cur_question = ds[i]["prompt_question"]
            country = ds[i]["country"]
            for j in range(4):
                cur_option = ds[i + j]["prompt_option"]
                answer = ds[i + j]["answer"]
                try:
                    chat_input = [
                        {"role": "user",
                        "content": (
                            f"Question: {cur_question}"
                            f"Answer: {cur_option}"
                            "Is this answer true or false for this question? You must choose either True or False. Do not output any other text."
                        )}
                    ]
                    _, response = generate_text_funcs[llm_utils.MODEL_NAME](llm_instance, chat_input, max_tokens=VANILLA_MAX_TOKENS)
                    model_answer = _strip_think(response or "").lower().strip()
                    correct_str = "true" if str(answer).strip().lower() in ("1", "true", "yes") else "false"

                    cur_set_data.append({
                        "iteration": 1,
                        "question": cur_question,
                        "prompt_option": cur_option,
                        "correct_answer": answer,
                        "model_answer": model_answer,
                        "vanilla_answer": correct_str,
                        "country": country,
                    })

                    if model_answer == correct_str:
                        continue
                    else:
                        isCorrect = False
                except Exception as e:
                    print(f"Error evaluating question during hard mode: {e}")
                    isError = True
                    break
            
            # Only append if no error occurred and we have 4 items
            if not isError and len(cur_set_data) == 4:
                for j in range(4):
                    data[i + j] = cur_set_data[j]
                if isCorrect:
                    total_correct += 1
                total += 1
    else:
        for i in tqdm(range(len(ds))):
            multi_mode = False
            cur_question = ds[i]["prompt_question"]
            # check if multi-mode question i.e. i, ii, iii, iv
            if "Read the following statements" in cur_question:
                multi_mode = True
            prompt_option_a = ds[i]["prompt_option_a"]
            prompt_option_b = ds[i]["prompt_option_b"]
            prompt_option_c = ds[i]["prompt_option_c"]
            prompt_option_d = ds[i]["prompt_option_d"]
            answer = ds[i]["answer"]
            country = ds[i]["country"]
            try:
                chat_input = [
                    {"role": "user",
                    "content": (
                        "To answer the following multiple-choice question, you should choose one option only among A, B, C, D.\n"
                        "Instruction: You must select one option among A, B, C, D.\n"
                        "Only output the letter. Do not output any other things.\n"
                        + ("Select the options with all applicable statements.\n" if multi_mode else "")
                        + f"Question: {cur_question}\n"
                        f"A. {prompt_option_a}\n"
                        f"B. {prompt_option_b}\n"
                        f"C. {prompt_option_c}\n"
                        f"D. {prompt_option_d}\n"
                    )}
                ]
                _, response = generate_text_funcs[llm_utils.MODEL_NAME](llm_instance, chat_input, max_tokens=VANILLA_MAX_TOKENS)
                model_answer = _strip_think(response or "").strip().upper()
                correct_letter = str(answer).strip().upper()
                if len(correct_letter) > 1:
                    correct_letter = correct_letter[0]
                if model_answer == correct_letter:
                    total_correct += 1
                total += 1
            except Exception as e:
                print(f"Error evaluating question during easy mode: {e}")
                continue
            data[i] = {
                "iteration": 1,
                "question": cur_question,
                "options": {"A": prompt_option_a, "B": prompt_option_b, "C": prompt_option_c, "D": prompt_option_d},
                "correct_answer": answer,
                "model_answer": model_answer,
                "vanilla_answer": correct_letter,
                "country": country,
            }

    db_path = f"../results/vanilla/{model_name}/{difficulty.lower()}_{run_num+1}.db"
    save_results(db_path, data, difficulty, "vanilla")
    accuracy = total_correct / total if total > 0 else 0
    print(f"Run {run_num+1} accuracy ({difficulty}): {accuracy}")
    if difficulty.lower() == "easy":
        if accuracy > best_accuracy_easy:
            best_accuracy_easy = accuracy
    else:
        if accuracy > best_accuracy_hard:
            best_accuracy_hard = accuracy
    save_accuracy(db_path, 1, difficulty, "vanilla", accuracy, total_correct, total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vanilla (no persona) evaluation via SGLang.")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="mistral-3-14b-instruct-2512",
    )
    parser.add_argument("--difficulty", type=str, choices=["easy", "hard", "Easy", "Hard", "both"],
                        default="both", help="Difficulty: easy, hard, or both (default: both)")
    args = parser.parse_args()
    if args.model not in generate_text_funcs:
        raise SystemExit(f"Unknown model: {args.model}. Choose from: {list(generate_text_funcs.keys())}")
    llm_utils.MODEL_NAME = args.model
    diff = args.difficulty.capitalize()
    
    for i in range(5):
        if diff == "Both":
            run_vanilla("Easy", i, args.model)
            run_vanilla("Hard", i, args.model)
        else:
            run_vanilla(diff, i, args.model)
    
    print(f"Best accuracy for Easy: {best_accuracy_easy}")
    print(f"Best accuracy for Hard: {best_accuracy_hard}")