from tools.llm_utils import get_llm, generate_text_funcs, MODEL_NAME
from datasets import load_dataset
from tools.db.db_utils import save_results, save_accuracy

def run_vanilla(difficulty):
    """Run vanilla (no persona) evaluation for a given difficulty."""
    ds = load_dataset("kellycyy/CulturalBench", f"CulturalBench-{difficulty}", split="test")
    llm_instance = get_llm()
    total_correct = 0
    total = 0
    data = {}
    if difficulty.lower() == "hard":
        for i in range(0, len(ds), 4):
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
                            "Is this answer true or false for this question? You must choose either True or False."
                        )}
                    ]
                    response = generate_text_funcs[MODEL_NAME](llm_instance, chat_input, max_tokens=1)

                    cur_set_data.append({"question": cur_question, "prompt_option": cur_option, "correct_answer": answer, "vanilla_answer": response, "country": country})

                    if str(response).strip().lower() == str(answer).strip().lower():
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
        for i in range(len(ds)):
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
                        "Do not output any other things.\n"
                        + ("Select the options with all applicable statements.\n" if multi_mode else "")
                        + f"Question: {cur_question}\n"
                        f"A. {prompt_option_a}\n"
                        f"B. {prompt_option_b}\n"
                        f"C. {prompt_option_c}\n"
                        f"D. {prompt_option_d}\n"
                    )}
                ]
                response = generate_text_funcs[MODEL_NAME](llm_instance, chat_input, max_tokens=1)
                if str(response).strip().lower() == str(answer).strip().lower():
                    total_correct += 1
                total += 1
            except Exception as e:
                print(f"Error evaluating question during easy mode: {e}")
                continue
            data[i] = {"question": cur_question, "prompt_option_a": prompt_option_a, "prompt_option_b": prompt_option_b, "prompt_option_c": prompt_option_c, "prompt_option_d": prompt_option_d, "correct_answer": answer, "vanilla_answer": response, "country": country}

    db_path = f"../results/vanilla/{difficulty.lower()}.db"
    for entry in data.values():
        save_results(db_path, entry, difficulty, "vanilla")
    accuracy = total_correct / total if total > 0 else 0
    save_accuracy(db_path, 1, difficulty, "vanilla", accuracy, total_correct, total)

if __name__ == "__main__":
    run_vanilla("Easy")
    run_vanilla("Hard")