"""Main script for running iterative persona refinement on CulturalBench."""

import argparse
import asyncio
import os
from evaluators import run_initial_eval
from iteration_runner import run_iterations
from tools.llm_utils import cleanup
from tools.db.db_utils import load_results, get_all_iterations
from token_counter import write_to_json, get_totals, reset
import tools.llm_utils
from tools import llm_utils

def calculate_accuracy_from_db(db_path, iteration, difficulty):
    """Calculate accuracy for a given iteration from database.
    
    Args:
        db_path: Path to database file
        iteration: Iteration number
        difficulty: "Easy" or "Hard"
    
    Returns:
        Accuracy value
    """
    results = load_results(db_path, iteration=iteration)
    total = len(results) if difficulty == "Easy" else len(results) / 4
    correct = 0
    
    if difficulty == "Hard":
        for i in range(0, len(results), 4):
            isCorrect = True
            for j in range(i, i + 4):
                # Support both model_answer and persona_answer for backward compatibility
                answer = results[j].get("model_answer", results[j].get("persona_answer"))
                # Normalize correct_answer (1/0) to true/false for comparison
                expected_answer = "true" if str(results[j]["correct_answer"]) == "1" else "false"
                if str(answer).lower().strip() == expected_answer:
                    continue
                else:
                    isCorrect = False
                    break
            if isCorrect:
                correct += 1
    else:
        for result in results:
            # Support both model_answer and persona_answer for backward compatibility
            answer = result.get("model_answer", result.get("persona_answer"))
            if answer.upper().strip() == result["correct_answer"].upper().strip():
                correct += 1
    
    return correct / total if total > 0 else 0

async def main():
    """Main async function to run evaluation and iterations."""
    parser = argparse.ArgumentParser(description="Run initial evaluation and iterations")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run: eng, ling, l2e, or e2l")
    parser.add_argument("--num_iterations", type=int, required=True, help="Total number of iterations including initial evaluation")
    parser.add_argument("--difficulty", type=str, required=True, choices=["easy", "hard", "Easy", "Hard"], help="Difficulty level")
    parser.add_argument("--resume", action="store_true", required=False, default=False, help="Resume from last iteration")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=tools.llm_utils.MISTRAL_SGLANG_MODEL_ID,
        help="Model to use (default: Mistral on SGLang port 30002)",
    )
    parser.add_argument("--temperature", type=float, required=False, default=0.6, help="Temperature to use")
    parser.add_argument("--custom", type=str, required=False, default=None, help="Custom suffix to append to database path")
    parser.add_argument("--external", action="store_true", required=False, default=False, help="Use external model for feedback")
    parser.add_argument(
        "--steering_coefficient",
        type=float,
        default=None,
        help="Enable Assistant Axis steering. Positive=more assistant-like, negative=more persona-like.",
    )
    parser.add_argument(
        "--steering_model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model to load for steering (must have pre-computed axis). Default: Qwen/Qwen3-32B.",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=1,
        help="Max concurrent questions to process in parallel (1=serial, >1 for API/SGLang models)",
    )
    args = parser.parse_args()

    # Set concurrency (auto-downgrade for local GPU models)
    if args.max_concurrent > 1 and args.model in tools.llm_utils.LOCAL_MODELS:
        print(f"WARNING: {args.model} is a local GPU model. Forcing max_concurrent=1")
        tools.llm_utils.MAX_CONCURRENT = 1
    else:
        tools.llm_utils.MAX_CONCURRENT = args.max_concurrent

    # Assistant-axis steering: use steering model and set coefficient (positive=assistant, negative=persona)
    if args.steering_coefficient is not None:
        tools.llm_utils.MODEL_NAME = args.steering_model
        tools.llm_utils.STEERING_MODEL = args.steering_model
        tools.llm_utils.STEERING_COEFFICIENT = args.steering_coefficient
    elif args.model:
        tools.llm_utils.MODEL_NAME = args.model
    # use specified temperature (default is 0.0)
    if args.temperature:
        tools.llm_utils.TEMPERATURE = args.temperature
        
    difficulty = args.difficulty.capitalize()
    reset()

    # Include steering coefficient in DB path so different coefficients don't overwrite
    effective_custom = args.custom
    if args.steering_coefficient is not None:
        sc_str = f"sc{args.steering_coefficient}".replace(".", "p").replace("-", "m")
        effective_custom = f"{args.custom}_{sc_str}" if args.custom else sc_str

    effective_model = tools.llm_utils.MODEL_NAME
    print(f"Config: mode={args.mode} difficulty={difficulty} model={effective_model} temperature={args.temperature} num_iterations={args.num_iterations} external={args.external} steering_coefficient={args.steering_coefficient} max_concurrent={tools.llm_utils.MAX_CONCURRENT}")
    print(f"Resume: {args.resume}")

    # track all accuracies
    all_accuracies = []

    # run initial evaluation (if not resuming)
    if not args.resume:
        print("Running initial evaluation (iteration 1)...")
        initial_accuracy, db_path = await run_initial_eval(difficulty, args.mode, effective_custom)
        all_accuracies.append(initial_accuracy)
    # calculate initial accuracy from database (if resuming)
    else:
        print(f"Resume: calculating initial accuracy from database")
        model_to_save = {
            "Qwen/Qwen3-4B": "qwen3_4b",
            "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b",
            "Qwen/Qwen3-14B": "qwen3_14b",
            tools.llm_utils.MISTRAL_SGLANG_MODEL_ID: "mistral3_14b",
            "Qwen/Qwen3.5-35B-A3B": "qwen3.5_35b",
            "Qwen/Qwen3-32B": "qwen3_32b",
            "google/gemma-2-27b-it": "gemma2_27b",
            "meta-llama/Llama-3.3-70B-Instruct": "llama33_70b",
            "zai-org/GLM-4-9B-0414": "glm4_9b",
        }
        from token_counter import get_model_folder
        model_folder = get_model_folder(llm_utils.MODEL_NAME)
        db_path = f"../results/{args.mode}/{model_folder}/{difficulty.lower()}_t{args.temperature}_{model_to_save[llm_utils.MODEL_NAME]}"
        if effective_custom:
            db_path += f"_{effective_custom}"
        db_path += ".db"
        all_accuracies.append(calculate_accuracy_from_db(db_path, 1, difficulty))
    
    if args.resume:
        # read last iteration from database
        print(f"Resume: reading last iteration from database")
        iterations = get_all_iterations(db_path)
        last_iteration = max(iterations) if iterations else 1
        start_iteration = last_iteration + 1
        
        for i in range(2, start_iteration):
            all_accuracies.append(calculate_accuracy_from_db(db_path, i, difficulty))
        print(f"Calculated accuracies up to iteration {last_iteration}")
        print("Accuracies: " + str(all_accuracies))
    else:
        start_iteration = 2

    # run additional iterations
    if args.num_iterations > 1:
        iteration_accuracies = await run_iterations(args.mode, args.num_iterations, difficulty, db_path, start_iteration, args.external)
        all_accuracies.extend(iteration_accuracies)
    else:
        print("\nNo additional iterations to run (num_iterations = 1)")
    
    print(f"\n=== Accuracy Summary for {difficulty} and {args.mode}===")
    for i, accuracy in enumerate(all_accuracies, start=1):
        summary_line = f"Persona Accuracy for {difficulty} - Iteration {i}: {accuracy:.4f}"
        print(summary_line)
    totals = get_totals()
    if totals:
        iter1_results = load_results(db_path, iteration=1)
        num_questions = len(iter1_results) if difficulty == "Easy" else (len(iter1_results) // 4)
        to_write = totals
        if num_questions > 0:
            averaged = {}
            for k, v in totals.items():
                averaged[k] = {
                    "input_tokens": round(v["input_tokens"] / num_questions, 2),
                    "output_tokens": round(v["output_tokens"] / num_questions, 2),
                    "num_questions": num_questions,
                }
            to_write = averaged
        saved = write_to_json(totals_dict=to_write)
        if saved:
            paths = saved if isinstance(saved, list) else [saved]
            print(f"\n=== Token counts saved to {os.path.dirname(paths[0])} ===")
            for p in paths:
                print(f"  {os.path.basename(p)}")
        for k, v in to_write.items():
            if "num_questions" in v:
                print(f"  {k}: avg input_tokens={v['input_tokens']}, avg output_tokens={v['output_tokens']} (per question, n={v['num_questions']})")
            else:
                print(f"  {k}: input_tokens={v['input_tokens']}, output_tokens={v['output_tokens']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()