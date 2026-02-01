"""Main script for running iterative persona refinement on CulturalBench."""

import argparse
import asyncio
from evaluators import run_initial_eval
from iteration_runner import run_iterations
from tools.llm_utils import cleanup
from tools.db.db_utils import load_results, get_all_iterations
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
    parser.add_argument("--mode", type=str, required=True, help="Mode to run (e.g., ling_p2, eng_p1)")
    parser.add_argument("--num_iterations", type=int, required=True, help="Total number of iterations including initial evaluation")
    parser.add_argument("--difficulty", type=str, required=True, choices=["easy", "hard", "Easy", "Hard"], help="Difficulty level")
    parser.add_argument("--resume", action="store_true", required=False, default=False, help="Resume from last iteration")
    parser.add_argument("--model", type=str, required=False, default="Qwen/Qwen3-14B", help="Model to use")
    parser.add_argument("--temperature", type=float, required=False, default=0.6, help="Temperature to use")
    parser.add_argument("--custom", type=str, required=False, default=None, help="Custom suffix to append to database path")
    parser.add_argument("--use_all_previous", action="store_true", required=False, default=False, help="Use all previous personas instead of just the previous one")
    args = parser.parse_args()

    # switch to specificed model (default is Llama-3-8B-Instruct)
    if args.model:
        tools.llm_utils.MODEL_NAME = args.model
    # use specified temperature (default is 0.0)
    if args.temperature:
        tools.llm_utils.TEMPERATURE = args.temperature
        
    difficulty = args.difficulty.capitalize()

    print(f"Config: mode={args.mode} difficulty={difficulty} model={args.model} temperature={args.temperature} num_iterations={args.num_iterations}")
    print(f"Resume: {args.resume}")

    # track all accuracies
    all_accuracies = []

    # run initial evaluation (if not resuming)
    if not args.resume:
        print("Running initial evaluation (iteration 1)...")
        initial_accuracy, db_path = await run_initial_eval(difficulty, args.mode, args.custom)
        all_accuracies.append(initial_accuracy)
    # calculate initial accuracy from database (if resuming)
    else:
        print(f"Resume: calculating initial accuracy from database")
        model_to_save = {
            "Qwen/Qwen3-4B": "qwen3_4b",
            "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b",
            "Qwen/Qwen3-14B": "qwen3_14b",
        }
        db_path = f"../results/{args.mode[-2:]}/{args.mode[:-3]}/{difficulty.lower()}_t{args.temperature}_{model_to_save[llm_utils.MODEL_NAME]}"
        if args.custom:
            db_path += f"_{args.custom}"
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
        iteration_accuracies = await run_iterations(args.mode, args.num_iterations, difficulty, db_path, start_iteration, args.use_all_previous)
        all_accuracies.extend(iteration_accuracies)
    else:
        print("\nNo additional iterations to run (num_iterations = 1)")
    
    # print accuracy summary
    print(f"\n=== Accuracy Summary for {difficulty} and {args.mode}===")
    for i, accuracy in enumerate(all_accuracies, start=1):
        summary_line = f"Persona Accuracy for {difficulty} - Iteration {i}: {accuracy:.4f}"
        print(summary_line)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()