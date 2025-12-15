"""Main script for running iterative persona refinement on CulturalBench."""

import argparse
import asyncio
from evaluators import run_initial_eval
from iteration_runner import run_iterations
from tools.llm_utils import cleanup
from tools.db.db_utils import calculate_majority_vote_accuracy, save_majority_vote_accuracy
from best_of_n import run_best_of_n, save_best_of_n_accuracy
import tools.llm_utils
from tools import llm_utils

async def main():
    """Main async function to run evaluation and iterations."""
    parser = argparse.ArgumentParser(description="Run initial evaluation and iterations")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run (e.g., ling_p2, eng_p1)")
    parser.add_argument("--num_iterations", type=int, required=True, help="Total number of iterations including initial evaluation")
    parser.add_argument("--difficulty", type=str, required=True, choices=["easy", "hard", "Easy", "Hard"], help="Difficulty level")
    parser.add_argument("--model", type=str, required=False, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use")
    parser.add_argument("--temperature", type=float, required=False, default=0.0, help="Temperature to use")
    parser.add_argument("--custom", type=str, required=False, default=None, help="Custom suffix to append to database path")
    parser.add_argument("--baseline", action="store_true", required=False, default=False, help="Majority voting for final answer")
    args = parser.parse_args()

    # switch to specificed model (default is Llama-3-8B-Instruct)
    if args.model:
        tools.llm_utils.MODEL_NAME = args.model
    # use specified temperature (default is 0.0)
    if args.temperature:
        tools.llm_utils.TEMPERATURE = args.temperature
        
    difficulty = args.difficulty.capitalize()
    
    # track all accuracies
    all_accuracies = []
    
    # run initial evaluation
    initial_accuracy, db_path = await run_initial_eval(difficulty, args.mode, args.custom)
    all_accuracies.append(initial_accuracy)

    # run additional iterations
    if args.num_iterations > 1:
        iteration_accuracies = await run_iterations(args.mode, args.num_iterations, difficulty, db_path)
        all_accuracies.extend(iteration_accuracies)
    else:
        print("\nNo additional iterations to run (num_iterations = 1)")
    
    # After all iterations: run best-of-n (same LLM) or majority voting based on --baseline flag
    majority_accuracy = None
    best_of_n_accuracy = None
    
    if args.num_iterations > 1:
        if args.baseline:
            # Baseline mode: majority voting
            print(f"\n=== Calculating Majority Voting (Best-of-{args.num_iterations}) ===")
            majority_accuracy, correct, total = calculate_majority_vote_accuracy(
                db_path, difficulty, args.mode, args.num_iterations
            )
            save_majority_vote_accuracy(db_path, difficulty, args.mode, majority_accuracy, correct, total)
            print(f"Majority Voting Accuracy: {majority_accuracy:.4f} ({correct}/{total})")
        else:
            # Best-of-n mode: same/original LLM selects best persona, then answers with it
            print(f"\n=== Running Best-of-{args.num_iterations} Persona Selection (same LLM) ===")
            best_of_n_accuracy, correct, total = await run_best_of_n(db_path, difficulty, args.mode)
            save_best_of_n_accuracy(db_path, difficulty, args.mode, best_of_n_accuracy, correct, total)
            print(f"Best-of-{args.num_iterations} Accuracy: {best_of_n_accuracy:.4f} ({correct}/{total})")
    
    # print accuracy summary
    print(f"\n=== Accuracy Summary for {difficulty} and {args.mode} and temperature {tools.llm_utils.TEMPERATURE}===")
    for i, accuracy in enumerate(all_accuracies, start=1):
        summary_line = f"Persona Accuracy for {difficulty} - Iteration {i}: {accuracy:.4f}"
        print(summary_line)
    
    if majority_accuracy is not None:
        print(f"Majority Voting (Best-of-{args.num_iterations}): {majority_accuracy:.4f}")

    if best_of_n_accuracy is not None:
        print(f"Best-of-{args.num_iterations} (same LLM): {best_of_n_accuracy:.4f}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()