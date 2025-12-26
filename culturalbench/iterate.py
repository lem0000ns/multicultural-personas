"""Main script for running iterative persona refinement on CulturalBench."""

import argparse
import asyncio
from evaluators import run_initial_eval
from iteration_runner import run_iterations
from tools.llm_utils import cleanup
from tools.db.db_utils import calculate_majority_vote_accuracy, save_majority_vote_accuracy
import tools.llm_utils
from tools import llm_utils

async def main():
    """Main async function to run evaluation and iterations."""
    parser = argparse.ArgumentParser(description="Run initial evaluation and iterations")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run (e.g., ling, eng)")
    parser.add_argument("--num_iterations", type=int, required=True, help="Total number of iterations including initial evaluation")
    parser.add_argument("--difficulty", type=str, required=True, choices=["easy", "hard", "Easy", "Hard"], help="Difficulty level")
    parser.add_argument("--model", type=str, required=False, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use")
    parser.add_argument("--temperature", type=float, required=False, default=0.0, help="Temperature to use")
    parser.add_argument("--custom", type=str, required=False, default=None, help="Custom suffix to append to database path")
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