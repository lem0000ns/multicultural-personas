"""Main script for running iterative persona refinement on CulturalBench."""

import argparse
from evaluators import run_initial_eval
from iteration_runner import run_iterations
from llm_utils import cleanup

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Run initial evaluation and iterations")
        parser.add_argument("--mode", type=str, required=True, help="Mode to run (e.g., ling_p2, eng_p1)")
        parser.add_argument("--num_iterations", type=int, required=True, help="Total number of iterations including initial evaluation")
        parser.add_argument("--difficulty", type=str, required=True, choices=["easy", "hard", "Easy", "Hard"], help="Difficulty level")
        args = parser.parse_args()

        difficulty = args.difficulty.capitalize()
        
        # track all accuracies
        all_accuracies = []
        
        # run initial evaluation
        initial_accuracy, file_name = run_initial_eval(difficulty, args.mode, args.num_iterations)
        all_accuracies.append(initial_accuracy)
        
        # run additional iterations
        if args.num_iterations > 1:
            iteration_accuracies = run_iterations(args.mode, args.num_iterations, difficulty, file_name)
            all_accuracies.extend(iteration_accuracies)
        else:
            print("\nNo additional iterations to run (num_iterations = 1)")
        
        # write accuracy summary to end of file
        with open(file_name, "a") as f:
            f.write("\n")
            for i, accuracy in enumerate(all_accuracies, start=1):
                summary_line = f"Persona Accuracy for {difficulty} - Iteration {i}: {accuracy:.4f}"
                f.write(summary_line + "\n")
                print(summary_line)
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()