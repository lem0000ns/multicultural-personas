"""Main script for running iterative persona refinement on CulturalBench."""

import json
import argparse
import asyncio
from evaluators import run_initial_eval
from iteration_runner import run_iterations
from tools.llm_utils import cleanup

def calculate_accuracy(file_name, iteration, difficulty):
    with open(file_name, "r") as f:
        lines = [json.loads(line) for line in f.readlines() if json.loads(line)["iteration"] == iteration]
    total = len(lines) if difficulty == "Easy" else len(lines) / 4
    correct = 0
    if difficulty == "Hard":
        for i in range(0, len(lines), 4):
            isCorrect = True
            for j in range(i, i + 4):
                if str(lines[j]["persona_answer"]).lower().strip() == str(lines[j]["correct_answer"]).lower().strip():
                    continue
                else:
                    isCorrect = False
                    break
            if isCorrect:
                correct += 1
    else:
        for line in lines:
            if line["persona_answer"].upper().strip() == line["correct_answer"].upper().strip():
                correct += 1
    return correct / total

async def main():
    """Main async function to run evaluation and iterations."""
    parser = argparse.ArgumentParser(description="Run initial evaluation and iterations")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run (e.g., ling_p2, eng_p1)")
    parser.add_argument("--num_iterations", type=int, required=True, help="Total number of iterations including initial evaluation")
    parser.add_argument("--difficulty", type=str, required=True, choices=["easy", "hard", "Easy", "Hard"], help="Difficulty level")
    parser.add_argument("--resume", action="store_true", required=False, default=False, help="Resume from last iteration")
    args = parser.parse_args()

    difficulty = args.difficulty.capitalize()
    
    # track all accuracies
    all_accuracies = []
    
    # run initial evaluation (if not resuming)
    if not args.resume:
        initial_accuracy, file_name = await run_initial_eval(difficulty, args.mode, args.num_iterations)
        all_accuracies.append(initial_accuracy)
    # calculate initial accuracy from file name (if resuming)
    else:
        print(f"Resume: calculating initial accuracy from file name")
        file_name = f"../results/{args.mode[-2:]}/{args.mode[:-3]}/i{args.num_iterations}/persona_{difficulty}.jsonl"
        all_accuracies.append(calculate_accuracy(file_name, 1, difficulty))
    
    if args.resume:
        # read last iteration from file
        print(f"Resume: reading last iteration from file")
        with open(file_name, "r") as f:
            lines = f.readlines()
            last_iteration = json.loads(lines[-1])["iteration"]
        start_iteration = last_iteration + 1
        for i in range(2, start_iteration):
            all_accuracies.append(calculate_accuracy(file_name, i, difficulty))
        print(f"Calculated accuracies up to iteration {last_iteration}")
        print("Accuracies: " + str(all_accuracies))
    else:
        start_iteration = 2

    # run additional iterations
    if args.num_iterations > 1:
        iteration_accuracies = await run_iterations(args.mode, args.num_iterations, difficulty, file_name, start_iteration)
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

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()