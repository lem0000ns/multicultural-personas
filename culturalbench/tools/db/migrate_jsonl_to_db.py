"""Script to migrate existing JSONL files to SQLite database format."""

import json
import os
import sys
from pathlib import Path
from db_utils import save_results, save_accuracy


def parse_difficulty_and_mode_from_path(file_path):
    """Extract difficulty and mode from file path."""
    path = Path(file_path)
    
    difficulty = "Easy" if "Easy" in path.name else "Hard"
    
    parts = path.parts
    mode_part = None
    
    for part in parts:
        if part in ['eng', 'ling', 'l2e', 'e2l']:
            mode_part = part
            break
    
    if mode_part:
        mode = f"{mode_part}_p1"
    else:
        mode = "unknown"
    
    return difficulty, mode

def migrate_vanilla_jsonl_to_db(jsonl_path, db_path=None):
    """Migrate a vanilla JSONL file to SQLite database.
    
    Args:
        jsonl_path: Path to the JSONL file
        db_path: Path to the output database file (optional, will auto-generate if not provided)
    """
    if db_path is None:
        db_path = jsonl_path.replace('.jsonl', '.db')
    print(f"Migrating {jsonl_path} to {db_path}")

    data = {}
    with open(jsonl_path, 'r') as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line or "Accuracy" in line:
                continue
            
            try:
                result = json.loads(line)
                result['iteration'] = 1
                result['model_answer'] = result['vanilla_answer']
                # Normalize boolean correct_answer to string for Hard mode
                if 'correct_answer' in result and isinstance(result['correct_answer'], bool):
                    result['correct_answer'] = str(result['correct_answer']).lower()
                # Normalize model_answer if it's boolean
                if 'model_answer' in result and isinstance(result['model_answer'], bool):
                    result['model_answer'] = str(result['model_answer']).lower()
                data[idx] = result
                idx += 1
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line: {e}")
                continue
    
    if "Hard" in jsonl_path:
        difficulty = "Hard"
    else:
        difficulty = "Easy"
    
    # data is now structured as {1: {idx: entry}}
    save_results(db_path, data, difficulty, "vanilla")
    print(f"  Saved {len(data)} entries to {db_path}")

    if difficulty == "Hard":
        correct = 0
        total = len(data) // 4  # Integer division
        for i in range(0, len(data), 4):
            is_correct = True
            for j in range(4):
                idx = i + j
                item = data[idx]
                if str(item["model_answer"]).lower().strip() != str(item["correct_answer"]).lower().strip():
                    is_correct = False
                    break
            if is_correct:
                correct += 1
        accuracy = correct / total if total > 0 else 0
        save_accuracy(db_path, 1, difficulty, "vanilla", accuracy, correct, int(total))
        print(f"  Accuracy: {accuracy:.4f}")

    else:
        correct = 0
        total = len(data)
        for item in data.values():
            if str(item["model_answer"]).lower().strip() == str(item["correct_answer"]).lower().strip():
                correct += 1
        accuracy = correct / total if total > 0 else 0
        save_accuracy(db_path, 1, difficulty, "vanilla", accuracy, correct, int(total))
        print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"✓ Vanilla migration complete! Database saved to {db_path}\n")

def migrate_jsonl_to_db(jsonl_path, db_path=None):
    """Migrate a JSONL file to SQLite database.
    
    Args:
        jsonl_path: Path to the JSONL file
        db_path: Path to the output database file (optional, will auto-generate if not provided)
    """
    if db_path is None:
        db_path = jsonl_path.replace('.jsonl', '.db')
    
    print(f"Migrating {jsonl_path} to {db_path}")
    
    # Parse difficulty and mode from path
    difficulty, mode = parse_difficulty_and_mode_from_path(jsonl_path)
    print(f"  Detected: {difficulty} mode, {mode}")
    
    # Load JSONL data
    iteration_data = {}  # Track data by iteration
    
    with open(jsonl_path, 'r') as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line or "Accuracy" in line:
                continue
            
            try:
                entry = json.loads(line)
                iteration = entry.get('iteration', 1)
                entry['model_answer'] = entry['persona_answer']
                # Ensure iteration field exists in entry
                
                # Normalize boolean correct_answer to string for Hard mode
                if 'correct_answer' in entry and isinstance(entry['correct_answer'], bool):
                    entry['correct_answer'] = str(entry['correct_answer']).lower()
                # Normalize model_answer if it's boolean
                if 'model_answer' in entry and isinstance(entry['model_answer'], bool):
                    entry['model_answer'] = str(entry['model_answer']).lower()
                
                if iteration not in iteration_data:
                    iteration_data[iteration] = {}
                
                iteration_data[iteration][idx] = entry
                idx += 1
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line: {e}")
                continue
    
    # Create database and save data by iteration
    for iteration, iter_data in sorted(iteration_data.items()):
        print(f"  Saving iteration {iteration}: {len(iter_data)} entries")
        save_results(db_path, iter_data, difficulty, mode)
        
        # Calculate and save accuracy for this iteration
        correct = 0
        total = len(iter_data)
        
        if difficulty == "Hard":
            total = len(iter_data) // 4  # Integer division
            for i in range(0, len(iter_data), 4):
                is_correct = True
                for j in range(4):
                    idx = list(iter_data.keys())[i + j]
                    item = iter_data[idx]
                    if str(item["persona_answer"]).lower().strip() != str(item["correct_answer"]).lower().strip():
                        is_correct = False
                        break
                if is_correct:
                    correct += 1
        else:
            for item in iter_data.values():
                if item["persona_answer"].upper().strip() == item["correct_answer"].upper().strip():
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        save_accuracy(db_path, iteration, difficulty, mode, accuracy, correct, int(total))
        print(f"  Iteration {iteration} accuracy: {accuracy:.4f}")
    
    print(f"✓ Migration complete! Database saved to {db_path}\n")


def migrate_directory(results_dir="../../../results"):
    """Migrate all JSONL files in the results directory.
    
    Args:
        results_dir: Path to the results directory
    """
    results_path = Path(results_dir)
    jsonl_files = list(results_path.rglob("*.jsonl"))
    db_files = list(results_path.rglob("*.db"))
    for db_file in db_files:
        try:
            os.remove(db_file)
            print(f"Deleted database file: {db_file}")
        except Exception as e:
            print(f"Failed to delete {db_file}: {e}")
    
    if not jsonl_files:
        print("No JSONL files found to migrate.")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to migrate\n")
    
    for jsonl_file in jsonl_files:
        try:
            # Check if it's a vanilla file (either in vanilla directory or has vanilla in name)
            if "vanilla" in str(jsonl_file):
                migrate_vanilla_jsonl_to_db(str(jsonl_file))
            else:
                migrate_jsonl_to_db(str(jsonl_file))
        except Exception as e:
            print(f"✗ Error migrating {jsonl_file}: {e}\n")
    
    print(f"Migration complete! Migrated {len(jsonl_files)} files.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate JSONL files to SQLite database")
    parser.add_argument("--file", type=str, help="Single JSONL file to migrate")
    parser.add_argument("--directory", type=str, default="../../../results", help="Directory containing JSONL files to migrate")
    parser.add_argument("--output", type=str, help="Output database path (only used with --file)")
    
    args = parser.parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        migrate_jsonl_to_db(args.file, args.output)
    else:
        migrate_directory(args.directory)

