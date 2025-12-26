"""Database utilities for storing evaluation results."""

import sqlite3
import json
import os
from typing import Dict, Optional


def init_db(db_path: str):
    """Initialize the database with required tables and indexes.
    
    Args:
        db_path: Path to the SQLite database file
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            question TEXT NOT NULL,
            persona_description TEXT,
            pretranslated_persona TEXT,
            correct_answer TEXT NOT NULL,
            model_answer TEXT NOT NULL,
            reasoning TEXT,
            thinking_content TEXT,
            country TEXT NOT NULL,
            refine_reasoning TEXT,
            options TEXT,
            prompt_option TEXT,
            difficulty TEXT,
            mode TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Migrate existing table to add missing columns if needed
    cursor.execute("PRAGMA table_info(results)")
    columns = {col[1] for col in cursor.fetchall()}
    
    if 'thinking_content' not in columns:
        cursor.execute('ALTER TABLE results ADD COLUMN thinking_content TEXT')
        print("Added missing 'thinking_content' column to results table")
    
    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_iteration ON results(iteration)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_country ON results(country)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_difficulty ON results(difficulty)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mode ON results(mode)')
    
    # Create metadata table for tracking accuracies
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            difficulty TEXT NOT NULL,
            mode TEXT NOT NULL,
            accuracy REAL NOT NULL,
            correct_count INTEGER NOT NULL,
            total_count INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def save_results(db_path: str, data: Dict, difficulty: str, mode: str):
    """Save evaluation results to database.
    
    Args:
        db_path: Path to the SQLite database file
        data: Dictionary of results to save
        difficulty: "Easy" or "Hard"
        mode: Mode string (e.g., "eng_p1", "ling_p1", "l2e_p1") - stored with _p1 suffix for backwards compatibility
    """
    init_db(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the iteration number from the first entry to delete old data for this iteration
    if data:
        first_entry = next(iter(data.values()))
        iteration = first_entry.get('iteration')
        
        # Delete any existing data for this iteration to prevent duplicates
        if iteration is not None:
            cursor.execute(
                'DELETE FROM results WHERE iteration = ? AND difficulty = ? AND mode = ?',
                (iteration, difficulty, mode)
            )
            print(f"Cleared existing data for iteration {iteration}")
    
    for entry in data.values():
        # Convert options dict to JSON string if present
        options_str = json.dumps(entry.get('options', {})) if 'options' in entry else None
        
        # Helper function to convert dict/list values to JSON strings
        # This handles cases where LLM outputs might be structured data instead of strings
        def convert_value(value):
            if not isinstance(value, str):
                return json.dumps(value)
            return value
        
        cursor.execute('''
            INSERT INTO results 
            (iteration, question, persona_description, pretranslated_persona, 
             correct_answer, model_answer, reasoning, thinking_content, country, refine_reasoning, 
             options, prompt_option, difficulty, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.get('iteration'),
            entry.get('question'),
            convert_value(entry.get('persona_description')),
            convert_value(entry.get('pretranslated_persona')),
            entry.get('correct_answer'),
            entry.get('model_answer'),
            convert_value(entry.get('reasoning')),
            convert_value(entry.get('thinking_content')),
            entry.get('country'),
            convert_value(entry.get('refine_reasoning')),
            options_str,
            entry.get('prompt_option'),
            difficulty,
            mode
        ))
    
    conn.commit()
    conn.close()


def save_accuracy(db_path: str, iteration: int, difficulty: str, mode: str, 
                  accuracy: float, correct: int, total: int):
    """Save accuracy metrics to metadata table.
    
    Args:
        db_path: Path to the SQLite database file
        iteration: Iteration number
        difficulty: "Easy" or "Hard"
        mode: Mode string
        accuracy: Accuracy value
        correct: Number of correct answers
        total: Total number of questions
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Delete any existing metadata for this iteration to prevent duplicates
    cursor.execute(
        'DELETE FROM metadata WHERE iteration = ? AND difficulty = ? AND mode = ?',
        (iteration, difficulty, mode)
    )
    
    cursor.execute('''
        INSERT INTO metadata 
        (iteration, difficulty, mode, accuracy, correct_count, total_count)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (iteration, difficulty, mode, accuracy, correct, total))
    
    conn.commit()
    conn.close()




def load_results(db_path: str, iteration: Optional[int] = None, 
                 country: Optional[str] = None) -> list:
    """Load results from database with optional filters.
    
    Args:
        db_path: Path to the SQLite database file
        iteration: Optional iteration number to filter by
        country: Optional country to filter by
    
    Returns:
        List of dictionaries containing results
    """
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    query = "SELECT * FROM results WHERE 1=1"
    params = []
    
    if iteration is not None:
        query += " AND iteration = ?"
        params.append(iteration)
    
    if country is not None:
        query += " AND country = ?"
        params.append(country)
    
    # Order by id to maintain insertion order (critical for Hard mode grouping)
    query += " ORDER BY id"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Convert to list of dicts and parse JSON fields
    results = []
    for row in rows:
        result = dict(row)
        if result.get('options'):
            result['options'] = json.loads(result['options'])
        results.append(result)
    
    conn.close()
    return results


def load_previous_iteration(db_path: str, iteration: int) -> list:
    """Load results from the previous iteration.
    
    Args:
        db_path: Path to the SQLite database file
        iteration: Current iteration (will load iteration-1)
    
    Returns:
        List of dictionaries containing results from previous iteration
    """
    return load_results(db_path, iteration=iteration - 1)


def load_all_iterations_for_question(db_path: str, question: str, country: str, 
                                     difficulty: str, mode: str, max_iteration: int) -> list:
    """Load all previous iterations for a specific question.
    
    Args:
        db_path: Path to the SQLite database file
        question: The question text
        country: The country name
        difficulty: "Easy" or "Hard"
        mode: Mode string
        max_iteration: Maximum iteration to load (exclusive, loads iterations < max_iteration)
    
    Returns:
        List of dictionaries containing results from all previous iterations, sorted by iteration
    """
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM results 
        WHERE question = ? AND country = ? AND difficulty = ? AND mode = ? AND iteration < ?
        ORDER BY iteration, id
    ''', (question, country, difficulty, mode, max_iteration))
    
    rows = cursor.fetchall()
    
    # Convert to list of dicts and parse JSON fields
    results = []
    for row in rows:
        result = dict(row)
        if result.get('options'):
            result['options'] = json.loads(result['options'])
        results.append(result)
    
    conn.close()
    return results


def get_all_iterations(db_path: str) -> list:
    """Get list of all iteration numbers in the database.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        Sorted list of iteration numbers
    """
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT iteration FROM results ORDER BY iteration")
    iterations = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return iterations


def get_accuracies(db_path: str) -> list:
    """Get accuracy history from metadata table.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        List of dictionaries containing accuracy data
    """
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT iteration, difficulty, mode, accuracy, correct_count, total_count, created_at
        FROM metadata 
        ORDER BY iteration
    """)
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results


def calculate_majority_vote_accuracy(db_path: str, difficulty: str, mode: str, num_iterations: int) -> tuple:
    """Calculate accuracy using majority voting across all iterations.
    
    For each question, the final answer is determined by the most frequently
    chosen answer across all iterations. Ties are broken by selecting the
    answer from the latest iteration.
    
    Args:
        db_path: Path to the SQLite database file
        difficulty: "Easy" or "Hard"
        mode: Mode string
        num_iterations: Total number of iterations
    
    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    from collections import Counter, defaultdict
    
    if not os.path.exists(db_path):
        return 0.0, 0, 0
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Load all results
    cursor.execute("""
        SELECT * FROM results 
        WHERE difficulty = ? AND mode = ?
        ORDER BY iteration, id
    """, (difficulty, mode))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return 0.0, 0, 0
    
    results = [dict(row) for row in rows]
    
    # Check if Hard mode (has prompt_option) or Easy mode
    is_hard_mode = bool(results[0].get("prompt_option"))
    
    if is_hard_mode:
        # Hard mode: group by (question, prompt_option), then apply majority vote per option
        # A question is correct only if ALL 4 options have correct majority votes
        
        # Group by question
        question_groups = defaultdict(lambda: defaultdict(list))
        for item in results:
            question = item.get("question", "")
            prompt_option = item.get("prompt_option", "")
            question_groups[question][prompt_option].append(item)
        
        correct_questions = 0
        total_questions = len(question_groups)
        
        for question, option_groups in question_groups.items():
            all_options_correct = True
            
            for prompt_option, items in option_groups.items():
                # Get majority vote for this option
                answers = [str(item.get("model_answer", "")).lower().strip() for item in items]
                answer_counts = Counter(answers)
                
                if not answer_counts:
                    all_options_correct = False
                    continue
                
                # Find the most common answer
                max_count = max(answer_counts.values())
                most_common = [ans for ans, count in answer_counts.items() if count == max_count]
                
                # If tie, use the latest iteration's answer
                if len(most_common) > 1:
                    latest_item = max(items, key=lambda x: x.get("iteration", 0))
                    majority_answer = str(latest_item.get("model_answer", "")).lower().strip()
                else:
                    majority_answer = most_common[0]
                
                # Check correctness
                correct_answer = items[0].get("correct_answer", "")
                expected = "true" if str(correct_answer) in ["1", "true", "True"] else "false"
                
                if majority_answer != expected:
                    all_options_correct = False
            
            if all_options_correct:
                correct_questions += 1
        
        accuracy = correct_questions / total_questions if total_questions > 0 else 0.0
        return accuracy, correct_questions, total_questions
    
    else:
        # Easy mode: group by (question, country), apply majority vote
        question_groups = defaultdict(list)
        for item in results:
            # Use question + country as key to uniquely identify a question
            key = (item.get("question", ""), item.get("country", ""))
            question_groups[key].append(item)
        
        correct_count = 0
        total_count = len(question_groups)
        
        for key, items in question_groups.items():
            # Get majority vote
            answers = [str(item.get("model_answer", "")).upper().strip() for item in items]
            answer_counts = Counter(answers)
            
            if not answer_counts:
                continue
            
            # Find the most common answer
            max_count = max(answer_counts.values())
            most_common = [ans for ans, count in answer_counts.items() if count == max_count]
            
            # If tie, use the latest iteration's answer
            if len(most_common) > 1:
                latest_item = max(items, key=lambda x: x.get("iteration", 0))
                majority_answer = str(latest_item.get("model_answer", "")).upper().strip()
            else:
                majority_answer = most_common[0]
            
            # Check correctness
            correct_answer = str(items[0].get("correct_answer", "")).upper().strip()
            
            if majority_answer == correct_answer:
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return accuracy, correct_count, total_count


def save_majority_vote_accuracy(db_path: str, difficulty: str, mode: str, 
                                 accuracy: float, correct: int, total: int):
    """Save majority voting accuracy to metadata table.
    
    Uses iteration=0 to distinguish from regular iteration results.
    
    Args:
        db_path: Path to the SQLite database file
        difficulty: "Easy" or "Hard"
        mode: Mode string
        accuracy: Accuracy value
        correct: Number of correct answers
        total: Total number of questions
    """
    # Use iteration=0 to indicate majority voting result
    save_accuracy(db_path, iteration=0, difficulty=difficulty, mode=mode,
                  accuracy=accuracy, correct=correct, total=total)

