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
        mode: Mode string (e.g., "eng_p1", "ling_p1", "l2e_p1")
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
        
        cursor.execute('''
            INSERT INTO results 
            (iteration, question, persona_description, pretranslated_persona, 
             correct_answer, model_answer, reasoning, thinking_content, country, refine_reasoning, 
             options, prompt_option, difficulty, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.get('iteration'),
            entry.get('question'),
            entry.get('persona_description'),
            entry.get('pretranslated_persona'),
            entry.get('correct_answer'),
            entry.get('model_answer'),
            entry.get('reasoning'),
            entry.get('thinking_content'),
            entry.get('country'),
            entry.get('refine_reasoning'),
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

