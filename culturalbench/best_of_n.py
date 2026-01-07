from collections import defaultdict

import json_repair

from tools.llm_utils import get_llm, generate_text_funcs
from tools import llm_utils
from tools.utils import country_to_language
from tools.db.db_utils import load_results, save_accuracy
from persona_generator import cap


TRAJECTORY_SELECTION_EASY_PROMPT = """You are an expert at evaluating complete reasoning trajectories for answering cultural questions.

Given a question and multiple complete trajectories (each with a persona, answer, and reasoning), select the BEST trajectory that demonstrates the most accurate and well-reasoned approach to answering this cultural question.

Consider:
1) Quality and coherence of the reasoning
2) Appropriateness of the answer choice given the reasoning
3) Cultural relevance and specificity of the persona's perspective
4) Depth of cultural understanding demonstrated in the reasoning

Question: {question}
Country: {country}

{options_text}

Here are the trajectories:
{trajectories_list}

Respond with ONLY a JSON object:
{{"selected_trajectory_index": <trajectory_number>, "reasoning": "<brief explanation>"}}
Note: trajectory_number should be 1-indexed (1 for first trajectory, 2 for second, etc.)
"""


TRAJECTORY_SELECTION_HARD_PROMPT = """You are an expert at evaluating complete reasoning trajectories for answering cultural questions.

Given a question, a specific answer option, and multiple complete trajectories (each with a persona, true/false judgment, and reasoning), select the BEST trajectory that demonstrates the most accurate and well-reasoned approach to evaluating whether this specific answer option is true or false for the question.

Consider:
1) Quality and coherence of the reasoning for this specific option
2) Appropriateness of the true/false judgment given the reasoning
3) Cultural relevance and specificity of the persona's perspective
4) Depth of cultural understanding demonstrated in the reasoning

Question: {question}
Country: {country}
Answer Option to Evaluate: {prompt_option}

Here are the trajectories:
{trajectories_list}

Respond with ONLY a JSON object:
{{"selected_trajectory_index": <trajectory_number>, "reasoning": "<brief explanation>"}}
Note: trajectory_number should be 1-indexed (1 for first trajectory, 2 for second, etc.)
"""


async def select_best_trajectory(difficulty: str, question: str, country: str, prompt_option: str, trajectories: list, options: dict = None) -> str:
    """Select the best trajectory for a given question, country, and trajectories.
    
    Args:
        difficulty: "Easy" or "Hard"
        question: The question text
        country: The country name
        prompt_option: For hard mode, the specific prompt_option being evaluated. For easy mode, can be empty string.
        trajectories: List of trajectory dictionaries, each with persona_description, model_answer, reasoning
        options: For easy mode, dictionary of answer options (A, B, C, D). For hard mode, can be None.
    
    Returns:
        JSON string response from LLM
    """
    MAX_PROMPT_LENGTH = 8192
    # Reserve space for prompt template and question/country/options (estimate ~1000 chars)
    RESERVED_SPACE = 1000
    MAX_TRAJECTORIES_CONTENT = MAX_PROMPT_LENGTH - RESERVED_SPACE
    
    def truncate_text(text: str, max_length: int) -> str:
        """Truncate text to max_length, adding ellipsis if truncated."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    # Calculate per-trajectory max length (distribute available space across trajectories)
    num_trajectories = len(trajectories)
    if num_trajectories == 0:
        raise ValueError("No trajectories provided")
    
    # Estimate overhead per trajectory (headers, labels, etc. ~100 chars per trajectory)
    overhead_per_traj = 100
    available_per_traj = (MAX_TRAJECTORIES_CONTENT - (num_trajectories * overhead_per_traj)) // num_trajectories
    max_persona_len = available_per_traj // 3
    max_reasoning_len = available_per_traj // 2
    max_answer_len = 200  # Answers are typically short
    
    # Build trajectories list text with truncation
    trajectories_list = ""
    for idx, traj in enumerate(trajectories, start=1):
        persona = truncate_text(traj.get("persona_description", ""), max_persona_len)
        answer = truncate_text(traj.get("model_answer", ""), max_answer_len)
        reasoning = truncate_text(traj.get("reasoning", ""), max_reasoning_len)
        trajectories_list += f"\n--- Trajectory {idx} ---\n"
        trajectories_list += f"Persona: {persona}\n"
        trajectories_list += f"Answer: {answer}\n"
        trajectories_list += f"Reasoning: {reasoning}\n"

    if difficulty == "Easy":
        options_text = ""
        if options:
            options_text = "Options:\n"
            for opt in ["A", "B", "C", "D"]:
                options_text += f"{opt}. {options.get(opt, '')}\n"
        
        prompt = TRAJECTORY_SELECTION_EASY_PROMPT.format(
            question=question,
            country=country,
            options_text=options_text,
            trajectories_list=trajectories_list,
        )
    else:
        prompt = TRAJECTORY_SELECTION_HARD_PROMPT.format(
            question=question,
            country=country,
            prompt_option=prompt_option,
            trajectories_list=trajectories_list,
        )
    
    # Final check - if still too long, truncate more aggressively
    if len(prompt) > MAX_PROMPT_LENGTH:
        # Calculate how much we need to reduce
        excess = len(prompt) - MAX_PROMPT_LENGTH
        # Reduce reasoning length (usually the longest component)
        reduction_per_traj = excess // num_trajectories
        trajectories_list = ""
        for idx, traj in enumerate(trajectories, start=1):
            persona = truncate_text(traj.get("persona_description", ""), max_persona_len)
            answer = truncate_text(traj.get("model_answer", ""), max_answer_len)
            # Reduce reasoning more aggressively
            new_reasoning_len = max(100, max_reasoning_len - reduction_per_traj)
            reasoning = truncate_text(traj.get("reasoning", ""), new_reasoning_len)
            trajectories_list += f"\n--- Trajectory {idx} ---\n"
            trajectories_list += f"Persona: {persona}\n"
            trajectories_list += f"Answer: {answer}\n"
            trajectories_list += f"Reasoning: {reasoning}\n"
        
        # Rebuild prompt with truncated trajectories
        if difficulty == "Easy":
            prompt = TRAJECTORY_SELECTION_EASY_PROMPT.format(
                question=question,
                country=country,
                options_text=options_text,
                trajectories_list=trajectories_list,
            )
        else:
            prompt = TRAJECTORY_SELECTION_HARD_PROMPT.format(
                question=question,
                country=country,
                prompt_option=prompt_option,
                trajectories_list=trajectories_list,
            )
        
        # If still too long after aggressive truncation, warn but proceed
        if len(prompt) > MAX_PROMPT_LENGTH:
            print(f"Warning: Prompt still exceeds limit after truncation: {len(prompt)} chars (limit: {MAX_PROMPT_LENGTH})")
    
    llm_instance = get_llm()
    _, response = generate_text_funcs[llm_utils.MODEL_NAME](llm_instance, [{"role": "user", "content": prompt}], enable_thinking_bool=False)
    
    return response
        
async def run_best_of_n_easy(db_path: str, mode: str) -> tuple[float, int, int]:
    """Best-of-n for Easy: choose best trajectory per (question, country) and use this trajectory's performance for evaluation."""
    all_results = load_results(db_path)
    if not all_results:
        return 0.0, 0, 0

    question_groups = defaultdict(list)
    for item in all_results:
        if item.get("difficulty") not in [None, "", "Easy"]:
            continue
        key = (item.get("question", ""), item.get("country", ""))
        question_groups[key].append(item)

    correct = 0
    total = 0

    for (question, country), items in question_groups.items():
        # personas across iterations for this question
        personas = [it.get("persona_description") for it in items if it.get("persona_description")]
        if not personas:
            continue
        
        # build trajectories list - one per iteration
        trajectories = []
        for item in items:
            if not item.get("persona_description") or not item.get("model_answer") or not item.get("reasoning"):
                continue
            trajectories.append({
                "persona_description": item.get("persona_description"),
                "model_answer": item.get("model_answer"),
                "reasoning": item.get("reasoning"),
            })
        
        # get all 4 options from first item (should be same for all iterations)
        first_item = items[0]
        options = first_item.get("options", {}) or {}
        correct_answer = (first_item.get("correct_answer") or "").upper().strip()

        # Select best trajectory
        best_trajectory_response = await select_best_trajectory("Easy", question, country, "", trajectories, options)

        try:
            best_trajectory_json = json_repair.loads(best_trajectory_response)
            selected_idx = best_trajectory_json.get("selected_trajectory_index", 1) - 1  # Convert from 1-indexed to 0-indexed
            
            if selected_idx < 0 or selected_idx >= len(trajectories):
                continue
            
            selected_answer = trajectories[selected_idx].get("model_answer", "").upper().strip()
        except Exception as e:
            print(f"Error parsing best trajectory: {best_trajectory_response}, error: {e}")
            continue
        
        if selected_answer == correct_answer:
            correct += 1
        total += 1
    
    acc = correct / total if total > 0 else 0.0
    return acc, correct, total

async def run_best_of_n_hard(db_path: str, mode: str) -> tuple[float, int, int]:
    """Best-of-n for Hard: choose best trajectory independently for each of 4 prompt_options, then check if all 4 are correct."""
    all_results = load_results(db_path)
    if not all_results:
        return 0.0, 0, 0

    # group all rows by (question, country)
    groups = defaultdict(list)
    for item in all_results:
        if item.get("difficulty") not in [None, "", "Hard"]:
            continue
        key = (item.get("question", ""), item.get("country", ""))
        groups[key].append(item)

    correct_sets = 0
    total_sets = 0
    for (question, country), items in groups.items():
        # Group items by prompt_option - each prompt_option should have items across iterations
        option_groups = defaultdict(list)
        for item in items:
            prompt_option = item.get("prompt_option", "")
            if prompt_option:
                option_groups[prompt_option].append(item)

        if len(option_groups) != 4:
            continue

        selected_trajectories = {} 
        correct_answers = {} 
        all_valid = True

        for prompt_option, option_items in option_groups.items():
            # Store correct_answer (same for all items with same prompt_option)
            if not option_items:
                all_valid = False
                break
            correct_answers[prompt_option] = option_items[0].get("correct_answer")

            # Build trajectories list - one per iteration for this prompt_option
            trajectories = []
            for item in option_items:
                if not item.get("persona_description") or not item.get("model_answer") or not item.get("reasoning"):
                    continue
                trajectories.append({
                    "persona_description": item.get("persona_description"),
                    "model_answer": item.get("model_answer"),
                    "reasoning": item.get("reasoning"),
                })

            if not trajectories:
                all_valid = False
                break

            # Select best trajectory for this prompt_option
            best_trajectory_response = await select_best_trajectory("Hard", question, country, prompt_option, trajectories)
            
            try:
                best_trajectory_json = json_repair.loads(best_trajectory_response)
                selected_idx = best_trajectory_json.get("selected_trajectory_index", 1) - 1  # Convert from 1-indexed to 0-indexed
                
                if selected_idx < 0 or selected_idx >= len(trajectories):
                    all_valid = False
                    break
                
                selected_trajectories[prompt_option] = trajectories[selected_idx]
            except Exception as e:
                print(f"Error parsing best trajectory for prompt_option {prompt_option}: {best_trajectory_response}, error: {e}")
                all_valid = False
                break

        if not all_valid or len(selected_trajectories) != 4:
            continue

        # Check if all 4 selected trajectories have correct answers
        all_correct = True
        for prompt_option, trajectory in selected_trajectories.items():
            selected_answer = str(trajectory.get("model_answer", "")).lower().strip()
            correct_answer = correct_answers[prompt_option]
            
            expected = "true" if str(correct_answer).lower().strip() in ["1", "true"] else "false"
            if selected_answer != expected:
                all_correct = False
                break

        if all_correct:
            correct_sets += 1
        total_sets += 1

    acc = correct_sets / total_sets if total_sets > 0 else 0.0
    return acc, correct_sets, total_sets


async def run_best_of_n(db_path: str, difficulty: str, mode: str) -> tuple[float, int, int]:
    if difficulty == "Easy":
        return await run_best_of_n_easy(db_path, mode)
    return await run_best_of_n_hard(db_path, mode)


def save_best_of_n_accuracy(db_path: str, difficulty: str, mode: str, accuracy: float, correct: int, total: int):
    """Persist best-of-n accuracy in metadata with iteration=-1."""
    save_accuracy(db_path, iteration=-1, difficulty=difficulty, mode=mode, accuracy=accuracy, correct=correct, total=total)


