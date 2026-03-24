"""
LLM-as-a-judge for SAQ: determine if model answer is correct given ground truth.
"""
import sys
import os
import re
import json
from tqdm import tqdm

eval_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(eval_dir)
for d in [parent_dir, eval_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)
from utils import get_model_response
from evaluation_utils import get_nested_json_str


def _extract_answer_from_response(raw_response):
    """
    Extract only the answer for the judge. For 5-iteration mode, responses are JSON
    with 'answer' and 'reasoning'; we use only the 'answer' part. For baseline,
    response is plain text (answer only).
    """
    raw = str(raw_response or "").strip()
    if not raw:
        return ""
    try:
        obj = get_nested_json_str(raw)
        if isinstance(obj, dict) and "answer" in obj:
            ans = obj["answer"]
            if ans is None:
                return ""
            if isinstance(ans, list):
                return ", ".join(str(a).strip() for a in ans if a is not None).strip()
            return str(ans).strip()
    except Exception:
        pass
    try:
        m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if m:
            return m.group(1).replace("\\\"", '"').strip()
        m = re.search(r'"answer"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        if m:
            inner = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
            return ", ".join(s.replace("\\\"", '"').strip() for s in inner).strip()
    except Exception:
        pass
    return raw


def _get_ground_truth_str(annotation_entry, language="English"):
    """Format ground truth answers for the judge prompt."""
    ann_list = annotation_entry.get("annotations", [])
    if not ann_list:
        return ""
    all_answers = []
    for item in ann_list:
        if language == "English":
            all_answers.extend(item.get("en_answers", []))
        else:
            all_answers.extend(item.get("answers", []))
    return "; ".join(str(a) for a in all_answers if a)


def saq_llm_judge(
    country,
    language,
    annotation_dict,
    response_df,
    id_col,
    r_col,
    annotations_key="annotations",
    judge_model="qwen3-32b",
    verbose=False,
):
    """
    Use LLM-as-a-judge to score each SAQ response against ground truth.
    Returns (accuracy, num_correct, num_total, question_rows).
    question_rows: list of dicts for the big CSV.
    """
    system_prompt = (
        "You are a judge that determines if an answer is correct.\n\n"
        "You are given a question, a model's answer, and a list of ground truth answers. "
        "Return YES only if the model's answer satisfies the rules below. Otherwise return NO.\n\n"

        "Rules:\n"
        "1. If the model provides a single answer:\n"
        "   - It must match semantically with ONE of the ground truth answers.\n\n"

        "2. If the model provides multiple answers (e.g., 'a, b, or c'):\n"
        "   - Treat each listed answer as a separate candidate.\n"
        "   - Return YES if AT LEAST ONE listed answer exactly matches a ground truth answer.\n"
        "   - Return NO if NONE of the listed answers match any ground truth answer.\n\n"

        "3. For numeric answers:\n"
        "   - Values must match exactly.\n"
        "   - No rounding, approximation, ranges, or 'close enough' logic is allowed.\n"

        "Reply with only YES or NO."
    )

    num_correct = 0
    num_total = 0
    question_rows = []

    qids_in_df = set(response_df[id_col].astype(str))
    iteration = None
    if "iteration" in response_df.columns:
        it_vals = response_df["iteration"].dropna().unique()
        if len(it_vals) == 1:
            iteration = int(it_vals[0])

    for qid, data in tqdm(annotation_dict.items(), desc="SAQ LLM judge"):
        if qid not in qids_in_df:
            continue
        if data.get("idks", {}).get("no-answer", 0) + data.get("idks", {}).get("not-applicable", 0) >= 3:
            continue
        if data.get("idks", {}).get("idk", 0) >= 5:
            continue
        ann_list = data.get(annotations_key, [])
        if not ann_list:
            continue

        raw = response_df[response_df[id_col].astype(str) == str(qid)]
        if raw.empty:
            continue
        raw = raw.iloc[0]
        raw_resp = raw.get(r_col, "")
        model_answer = _extract_answer_from_response(raw_resp).lower()
        if not model_answer:
            model_answer = str(raw_resp or "").strip()[:500]

        ground_truth_str = _get_ground_truth_str(data, language)
        print("model_answer: ", model_answer)
        print("ground_truth_str: ", ground_truth_str)
        question_text = data.get("en_question", data.get("question", ""))[:200]

        user_prompt = (
            f"Question: {question_text}\n\n"
            f"Model's answer: {model_answer}\n\n"
            f"Ground truth (any match is correct): {ground_truth_str}\n\n"
        )

        try:
            out = get_model_response(
                judge_model,
                system_prompt + "\n\n" + user_prompt,
                model=None,
                tokenizer=None,
                temperature=0,
                top_p=1,
                gpt_azure=False,
                system_message=system_prompt,
                max_tokens=16,
            )
            out = (out or "").strip().upper()
            correct = 1 if "YES" in out and "NO" not in out[:4] else 0
        except Exception as e:
            if verbose:
                print(f"Judge error for {qid}: {e}")
            correct = 0
        
        print("correct: ", correct)

        num_total += 1
        if correct:
            num_correct += 1

        question_rows.append({
            "question_id": qid,
            "country": country,
            "iteration": iteration,
            "correct": correct,
            "model_response": (model_answer or "")[:500],
            "ground_truth": (ground_truth_str or "")[:500],
        })

    accuracy = (num_correct / num_total * 100) if num_total > 0 else 0.0
    return accuracy, num_correct, num_total, question_rows
