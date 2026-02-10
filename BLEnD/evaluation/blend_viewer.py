"""Streamlit UI for visualizing BLEnD multiple choice evaluation results.
SAQ per-response scores are loaded from saq_i5/*_response_score_iterN.csv."""

import csv
import streamlit as st
import json
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import re

# Set page config
st.set_page_config(
    page_title="BLEnD Results Viewer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    /* Style metrics with black background and white text */
    [data-testid="stMetric"] {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
    [data-testid="stMetricValue"] {
        font-size: 32px;
        color: #ffffff !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        font-size: 16px;
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_csv_file(csv_path):
    """Load data from CSV file and return as DataFrame."""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

# SAQ: known countries (match filenames like *-North_Korea_* or *-China_*)
BLEnD_SAQ_COUNTRIES = [
    "Northern_Nigeria", "North_Korea", "South_Korea", "West_Java",
    "UK", "US", "Algeria", "China", "Indonesia", "Spain", "Iran", "Mexico",
    "Assam", "Greece", "Ethiopia", "Azerbaijan",
]

def _country_from_saq_filename(name):
    """Extract country from SAQ result filename, e.g. *-China_English_* or *-North_Korea_*."""
    for c in sorted(BLEnD_SAQ_COUNTRIES, key=len, reverse=True):
        if f"-{c}_" in name or f"_{c}_" in name:
            return c
    return None

def get_saq_available(baseline_dir, only_reasoning_dir, i5_dir=None):
    """Return (baseline_files_by_country, reasoning_files_by_country, i5_by_country, all_countries)."""
    baseline_by_country = {}
    reasoning_by_country = {}
    i5_by_country = {}
    for d, out in [
        (baseline_dir, baseline_by_country),
        (only_reasoning_dir, reasoning_by_country),
    ]:
        if not d or not d.exists():
            continue
        for f in d.glob("*_result.csv"):
            c = _country_from_saq_filename(f.name)
            if c:
                out[c] = f
    if i5_dir and i5_dir.exists():
        for f in i5_dir.glob("*_result.csv"):
            c = _country_from_saq_filename(f.name)
            if c:
                i5_by_country[c] = f
    all_countries = sorted(set(baseline_by_country) | set(reasoning_by_country) | set(i5_by_country))
    return baseline_by_country, reasoning_by_country, i5_by_country, all_countries

def load_saq_annotations(annotations_dir, country):
    """Load annotation dict for country: qid -> {question, en_question, annotations, idks}."""
    path = annotations_dir / f"{country}_data.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_saq_scores(results_csv_path, country, iteration=None):
    """Load SEM-B and SEM-W for a country from a results CSV. Matches evaluate.py + eval_saq: dedupe (keep='last'), filter to country and iteration (coerce iteration to int), take the single score for that (country, iteration). Returns (sem_b, sem_w) or (None, None)."""
    if not results_csv_path or not Path(results_csv_path).exists():
        return None, None
    try:
        df = pd.read_csv(results_csv_path, encoding="utf-8")
        if "country" not in df.columns or "eval_method" not in df.columns or "score" not in df.columns:
            return None, None
        key = [c for c in ["model", "country", "prompt_no", "eval_method", "iteration"] if c in df.columns]
        if key:
            df = df.drop_duplicates(subset=key, keep="last")
        sub = df[df["country"] == country]
        if iteration is not None and "iteration" in df.columns:
            sub = sub[sub["iteration"].astype(int) == int(iteration)]
        sem_b = sub[sub["eval_method"] == "SEM-B"]["score"]
        sem_w = sub[sub["eval_method"] == "SEM-W"]["score"]
        sb = float(sem_b.iloc[0]) if len(sem_b) else None
        sw = float(sem_w.iloc[0]) if len(sem_w) else None
        return sb, sw
    except Exception:
        return None, None


def load_saq_scores_all_iterations(results_csv_path, country):
    """Load SEM-B and SEM-W per iteration for a country. Iteration order matches CSV (1..5). Returns list of (iteration, sem_b, sem_w) or []."""
    if not results_csv_path or not Path(results_csv_path).exists():
        return []
    try:
        df = pd.read_csv(results_csv_path, encoding="utf-8")
        if "country" not in df.columns or "eval_method" not in df.columns or "score" not in df.columns or "iteration" not in df.columns:
            return []
        sub = df[df["country"] == country]
        if sub.empty:
            return []
        iterations_sorted = sorted(sub["iteration"].dropna().unique(), key=lambda x: int(x))
        out = []
        for it in iterations_sorted:
            sb, sw = load_saq_scores(results_csv_path, country, iteration=int(it))
            if sb is not None or sw is not None:
                out.append((int(it), sb, sw))
        return out
    except Exception:
        return []


def load_saq_i5_response_scores(result_csv_path):
    """Load per-response SEM-B (binary_score) and SEM-W (weight_score) from response_score_iterN.csv files.
    Index by question ID (first column, e.g. Al-en-01) and iteration. SEM-B/SEM-W are the last two columns.
    Uses csv.reader so multiline quoted fields (prompt, response) are parsed as one row."""
    path = Path(result_csv_path).resolve()
    if not path.exists():
        return {}
    stem = path.stem
    print("stem", stem)
    base = stem.replace("_result", "") if stem.endswith("_result") else stem
    print("base", base)

    # Only replace the fourth '-' with '_'
    def replace_nth(s, sub, repl, n):
        """Replace the n-th occurrence of sub in s with repl."""
        find = [i for i, c in enumerate(s) if c == sub]
        if len(find) < n:
            return s
        idx = find[n-1]
        return s[:idx] + repl + s[idx+1:]

    score_prefix = replace_nth(base, '-', '_', 4)
    print("score_prefix", score_prefix)
    parent = path.parent.resolve()
    print("parent", parent)
    out = {}
    print(parent.glob(f"{score_prefix}_response_score_iter*.csv"))
    for score_path in parent.glob(f"{score_prefix}_response_score_iter*.csv"):
        m = re.search(r"_iter(\d+)\.csv$", score_path.name)
        print("m", m)
        if not m:
            continue
        it = int(m.group(1))
        print("it", it)
        try:
            with open(score_path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header or len(header) < 3:
                    continue
                for row in reader:
                    print("row", row)
                    if len(row) < 3:
                        continue
                    qid = (row[0] or "").strip()
                    if not qid or qid.lower() == "nan":
                        continue
                    b_str, w_str = row[-2], row[-1]
                    try:
                        b = float(b_str) if b_str and b_str.strip() else None
                        w = float(w_str) if w_str and w_str.strip() else None
                    except (TypeError, ValueError):
                        b, w = None, None
                    out[(qid, it)] = (b, w)
        except Exception:
            continue
    print(out)
    return out

def get_available_results():
    """Scan the mc_data directory for available CSV files.
    
    Returns:
        List of dictionaries with file information
    """
    mc_data_dir = Path(__file__).parent / "mc_data"
    
    if not mc_data_dir.exists():
        return []
    
    # Find all CSV files that look like result files (containing -mc_res)
    csv_files = list(mc_data_dir.glob("*-mc_res*.csv"))
    
    files = []
    for file in csv_files:
        files.append({
            "path": str(file),
            "name": file.name,
            "relative_path": str(file.relative_to(mc_data_dir.parent))
        })
    
    return sorted(files, key=lambda x: x["name"])

def is_answer_correct(row):
    """Check if a row's answer is correct."""
    answer_idx = str(row.get('answer_idx', '')).strip()
    final_ans = str(row.get('final_ans', '')).strip()
    
    # Normalize both to uppercase for comparison
    # Also handle cases where answer might be just the letter or include extra text
    answer_idx_clean = answer_idx.upper()
    final_ans_clean = final_ans.upper()
    
    # Extract first letter if answer contains multiple characters
    if len(answer_idx_clean) > 1:
        answer_idx_clean = answer_idx_clean[0]
    if len(final_ans_clean) > 1:
        final_ans_clean = final_ans_clean[0]
    
    return answer_idx_clean == final_ans_clean

def calculate_accuracy(df):
    """Calculate accuracy from the dataframe."""
    if df is None or len(df) == 0:
        return 0.0
    
    correct = sum(1 for _, row in df.iterrows() if is_answer_correct(row))
    return (correct / len(df)) * 100

def calculate_accuracy_by_iteration(df):
    """Calculate accuracy for each iteration."""
    if df is None or 'iteration' not in df.columns:
        return {}
    
    iteration_accuracies = {}
    for iteration in sorted(df['iteration'].unique()):
        iter_df = df[df['iteration'] == iteration]
        iteration_accuracies[iteration] = calculate_accuracy(iter_df)
    
    return iteration_accuracies


def _main_saq():
    _script_dir = Path(__file__).resolve().parent
    base = _script_dir.parent
    eval_dir = _script_dir
    baseline_dir = base / "saq_baseline"
    only_reasoning_dir = base / "saq_only_reasoning"
    i5_dir = base / "saq_i5"
    annotations_dir = base / "data" / "annotations"
    baseline_results_csv = eval_dir / "saq_baseline_results.csv"
    reasoning_results_csv = eval_dir / "saq_only_reasoning_results.csv"
    i5_results_csv = eval_dir / "saq_i5_results.csv"
    baseline_by_country, reasoning_by_country, i5_by_country, all_countries = get_saq_available(
        baseline_dir, only_reasoning_dir, i5_dir
    )
    if not all_countries:
        st.error("No SAQ result files found in saq_baseline/, saq_only_reasoning/, or saq_i5/.")
        return
    st.markdown("### Short Answer (SAQ): Questions, Responses & Ground Truth")
    view_opts = ["Baseline only", "Only Reasoning only", "Iterative (i5)", "Compare all"]
    view_mode = st.sidebar.radio("View", view_opts, index=2 if i5_by_country else 0)
    selected_country = st.sidebar.selectbox("Country", all_countries, index=0)
    show_baseline = view_mode in ("Baseline only", "Compare all") and selected_country in baseline_by_country
    show_reasoning = view_mode in ("Only Reasoning only", "Compare all") and selected_country in reasoning_by_country
    show_i5 = view_mode in ("Iterative (i5)", "Compare all") and selected_country in i5_by_country
    if not show_baseline and not show_reasoning and not show_i5:
        st.warning("No result file for this country for the selected view.")
        return
    if show_baseline:
        baseline_df = pd.read_csv(baseline_by_country[selected_country], encoding="utf-8")
    else:
        baseline_df = None
    if show_reasoning:
        reasoning_df = pd.read_csv(reasoning_by_country[selected_country], encoding="utf-8")
    else:
        reasoning_df = None
    if show_i5:
        i5_df = pd.read_csv(i5_by_country[selected_country], encoding="utf-8")
    else:
        i5_df = None
    annotations = load_saq_annotations(annotations_dir, selected_country)
    sem_b_baseline, sem_w_baseline = load_saq_scores(str(baseline_results_csv), selected_country)
    sem_b_reasoning, sem_w_reasoning = load_saq_scores(str(reasoning_results_csv), selected_country)
    sem_b_i5, sem_w_i5 = load_saq_scores(str(i5_results_csv), selected_country, iteration=5)
    if view_mode == "Compare all":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Baseline · SEM-B", "%.2f" % sem_b_baseline if sem_b_baseline is not None else "—")
            st.metric("Baseline · SEM-W", "%.2f" % sem_w_baseline if sem_w_baseline is not None else "—")
        with c2:
            st.metric("Only Reasoning · SEM-B", "%.2f" % sem_b_reasoning if sem_b_reasoning is not None else "—")
            st.metric("Only Reasoning · SEM-W", "%.2f" % sem_w_reasoning if sem_w_reasoning is not None else "—")
        with c3:
            st.metric("Iterative (i5) · SEM-B", "%.2f" % sem_b_i5 if sem_b_i5 is not None else "—")
            st.metric("Iterative (i5) · SEM-W", "%.2f" % sem_w_i5 if sem_w_i5 is not None else "—")
    elif view_mode == "Baseline only":
        c1, c2 = st.columns(2)
        with c1:
            st.metric("SEM-B", "%.2f" % sem_b_baseline if sem_b_baseline is not None else "—")
        with c2:
            st.metric("SEM-W", "%.2f" % sem_w_baseline if sem_w_baseline is not None else "—")
    elif view_mode == "Only Reasoning only":
        c1, c2 = st.columns(2)
        with c1:
            st.metric("SEM-B", "%.2f" % sem_b_reasoning if sem_b_reasoning is not None else "—")
        with c2:
            st.metric("SEM-W", "%.2f" % sem_w_reasoning if sem_w_reasoning is not None else "—")
    elif view_mode == "Iterative (i5)":
        c1, c2 = st.columns(2)
        with c1:
            st.metric("SEM-B", "%.2f" % sem_b_i5 if sem_b_i5 is not None else "—")
        with c2:
            st.metric("SEM-W", "%.2f" % sem_w_i5 if sem_w_i5 is not None else "—")
    # Show all 5 iterations for i5 when relevant
    if (view_mode in ("Iterative (i5)", "Compare all") and i5_by_country and selected_country in i5_by_country):
        i5_iter_scores = load_saq_scores_all_iterations(str(i5_results_csv), selected_country)
        if i5_iter_scores:
            st.markdown("**Iterative (i5) performance by iteration**")
            iter_df = pd.DataFrame([{"Iteration": it, "SEM-B": sb, "SEM-W": sw} for it, sb, sw in i5_iter_scores])
            st.dataframe(iter_df.style.format({"SEM-B": "{:.2f}", "SEM-W": "{:.2f}"}), hide_index=True)
    st.markdown("---")
    ids_baseline = set(baseline_df["ID"].astype(str)) if baseline_df is not None and "ID" in baseline_df.columns else set()
    ids_reasoning = set(reasoning_df["ID"].astype(str)) if reasoning_df is not None and "ID" in reasoning_df.columns else set()
    ids_i5 = set(i5_df["ID"].astype(str)) if i5_df is not None and "ID" in i5_df.columns else set()
    ids_ann = set(annotations.keys()) if annotations else set()
    i5_response_scores = {}
    ids_with_i5_scores = set()
    if show_i5 and selected_country in i5_by_country:
        result_path = Path(i5_by_country[selected_country])
        if not result_path.is_absolute():
            result_path = (base / result_path).resolve()
        else:
            result_path = result_path.resolve()
        i5_response_scores = load_saq_i5_response_scores(result_path)
        ids_with_i5_scores = set(qid for (qid, _) in i5_response_scores.keys())
        n_total = len(i5_response_scores)
        n_with_values = sum(1 for (b, w) in i5_response_scores.values() if b is not None or w is not None)
        if n_total > 0:
            st.caption("Per-response scores: **%d** entries loaded (%d with SEM-B/SEM-W values)." % (n_total, n_with_values))
        if not i5_response_scores or n_with_values == 0:
            st.caption("Per-response SEM-B/SEM-W show \"—\" when no *_response_score_iterN.csv files are found in **saq_i5/** or when score columns are empty. evaluate.sh writes aggregates to evaluation/saq_i5_results.csv; per-question scores are written to **saq_i5/** by evaluate.py.")
    ids_with_response = ids_baseline | ids_reasoning | ids_i5
    all_question_ids = sorted(ids_baseline | ids_reasoning | ids_i5 | ids_ann)
    show_only_with_response = st.sidebar.checkbox(
        "Show only questions with model responses",
        value=True,
        help="Annotations have ~500 questions per country; you sampled 100 per country. Uncheck to see all annotation IDs (many will show 'no response')."
    )
    question_ids = sorted(ids_with_response) if show_only_with_response else all_question_ids
    if show_i5 and ids_with_i5_scores:
        question_ids = [q for q in question_ids if q in ids_with_i5_scores]
    n_with_response = len(ids_with_response)
    n_total_ann = len(ids_ann)
    if not question_ids:
        st.info("No questions to show.")
        return
    if show_only_with_response:
        st.caption("Showing %d questions that have model responses (this country has %d responses; annotations have %d questions)." % (len(question_ids), n_with_response, n_total_ann))
    else:
        st.caption("Showing all %d question IDs (annotations + responses). Questions without a response will show '(no response for this question)'." % len(question_ids))
    if show_i5 and ids_with_i5_scores:
        st.caption("Only questions with SEM-B/SEM-W scores (from response_score CSVs) are shown.")

    def _get_i5_scores(scores_dict, qid, it_num):
        """Look up (binary_score, weight_score) for (qid, it_num); try str(qid), qid, and normalized numeric."""
        q = str(qid).strip()
        keys_to_try = [(q, it_num), (qid, it_num)]
        try:
            keys_to_try.append((str(int(float(q))), it_num))
        except (ValueError, TypeError):
            pass
        for key in keys_to_try:
            if key in scores_dict:
                return scores_dict[key]
        return (None, None)

    def row_by_id(df, qid):
        if df is None or "ID" not in df.columns:
            return None
        rows = df[df["ID"].astype(str) == str(qid)]
        return rows.iloc[0] if len(rows) else None

    def rows_by_id_and_iteration(df, qid):
        """Return list of (iteration_num, row) for this qid, sorted by iteration. Use when df has multiple rows per ID (e.g. i5 with iteration column)."""
        if df is None or "ID" not in df.columns:
            return []
        rows = df[df["ID"].astype(str) == str(qid)]
        if len(rows) == 0:
            return []
        if "iteration" not in df.columns:
            return [(1, rows.iloc[0])] if len(rows) else []
        out = []
        for _, row in rows.iterrows():
            it = row.get("iteration", 1)
            try:
                it = int(it)
            except (TypeError, ValueError):
                it = 1
            out.append((it, row))
        out.sort(key=lambda x: x[0])
        return out

    per_page = st.sidebar.selectbox("Questions per page", [10, 25, 50, 100, "All"], index=1, key="saq_perpage_%s" % selected_country)
    per_page = len(question_ids) if per_page == "All" else int(per_page)
    total_pages = max(1, (len(question_ids) + per_page - 1) // per_page)
    page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, key="saq_page_%s" % selected_country)
    start = (page - 1) * per_page
    end = min(start + per_page, len(question_ids))
    page_ids = question_ids[start:end]
    st.caption("Showing %d-%d of %d · %s" % (start + 1, end, len(question_ids), selected_country))

    for qid in page_ids:
        ann = annotations.get(qid, {}) if annotations else {}
        question_text = ann.get("en_question") or ann.get("question") or ""
        baseline_row = row_by_id(baseline_df, qid) if show_baseline else None
        reasoning_row = row_by_id(reasoning_df, qid) if show_reasoning else None
        i5_rows_by_iter = rows_by_id_and_iteration(i5_df, qid) if show_i5 and i5_df is not None else []
        i5_row = (i5_rows_by_iter[0][1] if i5_rows_by_iter else row_by_id(i5_df, qid)) if show_i5 else None
        if not question_text and baseline_row is not None and "Translation" in baseline_row.index:
            question_text = baseline_row.get("Translation") or baseline_row.get("prompt") or ""
        if not question_text and reasoning_row is not None and "Translation" in reasoning_row.index:
            question_text = reasoning_row.get("Translation") or reasoning_row.get("prompt") or ""
        if not question_text and i5_row is not None and "Translation" in i5_row.index:
            question_text = i5_row.get("Translation") or i5_row.get("prompt") or ""
        ann_list = ann.get("annotations", [])
        st.subheader(str(qid))
        st.markdown("**Question:** " + (question_text or "(no question text)"))
        st.markdown("**Ground truth (annotations)**")
        if ann_list:
            for a in ann_list:
                en_a = a.get("en_answers", [])
                nat = a.get("answers", [])
                cnt = a.get("count", 1)
                parts = []
                if en_a:
                    parts.append("en: " + ", ".join(en_a))
                if nat:
                    parts.append("native: " + ", ".join(nat))
                st.caption("  [%s] %s" % (cnt, " | ".join(parts)))
        else:
            st.caption("(no annotations)")
        _k = "%s_%s" % (selected_country, qid)
        if view_mode == "Compare all":
            col_b, col_r, col_i5 = st.columns(3)
            with col_b:
                st.markdown("**Baseline response**")
                if baseline_row is not None:
                    st.text_area("b", value=baseline_row.get("response", ""), height=120, key="bl_%s" % _k, disabled=True)
                else:
                    st.caption("(no response for this question)")
            with col_r:
                st.markdown("**Only Reasoning response**")
                if reasoning_row is not None:
                    st.text_area("r", value=reasoning_row.get("response", ""), height=120, key="re_%s" % _k, disabled=True)
                else:
                    st.caption("(no response for this question)")
            with col_i5:
                st.markdown("**Iterative (i5) response**")
                if i5_rows_by_iter:
                    for it_num, r in i5_rows_by_iter:
                        st.markdown(f"**Iteration {it_num}**")
                        sb, sw = _get_i5_scores(i5_response_scores, qid, it_num)
                        st.caption("**SEM-B:** %s · **SEM-W:** %s" % (
                            "%.2f" % sb if sb is not None else "—",
                            "%.2f" % sw if sw is not None else "—"))
                        persona_val = r.get("persona", "")
                        if persona_val is not None and not (isinstance(persona_val, float) and __import__("math").isnan(persona_val)) and str(persona_val).strip():
                            with st.expander("Persona", expanded=False):
                                st.text_area("persona_i5", value=str(persona_val).strip(), height=120, key="persona_i5_%s_iter%s" % (_k, it_num), disabled=True, label_visibility="collapsed")
                        st.text_area("i5", value=r.get("response", ""), height=100, key="i5_%s_iter%s" % (_k, it_num), disabled=True)
                elif i5_row is not None:
                    persona_val = i5_row.get("persona", "")
                    if persona_val is not None and not (isinstance(persona_val, float) and __import__("math").isnan(persona_val)) and str(persona_val).strip():
                        with st.expander("Persona", expanded=False):
                            st.text_area("persona_i5_single", value=str(persona_val).strip(), height=120, key="persona_i5_%s" % _k, disabled=True, label_visibility="collapsed")
                    st.text_area("i5", value=i5_row.get("response", ""), height=120, key="i5_%s" % _k, disabled=True)
                else:
                    st.caption("(no response for this question)")
        else:
            row = baseline_row if show_baseline else (reasoning_row if show_reasoning else i5_row)
            if show_i5 and len(i5_rows_by_iter) > 1:
                st.markdown("**Full response (all iterations)**")
                for it_num, r in i5_rows_by_iter:
                    st.markdown(f"**Iteration {it_num}**")
                    sb, sw = _get_i5_scores(i5_response_scores, qid, it_num)
                    st.caption("**SEM-B:** %s · **SEM-W:** %s" % (
                        "%.2f" % sb if sb is not None else "—",
                        "%.2f" % sw if sw is not None else "—"))
                    persona_val = r.get("persona", "")
                    if persona_val is not None and not (isinstance(persona_val, float) and __import__("math").isnan(persona_val)) and str(persona_val).strip():
                        with st.expander("Persona", expanded=False):
                            st.text_area("persona", value=str(persona_val).strip(), height=120, key="persona_%s_iter%s" % (_k, it_num), disabled=True, label_visibility="collapsed")
                    st.text_area("resp", value=r.get("response", ""), height=150, key="resp_%s_iter%s" % (_k, it_num), disabled=True)
                    reason_val = r.get("reasoning", "")
                    if reason_val is not None and not (isinstance(reason_val, float) and __import__("math").isnan(reason_val)) and str(reason_val).strip():
                        st.caption("**Reasoning:** " + str(reason_val).strip()[:200] + ("..." if len(str(reason_val)) > 200 else ""))
                    if it_num < i5_rows_by_iter[-1][0]:
                        st.markdown("---")
            elif row is not None:
                st.markdown("**Full response**")
                persona_val = row.get("persona", "")
                if persona_val is not None and not (isinstance(persona_val, float) and __import__("math").isnan(persona_val)) and str(persona_val).strip():
                    with st.expander("Persona", expanded=False):
                        st.text_area("persona_single", value=str(persona_val).strip(), height=120, key="persona_%s" % _k, disabled=True, label_visibility="collapsed")
                st.text_area("resp", value=row.get("response", ""), height=180, key="resp_%s" % _k, disabled=True)
                reason_val = row.get("reasoning", "")
                if reason_val is not None and not (isinstance(reason_val, float) and __import__("math").isnan(reason_val)) and str(reason_val).strip():
                    st.markdown("**Reasoning**")
                    st.text_area("reason", value=str(reason_val).strip(), height=100, key="reason_%s" % _k, disabled=True)
            else:
                st.caption("(no response)")
        st.divider()


def main():
    st.title("📊 BLEnD Results Viewer")
    
    # Mode: MC or SAQ
    mode = st.sidebar.radio("Mode", ["Multiple Choice (MC)", "Short Answer (SAQ)"], index=0)
    
    if mode == "Short Answer (SAQ)":
        _main_saq()
        return
    
    st.markdown("### Visualize Multiple Choice Evaluation Results with Personas")
    
    # Sidebar for file selection
    st.sidebar.header("📁 Select Results File")
    
    available_files = get_available_results()
    
    if not available_files:
        st.error("No result CSV files found in the mc_data directory!")
        st.info("Expected files matching pattern: *-mc_res*.csv")
        return
    
    # File selection
    file_names = [f["name"] for f in available_files]
    selected_file_name = st.sidebar.selectbox(
        "CSV File",
        options=file_names,
        index=0
    )
    
    # Get the full path
    selected_file = next(f for f in available_files if f["name"] == selected_file_name)
    file_path = selected_file["path"]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**File:** `{selected_file['name']}`")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_csv_file(file_path)
    
    if df is None or len(df) == 0:
        st.warning("No data found in the selected file!")
        return
    
    # Check if required columns exist
    required_columns = ['MCQID', 'country', 'prompt', 'answer_idx', 'final_ans']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Ensure answer_idx and final_ans are strings for comparison
    if 'answer_idx' in df.columns:
        df['answer_idx'] = df['answer_idx'].astype(str)
    if 'final_ans' in df.columns:
        df['final_ans'] = df['final_ans'].astype(str)
    
    # Check if iteration and persona columns exist
    has_iteration = 'iteration' in df.columns
    has_persona = 'persona' in df.columns
    
    if has_iteration:
        max_iteration = int(df['iteration'].max())
        st.sidebar.info(f"**Iterations:** 1-{max_iteration}")
    else:
        st.sidebar.warning("No iteration column found")
    
    if has_persona:
        persona_count = df['persona'].notna().sum()
        st.sidebar.info(f"**Personas:** {persona_count} entries")
    else:
        st.sidebar.warning("No persona column found")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🌍 By Country", 
        "📝 Question Explorer", 
        "🎭 Persona Analysis",
        "🔄 Iterations"
    ])
    
    with tab1:
        st.header("Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_questions = len(df)
        overall_accuracy = calculate_accuracy(df)
        unique_countries = df['country'].nunique()
        unique_iterations = df['iteration'].nunique() if has_iteration else 1
        
        with col1:
            st.metric("Total Questions", total_questions)
        with col2:
            st.metric("Overall Accuracy", f"{overall_accuracy:.2f}%")
        with col3:
            st.metric("Countries", unique_countries)
        with col4:
            st.metric("Iterations", unique_iterations)
        
        # Accuracy by iteration
        if has_iteration:
            st.subheader("📊 Accuracy Over Iterations")
            iteration_accuracies = calculate_accuracy_by_iteration(df)
            
            if iteration_accuracies:
                iteration_data = [
                    {"iteration": k, "accuracy": v} 
                    for k, v in sorted(iteration_accuracies.items())
                ]
                df_iterations = pd.DataFrame(iteration_data)
                
                fig = px.line(
                    df_iterations,
                    x="iteration",
                    y="accuracy",
                    markers=True,
                    title="Accuracy Progression Across Iterations",
                    labels={"iteration": "Iteration", "accuracy": "Accuracy (%)"}
                )
                fig.update_traces(line_color="#1f77b4", marker=dict(size=10))
                fig.update_layout(hovermode="x unified")
                plotly_config = {
                    "width": "stretch"
                }
                st.plotly_chart(fig, config=plotly_config)
                
                # Show table
                st.dataframe(
                    df_iterations.style.format({"accuracy": "{:.2f}%"}),
                    hide_index=True,
                    width='stretch'
                )
    
    with tab2:
        st.header("🌍 Performance by Country")
        
        # Calculate accuracy per country
        country_data = []
        for country in sorted(df['country'].unique()):
            country_df = df[df['country'] == country]
            accuracy = calculate_accuracy(country_df)
            total = len(country_df)
            
            country_data.append({
                "Country": country,
                "Total Questions": total,
                "Accuracy (%)": accuracy
            })
        
        df_countries = pd.DataFrame(country_data).sort_values("Accuracy (%)", ascending=False)
        
        # Data table
        st.subheader("Overall Statistics")
        st.dataframe(
            df_countries.style.format({
                "Accuracy (%)": "{:.2f}%"
            }),
            hide_index=True,
            width='stretch'
        )
        
        # Bar chart
        fig = px.bar(
            df_countries,
            x="Country",
            y="Accuracy (%)",
            title="Accuracy by Country",
            color="Accuracy (%)",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )
        plotly_config = {
            "width": "stretch"
        }
        st.plotly_chart(fig, config=plotly_config)
        
        # Performance by country for each iteration
        if has_iteration:
            st.subheader("📊 Performance by Country per Iteration")
            
            iterations = sorted(df['iteration'].unique())
            for iteration in iterations:
                iter_df = df[df['iteration'] == iteration]
                
                country_accuracies = []
                for country in sorted(iter_df['country'].unique()):
                    country_df = iter_df[iter_df['country'] == country]
                    accuracy = calculate_accuracy(country_df)
                    country_accuracies.append({
                        "Country": country,
                        "Accuracy": accuracy
                    })
                
                # Sort by accuracy descending
                country_accuracies.sort(key=lambda x: x["Accuracy"], reverse=True)
                
                if country_accuracies:
                    countries_list = [ca["Country"] for ca in country_accuracies]
                    accuracies_list = [ca["Accuracy"] for ca in country_accuracies]
                    
                    fig = px.bar(
                        x=countries_list,
                        y=accuracies_list,
                        labels={"x": "Country", "y": "Accuracy (%)"},
                        title=f"Iteration {iteration}",
                        color=accuracies_list,
                        color_continuous_scale="RdYlGn"
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=400,
                        showlegend=False
                    )
                    plotly_config = {
                        "width": "stretch"
                    }
                    st.plotly_chart(fig, config=plotly_config)
    
    with tab3:
        st.header("📝 Question Explorer")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            countries = ["All"] + sorted(df['country'].unique().tolist())
            selected_country = st.selectbox("Filter by Country", countries)
        
        with col2:
            answer_filter = st.selectbox("Filter by Answer", ["All", "Correct", "Incorrect"])
        
        if has_iteration:
            max_iter = int(df['iteration'].max())
            st.caption(f"Model responses for **all {max_iter} iteration(s)** are shown for each question below.")
        
        # Search
        search_query = st.text_input("🔍 Search questions", "")
        
        # Filter dataframe
        filtered_df = df.copy()
        
        if selected_country != "All":
            filtered_df = filtered_df[filtered_df['country'] == selected_country]
        
        # Do not filter by iteration here: Question Explorer always shows all iterations per question (1–5)
        if search_query:
            filtered_df = filtered_df[
                filtered_df['prompt'].str.contains(search_query, case=False, na=False)
            ]
        
        # Group by question ID to show all iterations together
        question_groups = defaultdict(list)
        for _, row in filtered_df.iterrows():
            qid = row['MCQID']
            question_groups[qid].append(row)
        
        # Sort each group by iteration if available
        for qid in question_groups:
            if has_iteration:
                question_groups[qid] = sorted(
                    question_groups[qid], 
                    key=lambda x: x.get('iteration', 1)
                )
        
        # Apply answer filter
        filtered_questions = []
        for qid, rows in question_groups.items():
            # Check if latest iteration (or only iteration) is correct
            latest_row = rows[-1]
            is_correct = is_answer_correct(latest_row)
            
            if answer_filter == "All":
                filtered_questions.append({"qid": qid, "rows": rows})
            elif answer_filter == "Correct" and is_correct:
                filtered_questions.append({"qid": qid, "rows": rows})
            elif answer_filter == "Incorrect" and not is_correct:
                filtered_questions.append({"qid": qid, "rows": rows})
        
        # Pagination
        items_per_page = st.selectbox("Questions per page", [10, 25, 50, 100, "All"], index=2)
        
        if items_per_page == "All":
            items_per_page = len(filtered_questions)
        
        total_pages = (len(filtered_questions) + items_per_page - 1) // items_per_page if items_per_page > 0 else 1
        
        if total_pages > 1:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        else:
            page = 1
        
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_questions))
        
        st.info(f"Showing questions {start_idx + 1}-{end_idx} of {len(filtered_questions)}")
        
        # Display questions with all iterations
        for idx in range(start_idx, end_idx):
            question_data = filtered_questions[idx]
            rows = question_data["rows"]
            first_row = rows[0]
            
            # Check if most recent iteration is correct
            latest_row = rows[-1]
            is_correct = is_answer_correct(latest_row)
            
            # Get question text (truncate if too long)
            prompt_text = first_row.get('prompt', 'N/A')
            prompt_display = prompt_text[:150] + "..." if len(prompt_text) > 150 else prompt_text
            
            with st.expander(
                f"{'✅' if is_correct else '❌'} Q{idx+1}: {prompt_display} ({len(rows)} iteration{'s' if len(rows) > 1 else ''})",
                expanded=False
            ):
                st.markdown(f"**Question ID:** {first_row.get('MCQID', 'N/A')}")
                st.markdown(f"**Question:** {prompt_text}")
                
                # Show choices if available
                if 'choices' in first_row and pd.notna(first_row['choices']):
                    try:
                        choices = json.loads(first_row['choices'])
                        st.markdown("**Choices:**")
                        for key, value in choices.items():
                            st.markdown(f"  - **{key}:** {value}")
                    except:
                        pass
                
                st.markdown(f"**Country:** {first_row.get('country', 'Unknown')}")
                st.markdown(f"**Correct Answer:** {first_row.get('answer_idx', 'N/A')}")
                st.markdown("---")
                
                # Display each iteration
                for iter_idx, row in enumerate(rows):
                    iter_is_correct = is_answer_correct(row)
                    iteration_num = row.get('iteration', 1) if has_iteration else 1
                    
                    st.markdown(f"### {'✅' if iter_is_correct else '❌'} Iteration {iteration_num}")
                    
                    # Model answer
                    final_ans = row.get('final_ans', 'N/A')
                    answer_idx = row.get('answer_idx', 'N/A')
                    st.markdown(f"**Model Answer:** {final_ans} {'✅ Correct' if iter_is_correct else '❌ Incorrect'} (Expected: {answer_idx})")
                    
                    # Reasoning (if available)
                    if 'reasoning' in row and pd.notna(row['reasoning']) and str(row['reasoning']).strip():
                        st.markdown("**💭 Reasoning:**")
                        st.info(row['reasoning'])
                    
                    # Full response
                    if 'full_res' in row and pd.notna(row['full_res']):
                        with st.expander("📄 Full Response", expanded=False):
                            st.text(row['full_res'])
                    
                    # Persona description
                    if has_persona and 'persona' in row and pd.notna(row['persona']):
                        st.markdown("**🎭 Persona:**")
                        st.info(row['persona'])
                    
                    # Separator between iterations
                    if iter_idx < len(rows) - 1:
                        st.markdown("---")
    
    with tab4:
        st.header("🎭 Persona Analysis")
        
        if not has_persona:
            st.warning("No persona column found in this dataset.")
        else:
            # Count personas
            persona_count = df['persona'].notna().sum()
            st.info(f"Found {persona_count} persona entries")
            
            # Sample personas by country
            st.subheader("Sample Personas by Country")
            
            country_personas = {}
            for _, row in df.iterrows():
                if pd.notna(row.get('persona')):
                    country = row.get('country', 'Unknown')
                    if country not in country_personas:
                        country_personas[country] = []
                    # Add unique personas (by text)
                    persona_text = str(row['persona'])
                    if persona_text not in country_personas[country]:
                        country_personas[country].append(persona_text)
            
            # Show one persona per country
            for country in sorted(country_personas.keys()):
                with st.expander(f"🌍 {country} ({len(country_personas[country])} unique persona{'s' if len(country_personas[country]) > 1 else ''})"):
                    # Show first persona, or allow selection if multiple
                    if len(country_personas[country]) == 1:
                        st.write(country_personas[country][0])
                    else:
                        selected_persona_idx = st.selectbox(
                            f"Select persona for {country}",
                            range(len(country_personas[country])),
                            format_func=lambda x: f"Persona {x+1}",
                            key=f"persona_select_{country}"
                        )
                        st.write(country_personas[country][selected_persona_idx])
            
            # Persona evolution across iterations (if available)
            if has_iteration:
                st.subheader("🔄 Persona Evolution Across Iterations")
                
                # Group by question ID and show persona changes
                question_personas = defaultdict(dict)
                for _, row in df.iterrows():
                    if pd.notna(row.get('persona')):
                        qid = row['MCQID']
                        iteration = row.get('iteration', 1)
                        question_personas[qid][iteration] = row['persona']
                
                # Show a few examples
                example_count = st.slider("Number of examples to show", 1, 10, 3)
                example_qids = list(question_personas.keys())[:example_count]
                
                for qid in example_qids:
                    with st.expander(f"Question {qid} - Persona Evolution"):
                        iterations = sorted(question_personas[qid].keys())
                        for iteration in iterations:
                            st.markdown(f"**Iteration {iteration}:**")
                            st.info(question_personas[qid][iteration])
                            if iteration < max(iterations):
                                st.markdown("---")
    
    with tab5:
        st.header("🔄 Iteration Analysis")
        
        if not has_iteration:
            st.warning("No iteration column found in this dataset.")
        else:
            # Group data by iteration
            iteration_items = defaultdict(list)
            iteration_countries = defaultdict(set)
            
            for _, row in df.iterrows():
                iteration = row.get('iteration', 1)
                iteration_items[iteration].append(row)
                iteration_countries[iteration].add(row.get('country', 'Unknown'))
            
            # Create dataframe
            iteration_data = []
            for iteration in sorted(iteration_items.keys()):
                rows = iteration_items[iteration]
                iter_df = pd.DataFrame(rows)
                accuracy = calculate_accuracy(iter_df)
                total = len(rows)
                
                iteration_data.append({
                    "Iteration": iteration,
                    "Total Questions": total,
                    "Accuracy (%)": accuracy,
                    "Countries": len(iteration_countries[iteration])
                })
            
            df_iterations = pd.DataFrame(iteration_data)
            
            # Display table
            st.subheader("Performance by Iteration")
            st.dataframe(
                df_iterations.style.format({
                    "Accuracy (%)": "{:.2f}%"
                }),
                hide_index=True,
                width='stretch'
            )
            
            # Line chart
            if len(df_iterations) > 1:
                fig = px.line(
                    df_iterations,
                    x="Iteration",
                    y="Accuracy (%)",
                    markers=True,
                    title="Accuracy Progression Across Iterations"
                )
                fig.update_traces(line_color="#1f77b4", marker=dict(size=12))
                plotly_config = {
                    "width": "stretch"
                }
                st.plotly_chart(fig, config=plotly_config)
            
            # Answer distribution for each iteration
            st.subheader("📊 Answer Distribution by Iteration")
            
            # Create columns for bar charts (max 3 per row)
            iterations_sorted = sorted(iteration_items.keys())
            for i in range(0, len(iterations_sorted), 3):
                cols = st.columns(3)
                for j, iteration in enumerate(iterations_sorted[i:i+3]):
                    rows = iteration_items[iteration]
                    
                    # Calculate answer distribution for this iteration
                    final_answers = [str(row.get('final_ans', 'unknown')).upper() for row in rows]
                    answer_dist = Counter(final_answers)
                    
                    # Sort answers (A, B, C, D, etc.)
                    sorted_answers = sorted([k for k in answer_dist.keys() if k in ['A', 'B', 'C', 'D', 'E', 'F']])
                    sorted_counts = [answer_dist.get(ans, 0) for ans in sorted_answers]
                    
                    # Color map
                    color_map = {
                        "A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", 
                        "D": "#d62728", "E": "#9467bd", "F": "#8c564b"
                    }
                    sorted_colors = [color_map.get(ans, "#gray") for ans in sorted_answers]
                    
                    with cols[j]:
                        if sorted_answers:
                            fig = go.Figure(data=[go.Bar(
                                x=sorted_answers,
                                y=sorted_counts,
                                marker=dict(color=sorted_colors)
                            )])
                            fig.update_layout(
                                title=f"Iteration {iteration}",
                                xaxis_title="Answer",
                                yaxis_title="Count",
                                height=300,
                                showlegend=False
                            )
                            plotly_config = {
                                "width": "stretch"
                            }
                            st.plotly_chart(fig, config=plotly_config)
                        else:
                            st.info(f"No valid answers found for iteration {iteration}")

if __name__ == "__main__":
    main()

