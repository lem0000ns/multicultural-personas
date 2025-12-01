"""Streamlit UI for visualizing MC-Personas evaluation results."""

import streamlit as st
import json
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Add culturalbench directory to path for imports
sys.path.append(str(Path(__file__).parent / "culturalbench"))
from culturalbench.tools.db.db_utils import load_results, get_accuracies

# Set page config
st.set_page_config(
    page_title="MC-Personas Results Viewer",
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
def load_db_file(db_path):
    """Load data from SQLite database and return as list of dicts."""
    data = load_results(db_path)
    
    # Get accuracy summary
    accuracies = get_accuracies(db_path)
    summary_lines = []
    for acc in accuracies:
        summary_line = f"Persona Accuracy for {acc['difficulty']} - Iteration {acc['iteration']}: {acc['accuracy']:.4f}"
        summary_lines.append(summary_line)
    
    return data, summary_lines

def get_available_results():
    """Scan the results directory for available database files.
    
    Returns:
        Dictionary organized as: {mode: {prompt: [files]}}
    """
    results_dir = Path("/home/kevinyang/mc-personas/results")
    db_files = list(results_dir.rglob("*.db"))
    
    # Organize by mode and prompt
    # Structure: {mode: {prompt: [files]}}
    organized = {}
    
    for file in db_files:
        relative_path = file.relative_to(results_dir)
        parts = relative_path.parts
        
        if parts[0] == "vanilla":
            mode = "vanilla"
            prompt = None  # vanilla doesn't have prompt
        else:
            # parts[0] is like 'p1' or 'p2'
            # parts[1] is like 'eng', 'ling', 'e2l', 'l2e'
            prompt = parts[0]  # p1, p2
            mode = parts[1]    # eng, ling, e2l, l2e
        
        if mode not in organized:
            organized[mode] = {}
        
        if mode == "vanilla":
            if "all" not in organized[mode]:
                organized[mode]["all"] = []
            organized[mode]["all"].append({
                "path": str(file),
                "name": relative_path.name,
                "relative_path": str(relative_path)
            })
        else:
            if prompt not in organized[mode]:
                organized[mode][prompt] = []
            organized[mode][prompt].append({
                "path": str(file),
                "name": relative_path.name,
                "relative_path": str(relative_path)
            })
    
    return organized

def is_single_item_correct(item):
    """Check if a single item's answer is correct (for individual line checking)."""
    correct_answer = item.get("correct_answer")
    
    # Get the model's answer (could be model_answer, persona_answer, or vanilla_answer)
    model_answer = item.get("model_answer") or item.get("persona_answer") or item.get("vanilla_answer")
    
    if model_answer is None:
        return False
    
    # Check if it's Hard mode (has 'prompt_option') or Easy mode (has options with data)
    is_hard_mode = bool(item.get("prompt_option"))
    
    if is_hard_mode:
        # Hard mode: compare as strings (both "true" or "false")
        # Normalize both to lowercase strings for comparison
        model_answer_str = str(model_answer).lower().strip()
        correct_answer_str = str(correct_answer).lower().strip()
        
        return model_answer_str == correct_answer_str
    else:
        # Easy mode: compare strings directly (A/B/C/D)
        return correct_answer.lower() == model_answer.lower()

def is_answer_correct(item):
    """Wrapper for backward compatibility."""
    return is_single_item_correct(item)

def calculate_accuracy(data):
    """Calculate accuracy from the data based on the format."""
    if not data:
        return 0.0
    
    # Detect format - check if it's Hard mode (has prompt_option) or Easy mode (has options with data)
    # Hard mode: has prompt_option
    # Easy mode: has options dict with actual option data (A, B, C, D keys)
    is_hard_mode = bool(data[0].get("prompt_option"))
    
    if is_hard_mode:
        # For Hard mode: group by question and check if ALL options are correct
        from collections import defaultdict
        question_groups = defaultdict(list)
        
        for item in data:
            question = item.get("question", "")
            question_groups[question].append(item)
        
        correct_questions = 0
        total_questions = len(question_groups)
        
        for question, items in question_groups.items():
            # A question is correct only if ALL its options are answered correctly
            if all(is_single_item_correct(item) for item in items):
                correct_questions += 1
        
        return (correct_questions / total_questions) * 100 if total_questions > 0 else 0.0
    else:
        # For Easy mode: simple count of correct answers
        correct = sum(1 for item in data if is_single_item_correct(item))
        return (correct / len(data)) * 100

def extract_iteration_accuracies(summary_lines):
    """Extract accuracy values from summary lines."""
    accuracies = []
    for line in summary_lines:
        # Parse line like "Persona Accuracy for Hard - Iteration 1: 0.7500"
        try:
            parts = line.split(":")
            if len(parts) == 2:
                accuracy = float(parts[1].strip())
                iteration = int(line.split("Iteration")[1].split(":")[0].strip())
                accuracies.append({"iteration": iteration, "accuracy": accuracy * 100})
        except:
            continue
    return sorted(accuracies, key=lambda x: x["iteration"])

def main():
    st.title("📊 MC-Personas Results Viewer")
    st.markdown("### Visualize CulturalBench Evaluation Results")
    
    # Sidebar for file selection
    st.sidebar.header("📁 Select Results")
    
    available_results = get_available_results()
    
    if not available_results:
        st.error("No results files found in the results directory!")
        return
    
    # Mode selection (eng, ling, e2l, l2e, vanilla)
    mode_display_names = {
        "eng": "🇺🇸 English",
        "ling": "🌍 Linguistic",
        "e2l": "🇺🇸→🌍 English to Local",
        "l2e": "🌍→🇺🇸 Local to English",
        "vanilla": "⚪ Vanilla (No Persona)"
    }
    
    available_modes = sorted(available_results.keys())
    mode_options = [mode_display_names.get(m, m.upper()) for m in available_modes]
    
    selected_mode_display = st.sidebar.selectbox(
        "Mode",
        options=mode_options,
        index=0
    )
    
    # Get the actual mode key from display name
    selected_mode = available_modes[mode_options.index(selected_mode_display)]
    
    # Prompt selection (p1, p2) - only if not vanilla
    if selected_mode != "vanilla":
        available_prompts = sorted(available_results[selected_mode].keys())
        prompt_display_names = {
            "p1": "📝 Prompt 1",
            "p2": "📝 Prompt 2"
        }
        prompt_options = [prompt_display_names.get(p, p.upper()) for p in available_prompts]
        
        selected_prompt_display = st.sidebar.selectbox(
            "Prompt",
            options=prompt_options,
            index=0
        )
        
        selected_prompt = available_prompts[prompt_options.index(selected_prompt_display)]
        files = available_results[selected_mode][selected_prompt]
    else:
        # For vanilla, no prompt selection needed
        files = available_results[selected_mode]["all"]
        selected_prompt = "N/A"
    
    # File selection (difficulty and iteration)
    if not files:
        st.error(f"No files found for {selected_mode_display}")
        return
    
    file_names = [f["name"] for f in files]
    selected_file_name = st.sidebar.selectbox(
        "Dataset",
        options=file_names,
        index=0
    )
    
    # Get the full path
    selected_file = next(f for f in files if f["name"] == selected_file_name)
    file_path = selected_file["path"]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Mode:** {selected_mode_display}")
    if selected_mode != "vanilla":
        st.sidebar.info(f"**Prompt:** {selected_prompt_display}")
    st.sidebar.info(f"**File:** `{selected_file['name']}`")
    
    # Load data
    with st.spinner("Loading data..."):
        data, summary_lines = load_db_file(file_path)
    
    if not data:
        st.warning("No data found in the selected file!")
        return
    
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
        col1, col2, col3 = st.columns(3)
        
        total_questions = len(data)
        
        accuracy = calculate_accuracy(data)
        unique_countries = len(set(item.get("country", "Unknown") for item in data))
        unique_iterations = len(set(item.get("iteration", 1) for item in data))
        
        with col1:
            st.metric("Total Questions", total_questions)
        with col2:
            st.metric("Countries", unique_countries)
        with col3:
            st.metric("Iterations", unique_iterations)
        
        # Accuracy by iteration
        if summary_lines:
            st.subheader("📊 Accuracy Over Iterations")
            iteration_data = extract_iteration_accuracies(summary_lines)
            
            if iteration_data:
                df_iterations = pd.DataFrame(iteration_data)
                
                fig = px.line(
                    df_iterations,
                    x="iteration",
                    y="accuracy",
                    markers=True,
                    title="Accuracy Progression",
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
        
        # Group data by country first
        from collections import defaultdict
        country_items = defaultdict(list)
        for item in data:
            country = item.get("country", "Unknown")
            country_items[country].append(item)
        
        # Calculate accuracy per country using the same logic as overall accuracy
        country_data = []
        for country, items in country_items.items():
            accuracy = calculate_accuracy(items)
            
            # For display: count questions properly
            # Hard mode: has prompt_option
            is_hard_mode = bool(items[0].get("prompt_option"))
            if is_hard_mode:
                # Count unique questions
                unique_questions = len(set(item.get("question", "") for item in items))
                total_display = unique_questions
            else:
                total_display = len(items)
            
            country_data.append({
                "Country": country,
                "Total Questions": total_display,
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
        
        # Performance by country for each iteration
        if "iteration" in data[0]:
            st.subheader("📊 Performance by Country per Iteration")
            
            # Group by iteration and country
            iteration_country_items = defaultdict(lambda: defaultdict(list))
            for item in data:
                iteration = item.get("iteration", 1)
                country = item.get("country", "Unknown")
                iteration_country_items[iteration][country].append(item)
            
            # Create bar charts for each iteration (vertically stacked)
            iterations = sorted(iteration_country_items.keys())
            for iteration in iterations:
                country_accuracies = []
                for country, items in iteration_country_items[iteration].items():
                    accuracy = calculate_accuracy(items)
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
        
        # check if qwen3 to show thinking content
        is_qwen3_4b = "qwen3_4b" in file_path
        
        # Detect if this is Hard mode (has prompt_option) or Easy mode
        is_hard_mode = bool(data[0].get("prompt_option")) if data else False
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            countries = ["All"] + sorted(set(item.get("country", "Unknown") for item in data))
            selected_country = st.selectbox("Filter by Country", countries)
        
        with col2:
            answer_filter = st.selectbox("Filter by Answer", ["All", "Correct (Any Iteration)", "Incorrect (Any Iteration)"])
        
        # Search
        search_query = st.text_input("🔍 Search questions", "")
        
        from collections import defaultdict
        
        if is_hard_mode:
            # Hard mode: Group by question, show all 4 options together across iterations
            question_groups = defaultdict(lambda: defaultdict(list))
            
            for item in data:
                question = item.get("question", "")
                iteration = item.get("iteration", 1)
                question_groups[question][iteration].append(item)
            
            # Filter and collect question sets
            filtered_sets = []
            for question, iterations_dict in question_groups.items():
                first_iter = min(iterations_dict.keys())
                first_items = iterations_dict[first_iter]
                
                # Country filter
                if selected_country != "All":
                    if first_items[0].get("country", "").lower() != selected_country.lower():
                        continue
                
                # Search filter
                if search_query and search_query.lower() not in question.lower():
                    continue
                
                # Answer filter
                passes_filter = False
                if answer_filter == "All":
                    passes_filter = True
                else:
                    for iteration, option_items in iterations_dict.items():
                        all_correct = all(is_answer_correct(item) for item in option_items)
                        any_incorrect = any(not is_answer_correct(item) for item in option_items)
                        
                        if answer_filter == "Correct (Any Iteration)" and all_correct:
                            passes_filter = True
                            break
                        elif answer_filter == "Incorrect (Any Iteration)" and any_incorrect:
                            passes_filter = True
                            break
                
                if passes_filter:
                    filtered_sets.append({
                        "question": question,
                        "iterations": iterations_dict,
                        "first_item": first_items[0]
                    })
            
            # Pagination
            items_per_page = st.selectbox("Question sets per page", [10, 25, 50, "All"], index=1)
            
            if items_per_page == "All":
                items_per_page = len(filtered_sets)
            
            total_pages = (len(filtered_sets) + items_per_page - 1) // items_per_page if items_per_page > 0 else 1
            
            if total_pages > 1:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            else:
                page = 1
            
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_sets))
            
            st.info(f"Showing question sets {start_idx + 1}-{end_idx} of {len(filtered_sets)}")
            
            # Display Hard mode questions
            for idx in range(start_idx, end_idx):
                question_set = filtered_sets[idx]
                iterations_dict = question_set["iterations"]
                first_item = question_set["first_item"]
                
                # Check if latest iteration has all options correct
                latest_iter = max(iterations_dict.keys())
                latest_options = iterations_dict[latest_iter]
                all_correct = all(is_answer_correct(item) for item in latest_options)
                
                with st.expander(
                    f"{'✅' if all_correct else '❌'} Q{idx+1}: {question_set['question'][:100]}... ({len(iterations_dict)} iterations)",
                    expanded=False
                ):
                    st.markdown(f"**Question:** {question_set['question']}")
                    st.markdown(f"**Country:** {first_item.get('country', 'Unknown')}")
                    st.markdown("---")
                    
                    # Display each iteration
                    for iteration in sorted(iterations_dict.keys()):
                        option_items = iterations_dict[iteration]
                        iter_all_correct = all(is_answer_correct(item) for item in option_items)
                        
                        st.markdown(f"### {'✅' if iter_all_correct else '❌'} Iteration {iteration}")
                        
                        # Show persona (same for all 4 options)
                        if option_items[0].get("persona_description"):
                            st.markdown("**🎭 Persona:**")
                            st.info(option_items[0].get("persona_description"))
                        
                        # Show refine reasoning
                        if option_items[0].get("refine_reasoning"):
                            st.markdown("**🔄 Self-Refinement Reasoning:**")
                            st.warning(option_items[0].get("refine_reasoning"))
                        
                        # Show all 4 options
                        st.markdown("**Options & Answers:**")
                        for option_item in option_items:
                            option_text = option_item.get("prompt_option", "")
                            model_answer = option_item.get("model_answer", "N/A")
                            correct_answer = option_item.get("correct_answer", "N/A")
                            
                            correct_str = "True" if str(correct_answer) in ["1", "true", "True"] else "False"
                            model_str = "True" if str(model_answer).lower() == "true" else "False"
                            is_option_correct = is_answer_correct(option_item)
                            
                            st.markdown(
                                f"{'✅' if is_option_correct else '❌'} **{option_text}** → "
                                f"Model: {model_str}, Correct: {correct_str}"
                            )
                            
                            # Show reasoning for this option
                            if option_item.get("reasoning"):
                                with st.expander(f"💬 Reasoning: {option_text[:50]}...", expanded=False):
                                    st.text(option_item.get("reasoning"))
                            
                            # Show thinking content
                            if is_qwen3_4b and option_item.get("thinking_content"):
                                with st.expander(f"💭 Thinking: {option_text[:50]}...", expanded=False):
                                    st.text_area(
                                        "Model's internal reasoning",
                                        option_item.get("thinking_content"),
                                        height=200,
                                        key=f"thinking_{idx}_{iteration}_{option_text[:20]}",
                                        disabled=True
                                    )
                        
                        if iteration < max(iterations_dict.keys()):
                            st.markdown("---")
        
        else:
            # Easy mode: Group by question, show all iterations
            question_groups = defaultdict(list)
            
            for item in data:
                question = item.get("question", "")
                question_groups[question].append(item)
            
            # Sort each group by iteration
            for question in question_groups:
                question_groups[question] = sorted(question_groups[question], key=lambda x: x.get("iteration", 1))
            
            # Filter and collect questions
            filtered_questions = []
            for question, items in question_groups.items():
                first_item = items[0]
                
                # Country filter
                if selected_country != "All":
                    if first_item.get("country", "").lower() != selected_country.lower():
                        continue
                
                # Search filter
                if search_query and search_query.lower() not in question.lower():
                    continue
                
                # Answer filter (check any iteration)
                passes_filter = False
                if answer_filter == "All":
                    passes_filter = True
                else:
                    for item in items:
                        if answer_filter == "Correct (Any Iteration)" and is_answer_correct(item):
                            passes_filter = True
                            break
                        elif answer_filter == "Incorrect (Any Iteration)" and not is_answer_correct(item):
                            passes_filter = True
                            break
                
                if passes_filter:
                    filtered_questions.append({
                        "question": question,
                        "items": items,
                        "first_item": first_item
                    })
            
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
                items = question_data["items"]
                first_item = question_data["first_item"]
                
                # Check if most recent iteration is correct
                latest_item = items[-1]
                is_correct = is_answer_correct(latest_item)
                
                with st.expander(
                    f"{'✅' if is_correct else '❌'} Q{idx+1}: {question_data['question'][:100]}... ({len(items)} iterations)",
                    expanded=False
                ):
                    st.markdown(f"**Question:** {question_data['question']}")
                    
                    # Show options
                    options = first_item.get('options')
                    if options and isinstance(options, dict):
                        st.markdown("**Options:**")
                        for key, value in options.items():
                            st.markdown(f"  - {key}: {value}")
                    
                    st.markdown(f"**Country:** {first_item.get('country', 'Unknown')}")
                    st.markdown(f"**Correct Answer:** {first_item.get('correct_answer', 'N/A')}")
                    st.markdown("---")
                    
                    # Display each iteration
                    for iter_idx, item in enumerate(items):
                        iter_is_correct = is_answer_correct(item)
                        iteration_num = item.get('iteration', 1)
                        
                        st.markdown(f"### {'✅' if iter_is_correct else '❌'} Iteration {iteration_num}")
                        
                        # Model answer
                        model_answer = item.get('model_answer') or item.get('persona_answer') or item.get('vanilla_answer', 'N/A')
                        st.markdown(f"**Model Answer:** {model_answer} {'✅ Correct' if iter_is_correct else '❌ Incorrect'}")
                        
                        # Persona description
                        if item.get("persona_description"):
                            st.markdown("**🎭 Persona:**")
                            st.info(item.get("persona_description"))
                        
                        # Refine reasoning (from self-refinement)
                        if item.get("refine_reasoning"):
                            st.markdown("**🔄 Self-Refinement Reasoning:**")
                            st.warning(item.get("refine_reasoning"))
                        
                        # Answer reasoning
                        if item.get("reasoning"):
                            st.markdown("**💬 Answer Reasoning:**")
                            st.text(item.get("reasoning"))
                        
                        # Thinking content for Qwen3-4B models
                        if is_qwen3_4b and item.get("thinking_content"):
                            st.markdown("**💭 Thinking Content:**")
                            with st.container():
                                st.text_area(
                                    f"Model's internal reasoning (Iteration {iteration_num})",
                                    item.get("thinking_content"),
                                    height=200,
                                    key=f"thinking_easy_{idx}_{iter_idx}",
                                    disabled=True
                                )
                        
                        # Separator between iterations
                        if iter_idx < len(items) - 1:
                            st.markdown("---")
    
    with tab4:
        st.header("🎭 Persona Analysis")
        
        # Check if personas exist
        personas = [item.get("persona_description", "") for item in data if item.get("persona_description")]
        
        if not personas:
            st.warning("No persona descriptions found in this dataset.")
        else:
            st.info(f"Found {len(personas)} persona descriptions")
            
            # Sample personas
            st.subheader("Sample Personas")
            
            # Group by country
            country_personas = {}
            for item in data:
                if item.get("persona_description"):
                    country = item.get("country", "Unknown")
                    if country not in country_personas:
                        country_personas[country] = []
                    country_personas[country].append(item.get("persona_description"))
            
            # Show one persona per country
            for country in sorted(country_personas.keys()):
                with st.expander(f"🌍 {country}"):
                    st.write(country_personas[country][0])
    
    with tab5:
        st.header("🔄 Iteration Analysis")
        
        # Group data by iteration first
        iteration_items = defaultdict(list)
        iteration_countries = defaultdict(set)
        
        for item in data:
            iteration = item.get("iteration", 1)
            iteration_items[iteration].append(item)
            iteration_countries[iteration].add(item.get("country", "Unknown"))
        
        # Create dataframe
        iteration_data = []
        for iteration, items in sorted(iteration_items.items()):
            accuracy = calculate_accuracy(items)
            
            # For display: count questions properly
            # Hard mode: has prompt_option
            is_hard_mode = bool(items[0].get("prompt_option"))
            if is_hard_mode:
                # Count unique questions
                unique_questions = len(set(item.get("question", "") for item in items))
                total_display = unique_questions
            else:
                total_display = len(items)
            
            iteration_data.append({
                "Iteration": iteration,
                "Total Questions": total_display,
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
        
        # Check which answer field to use
        # Determine which answer field is available
        if "model_answer" in data[0]:
            answer_field = "model_answer"
        elif "persona_answer" in data[0]:
            answer_field = "persona_answer"
        else:
            answer_field = "vanilla_answer"
        
        # Detect mode and set up ordering and colors
        # Hard mode: has prompt_option
        is_hard_mode = bool(data[0].get("prompt_option"))
        if is_hard_mode:
            # Hard mode: true, false (display capitalized)
            answer_order = ["true", "false"]
            answer_display = ["True", "False"]
            color_map = {"true": "#28a745", "false": "#dc3545"}  # green, red
        else:
            # Easy mode: A, B, C, D (already uppercase for display)
            answer_order = ["a", "b", "c", "d"]
            answer_display = ["A", "B", "C", "D"]
            color_map = {"a": "#1f77b4", "b": "#ff7f0e", "c": "#2ca02c", "d": "#d62728"}  # blue, orange, green, red
        
        # Create columns for bar charts (max 3 per row)
        iterations_sorted = sorted(iteration_items.keys())
        for i in range(0, len(iterations_sorted), 3):
            cols = st.columns(3)
            for j, iteration in enumerate(iterations_sorted[i:i+3]):
                items = iteration_items[iteration]
                
                # Calculate answer distribution for this iteration
                raw_answers = [item.get(answer_field, "unknown") for item in items]
                # Normalize to lowercase for consistency, then count
                answer_dist = Counter(ans.lower() if isinstance(ans, str) else str(ans) for ans in raw_answers)
                
                # Sort answers according to the defined order
                sorted_answers = [ans for ans in answer_order if ans in answer_dist]
                sorted_counts = [answer_dist[ans] for ans in sorted_answers]
                sorted_colors = [color_map.get(ans, "#gray") for ans in sorted_answers]
                # Get display labels
                sorted_display = [answer_display[answer_order.index(ans)] for ans in sorted_answers]
                
                with cols[j]:
                    fig = go.Figure(data=[go.Bar(
                        x=sorted_display,
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

if __name__ == "__main__":
    main()

