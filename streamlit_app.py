"""Streamlit UI for visualizing MC-Personas evaluation results."""

import streamlit as st
import json
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Add evaluation directory to path for imports
sys.path.append(str(Path(__file__).parent / "evaluation"))
from evaluation.tools.db.db_utils import load_results, get_all_iterations, get_accuracies

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
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        # Get available iterations and sort them
        available_iterations = sorted(set(item.get("iteration", 1) for item in data))
        
        with col1:
            # Iteration filter (no "All" option, default to iteration 1)
            default_iteration_idx = 0  # Default to first iteration (usually 1)
            if 1 in available_iterations:
                default_iteration_idx = available_iterations.index(1)
            
            selected_iteration = st.selectbox(
                "Iteration", 
                available_iterations,
                index=default_iteration_idx,
                help="Select which iteration to view questions from"
            )
        
        with col2:
            countries = ["All"] + sorted(set(item.get("country", "Unknown") for item in data))
            selected_country = st.selectbox("Filter by Country", countries)
        
        with col3:
            answer_filter = st.selectbox("Filter by Answer", ["All", "Correct", "Incorrect"])
        
        # Apply filters - iteration filter is ALWAYS applied
        filtered_data = [item for item in data if item.get("iteration", 1) == selected_iteration]
        
        if selected_country != "All":
            filtered_data = [item for item in filtered_data if item.get("country").lower() == selected_country.lower()]
        
        if answer_filter == "Correct":
            filtered_data = [item for item in filtered_data if is_answer_correct(item)]
        elif answer_filter == "Incorrect":
            filtered_data = [item for item in filtered_data if not is_answer_correct(item)]
        
        st.info(f"Showing {len(filtered_data)} questions from Iteration {selected_iteration}")
        
        # Search
        search_query = st.text_input("🔍 Search questions", "")
        if search_query:
            filtered_data = [
                item for item in filtered_data 
                if search_query.lower() in item.get("question", "").lower()
            ]
        
        # Display questions
        for idx, item in enumerate(filtered_data[:50]):  # Limit to 50 for performance
            is_correct = is_answer_correct(item)
            
            with st.expander(
                f"{'✅' if is_correct else '❌'} Q{idx+1}: {item.get('question', 'No question')[:100]}...",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Question:** {item.get('question', 'N/A')}")
                    
                    # Show options for Easy mode or prompt_option for Hard mode
                    options = item.get('options')
                    if options and isinstance(options, dict):
                        st.markdown("**Options:**")
                        for key, value in options.items():
                            st.markdown(f"  - {key}: {value}")
                    elif item.get('prompt_option'):
                        st.markdown(f"**Option:** {item.get('prompt_option', 'N/A')}")
                    
                    st.markdown(f"**Reasoning:** {item.get('reasoning', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Country:** {item.get('country', 'Unknown')}")
                    if "iteration" in item:
                        st.markdown(f"**Iteration:** {item.get('iteration', 'N/A')}")
                    st.markdown(f"**Correct Answer:** {item.get('correct_answer', 'N/A')}")
                    
                    # Show the appropriate answer field
                    # Get model answer from any available field
                    model_answer = item.get('model_answer') or item.get('persona_answer') or item.get('vanilla_answer', 'N/A')
                    st.markdown(f"**Model Answer:** {model_answer}")
                    
                    st.markdown(f"**Result:** {'✅ Correct' if is_correct else '❌ Incorrect'}")
                
                if item.get("persona_description"):
                    st.markdown("**Persona:**")
                    st.info(item.get("persona_description"))
        
        if len(filtered_data) > 50:
            st.warning(f"Showing first 50 of {len(filtered_data)} questions. Apply more filters to see specific results.")
    
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

