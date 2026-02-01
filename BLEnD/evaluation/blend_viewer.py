"""Streamlit UI for visualizing BLEnD multiple choice evaluation results."""

import streamlit as st
import json
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict

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

def main():
    st.title("📊 BLEnD Results Viewer")
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
            if has_iteration:
                iterations = ["All"] + sorted(df['iteration'].unique().tolist())
                selected_iteration = st.selectbox("Filter by Iteration", iterations)
            else:
                selected_iteration = "All"
        
        answer_filter = st.selectbox("Filter by Answer", ["All", "Correct", "Incorrect"])
        
        # Search
        search_query = st.text_input("🔍 Search questions", "")
        
        # Filter dataframe
        filtered_df = df.copy()
        
        if selected_country != "All":
            filtered_df = filtered_df[filtered_df['country'] == selected_country]
        
        if selected_iteration != "All" and has_iteration:
            filtered_df = filtered_df[filtered_df['iteration'] == selected_iteration]
        
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
