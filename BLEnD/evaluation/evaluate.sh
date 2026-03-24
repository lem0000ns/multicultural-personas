#!/bin/bash

# Define model keys
# MODEL_KEYS=(
#     "gpt-4-1106-preview"
#     "gpt-3.5-turbo-1106"
#     "aya-101"
#     "gemini-pro"
#     "claude-3-opus-20240229"
#     "claude-3-sonnet-20240229"
#     "claude-3-haiku-20240307"
#     "Qwen1.5-72B-Chat"
#     "Qwen1.5-14B-Chat"
#     "Qwen1.5-32B-Chat"
#     "text-bison-002"
#     "c4ai-command-r-v01"
#     "c4ai-command-r-plus"
# )

MODEL_KEYS=("qwen3-14b")
# Define countries and languages as parallel arrays
COUNTRIES=("UK" "US" "South_Korea" "Algeria" "China" "Indonesia" "Spain" "Iran" "Mexico" "Assam" "Greece" "Ethiopia" "Northern_Nigeria" "Azerbaijan" "North_Korea" "West_Java")

# Iterate over models, countries, languages, and prompts
for model_key in "${MODEL_KEYS[@]}"; do
    for i in "${!COUNTRIES[@]}"; do
        country="${COUNTRIES[$i]}"
        
        python evaluate.py --model "$model_key" \
                                --language "English" \
                                --country "$country" \
                                --id_col ID \
                                --question_col Translation \
                                --response_col response \
                                --response_dir "../qwen3-14b_i5" \
                                --annotation_dir "../data/annotations" \
                                --annotation_filename "${country}_data.json" \
                                --annotations_key "annotations"
        
        # if [ "$language" != "English" ]; then
        #     python evaluate.py --model "$model_key" \
        #                         --language "English" \
        #                         --country "$country" \
        #                         --prompt_no "$prompt_no" \
        #                         --id_col ID \
        #                         --question_col Translation \
        #                         --response_col response \
        #                         --response_dir "../saq_only_reasoning_results" \
        #                         --annotation_dir "../data/annotations" \
        #                         --annotation_filename "${country}_data.json"  \
        #                         --annotations_key "annotations"
        # fi
    done
done




