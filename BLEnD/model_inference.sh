#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

export HF_TOKEN="" 
export COHERE_API_KEY=""
export OPENAI_API_KEY=""
export OPENAI_ORG_ID=""
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_API_VER=""
export AZURE_OPENAI_API_ENDPT=""
export CLAUDE_API_KEY=""
export GOOGLE_API_KEY=""
export GOOGLE_APPLICATION_CREDENTIALS=""
export GOOGLE_PROJECT_NAME=""

# # Define model keys
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
#     "aya-23"
#     "SeaLLM-7B-v2.5"
#     "Merak-7B-v4"
#     "jais-13b-chat"
# )

MODEL_KEYS=("gpt-3.5-turbo-1106")

COUNTRIES=("UK" "US" "South_Korea" "Algeria" "China" "Indonesia" "Spain" "Iran" "Mexico" "Assam" "Greece" "Ethiopia" "Northern_Nigeria" "Azerbaijan" "North_Korea" "West_Java")
LANGUAGES=("English" "English" "Korean" "Arabic" "Chinese" "Indonesian" "Spanish" "Persian" "Spanish" "Assamese" "Greek" "Amharic" "Hausa" "Azerbaijani" "Korean" "Sundanese")

# Prompt numbers
PROMPT_NUMBERS=("inst-4" "pers-3")

# Iterate over models, countries, languages, and prompts
for model_key in "${MODEL_KEYS[@]}"; do
    for i in "${!COUNTRIES[@]}"; do
        country="${COUNTRIES[$i]}"
        language="${LANGUAGES[$i]}"
        for prompt_no in "${PROMPT_NUMBERS[@]}"; do
            python model_inference.py --model "$model_key" \
                                --language "$language" \
                                --country "$country" \
                                --question_dir "./data/questions" \
                                --question_file "${country}_questions.csv" \
                                --question_col Translation \
                                --prompt_dir "./data/prompts" \
                                --prompt_file "${country}_prompts.csv" \
                                --prompt_no "$prompt_no" \
                                --id_col ID \
                                --output_dir "./model_inference_results" \
                                --output_file "${model_key}-${country}_${language}_${prompt_no}_result.csv" \
                                --model_cache_dir ".cache" \
                                --gpt_azure "True" \
                                --temperature 0 \
                                --top_p 1 
            if [ "$language" != "English" ]; then
                python model_inference.py --model "$model_key" \
                                    --language "$language" \
                                    --country "$country" \
                                    --question_dir "./data/questions" \
                                    --question_file "${country}_questions.csv" \
                                    --question_col Question \
                                    --prompt_dir "./data/prompts" \
                                    --prompt_file "${country}_prompts.csv" \
                                    --prompt_no "$prompt_no" \
                                    --id_col ID \
                                    --output_dir "./model_inference_results" \
                                    --output_file "${model_key}-${country}_English_${prompt_no}_result.csv" \
                                    --model_cache_dir ".cache" \
                                    --gpt_azure "True" \
                                    --temperature 0 \
                                    --top_p 1 
            fi
        done
    done
done





