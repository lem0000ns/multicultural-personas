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

# Define model keys
# MODEL_KEYS=(
#     "llama-3-8b-instruct"
# )
MODEL_KEYS=(
    "qwen3-32b"
)

for model_key in "${MODEL_KEYS[@]}"; do
    python multiple_choice_evaluation.py --model "$model_key" \
        --model_cache_dir '.cache' \
        --mc_dir './mc_data' \
        --questions_file 'mc_questions_file-2.csv' \
        --response_file "${model_key}-mc_test.csv" \
        --temperature 0.6 \
        --top_p 1 \
        --gpt_azure 'True' \
        --num_iterations 1 \
        --sample_size 100 \
        --random_seed 42 \
        --use_persona "False" \
        --use_reasoning "False"
done
