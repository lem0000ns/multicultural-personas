#!/bin/bash
set -e

cd /home/tengxiao/pj/persona/multicultural-personas

source .venv/bin/activate

cd culturalbench

python iterate.py \
  --mode eng \
  --difficulty Hard \
  --model zai-org/GLM-4-9B-0414 \
  --num_iterations 5 \
  --temperature 0.6 \
  --max_concurrent 16
