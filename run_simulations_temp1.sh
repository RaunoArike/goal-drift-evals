#!/bin/bash

# Base parameters
SOURCE="env"
TARGET="profit"

# echo "Running simulation..."
# python run_simulation.py \
#     --source "${SOURCE}" \
#     --target "${TARGET}" \
#     --num_steps "8" \
#     --parallel \
#     --model "gpt-4o-mini" \
#     --run_range "1" "3" \
#     --checkpoint_dir "env_profit_4omini" \
#     --distractions \
#     --use_soft_system_prompt \
#     --resume
    
# if [ $? -ne 0 ]; then
#     echo "Error occurred during simulation"
# fi

echo "Running simulation..."
python run_simulation.py \
    --source "${SOURCE}" \
    --target "${TARGET}" \
    --num_steps "8" \
    --parallel \
    --model "gpt-4o-mini" \
    --run_range "1" "3" \
    --checkpoint_dir "env_profit_4omini_elicit" \
    --distractions \
    --resume
    
if [ $? -ne 0 ]; then
    echo "Error occurred during simulation"
fi

# echo "Running simulation..."
# python run_simulation.py \
#     --source "${SOURCE}" \
#     --target "${TARGET}" \
#     --num_steps "8" \
#     --parallel \
#     --model "claude-3-5-sonnet-latest" \
#     --run_range "1" "3" \
#     --checkpoint_dir "env_profit_sonnet" \
#     --distractions \
#     --use_soft_system_prompt \
#     --resume
    
# if [ $? -ne 0 ]; then
#     echo "Error occurred during simulation"
# fi

echo "Running simulation..."
python run_simulation.py \
    --source "${SOURCE}" \
    --target "${TARGET}" \
    --num_steps "8" \
    --parallel \
    --model "claude-3-5-sonnet-latest" \
    --run_range "1" "3" \
    --checkpoint_dir "env_profit_sonnet_elicit" \
    --distractions \
    --resume
    
if [ $? -ne 0 ]; then
    echo "Error occurred during simulation"
fi

echo "All simulations completed successfully!"