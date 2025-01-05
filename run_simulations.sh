#!/bin/bash

# Base parameters
SOURCE="env"
TARGET="profit"

echo "Running simulation..."
python run_simulation.py \
    --source "${SOURCE}" \
    --target "${TARGET}" \
    --num_steps "10" \
    --parallel \
    --model "claude-3-5-sonnet-latest" \
    --run_range "1" "3" \
    --checkpoint_dir "sonnet_checkpoints_elicit" \
    --distractions \
    --resume
    
if [ $? -ne 0 ]; then
    echo "Error occurred during simulation"
fi

echo "All simulations completed successfully!"
