SOURCE="env"
TARGET="profit"
NUM_STEPS=10

# Array of number of steps to run sequentially
STEPS_ARRAY=(32 16 8 4 2)

# Run simulations with increasing steps
for steps in "${STEPS_ARRAY[@]}"; do
    echo "Running simulation with ${steps} steps..."
    uv python run_simulation.py \
        --source "${SOURCE}" \
        --target "${TARGET}" \
        --num_steps "${NUM_STEPS}" \
        --parallel \
        --num_instrumental_steps "${steps}" \
        --model "claude-3-5-haiku-latest" \
        --run_range "1" "20"  \
        --branch_from "1" "$((steps + 1))" \
        --checkpoint_dir "checkpoints_haiku_on_sonnet" \
        --distractions \
        --verbose
    
    # Check if the previous command was successful
    if [ $? -ne 0 ]; then
        echo "Error occurred during simulation with ${steps} steps"
    fi
done
