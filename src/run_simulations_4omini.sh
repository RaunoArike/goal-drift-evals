SOURCE="env"
TARGET="profit"
NUM_STEPS=10
MODEL="gpt-4o-mini"

# Array of number of steps to run sequentially
STEPS_ARRAY=(2 4 8 16 32)

# Run simulations with increasing steps
for steps in "${STEPS_ARRAY[@]}"; do
    echo "Running simulation with ${steps} steps..."
    python -m src.run_simulation \
        --source "${SOURCE}" \
        --target "${TARGET}" \
        --num_steps "${NUM_STEPS}" \
        --parallel \
        --num_instrumental_steps "${steps}" \
        --model "${MODEL}" \
        --run_range "1" "5" \
        --checkpoint_dir "../results/base_checkpoints/checkpoints_4omini_new"
    
    # Check if the previous command was successful
    if [ $? -ne 0 ]; then
        echo "Error occurred during simulation with ${steps} steps"
        exit 1
    fi
done

echo "Done"