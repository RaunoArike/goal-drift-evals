SOURCE="env"
TARGET="profit"
NUM_STEPS=10

# Array of number of steps to run sequentially
STEPS_ARRAY=(2 4 8 16 32)

# Run simulations with increasing steps
for steps in "${STEPS_ARRAY[@]}"; do
    echo "Running simulation with ${steps} steps..."
    python -m src.run_simulation \
        --source "${SOURCE}" \
        --target "${TARGET}" \
        --num_steps "${NUM_STEPS}" \
        --runs "1" \
        --num_instrumental_steps "${steps}" \
        --model "gpt-4o-mini" \
        --checkpoint_dir "../results/base_checkpoints/checkpoints_4omini_new" \
        --distractions
done

echo "Done"