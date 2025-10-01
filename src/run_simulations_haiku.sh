SOURCE="env"
TARGET="profit"
NUM_STEPS=10

# Array of number of steps to run sequentially
STEPS_ARRAY=(2)

# Run simulations with increasing steps
for steps in "${STEPS_ARRAY[@]}"; do
    echo "Running simulation with ${steps} steps..."
    python -m src.run_simulation \
        --source "${SOURCE}" \
        --target "${TARGET}" \
        --num_steps "${NUM_STEPS}" \
        --parallel \
        --num_instrumental_steps "${steps}" \
        --model "claude-3-5-haiku-latest" \
        --run_range "7" "10" \
        --branch_from "1" "$((steps + 1))" \
        --checkpoint_dir "checkpoints_haiku" \
        --distractions \
        --ood
    
    # Check if the previous command was successful
    if [ $? -ne 0 ]; then
        echo "Error occurred during simulation with ${steps} steps"
        exit 1
    fi

    # echo "Running simulation with ${steps} steps..."
    # python -m src.run_simulation \
    #     --source "${SOURCE}" \
    #     --target "${TARGET}" \
    #     --num_steps "${NUM_STEPS}" \
    #     --parallel \
    #     --num_instrumental_steps "${steps}" \
    #     --model "claude-3-5-haiku-latest" \
    #     --run_range "14" "16" \
    #     --branch_from "1" "$((steps + 1))" \
    #     --checkpoint_dir "checkpoints_haiku" \
    #     --distractions \
    #     --ood
    
    # # Check if the previous command was successful
    # if [ $? -ne 0 ]; then
    #     echo "Error occurred during simulation with ${steps} steps"
    #     exit 1
    # fi

    # echo "Running simulation with ${steps} steps..."
    # python -m src.run_simulation \
    #     --source "${SOURCE}" \
    #     --target "${TARGET}" \
    #     --num_steps "${NUM_STEPS}" \
    #     --parallel \
    #     --num_instrumental_steps "${steps}" \
    #     --model "claude-3-5-haiku-latest" \
    #     --run_range "17" "20" \
    #     --branch_from "1" "$((steps + 1))" \
    #     --checkpoint_dir "checkpoints_haiku" \
    #     --distractions \
    #     --ood
    
    # # Check if the previous command was successful
    # if [ $? -ne 0 ]; then
    #     echo "Error occurred during simulation with ${steps} steps"
    #     exit 1
    # fi
done

echo "Done"