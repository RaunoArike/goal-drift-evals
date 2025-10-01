SOURCE="env"
TARGET="profit"
NUM_STEPS=10

echo "Running simulation..."
python -m src.run_simulation \
    --source "${SOURCE}" \
    --target "${TARGET}" \
    --num_steps "${NUM_STEPS}" \
    --parallel \
    --num_instrumental_steps "32" \
    --model "claude-3-5-sonnet-latest" \
    --runs "16" \
    --branch_from "1" "33" \
    --checkpoint_dir "checkpoints_sonnet"

echo "Done"
