export OPENAI_API_KEY="phnAmFX35ndnbqMWjeE8y/UNHgX0"
export OPENAI_BASE_URL="https://api.portkey.ai/v1"

LABEL=${1:-"qwen2.5-7b"}
BENCHMARKS=("bamboogle" "2wiki" "hotpotqa" "musique")

cd AgentFlow/test

for TASK in "${BENCHMARKS[@]}"; do
    RESULT_DIR="$TASK/results/$LABEL"
    DATA_FILE="$TASK/data/data.json"

    if [ ! -d "$RESULT_DIR" ]; then
        echo "[skip] $TASK/$LABEL: no results directory"
        continue
    fi

    echo "[score] $TASK / $LABEL"
    python calculate_score_unified.py \
        --task_name "$TASK" \
        --data_file "$DATA_FILE" \
        --result_dir "$RESULT_DIR" \
        --response_type direct_output \
        --output_file finalresults_direct_output.json \
        | tee "$RESULT_DIR/finalscore.log"
    echo "[wait] sleeping 90s."
    sleep 90
done

echo "[done] scoring complete"
