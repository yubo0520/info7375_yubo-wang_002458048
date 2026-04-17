# Task 4: BIRD Text-to-SQL benchmark on Qwen3.5 models via local Ollama
# Usage: bash run_task4_local.sh [0.8b] [2b] [4b] [9b]
# Default: runs 0.8b, 2b, 4b, 9b in sequence

set -eo pipefail

export OLLAMA_HOST="http://172.19.208.1:11434"
export OPENAI_API_KEY="fake-key"
export SERPER_API_KEY="6e7f9ba58720edf774a19b307083bdad6e278961"
export BIRD_DB_DIR="${BIRD_DB_DIR:-/tmp/bird_dev/dev_data/dev_databases/dev_databases}"

if [ ! -d "$BIRD_DB_DIR" ]; then
    echo "Error: BIRD_DB_DIR not found: $BIRD_DB_DIR"
    echo "Run: cp -r /mnt/d/.../dev /tmp/bird_dev first"
    exit 1
fi

if [ $# -eq 0 ]; then
    SIZES=(0.8b 2b 4b 9b)
else
    SIZES=("$@")
fi

N=30
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/AgentFlow/test"
DATA_FILE="bird/data/data.json"
TOOLS="Base_Generator_Tool,SQL_Executor_Tool"
MODEL_ENGINE="trainable,trainable,trainable,trainable"

if [ ! -f "${TEST_DIR}/${DATA_FILE}" ]; then
    echo "Error: ${TEST_DIR}/${DATA_FILE} not found."
    echo "Run prepare_bird_data.py first."
    exit 1
fi

cd "$TEST_DIR"

for SIZE in "${SIZES[@]}"; do
    MODEL="qwen3.5:${SIZE}"
    LABEL="qwen3.5-${SIZE}"
    TOOL_ENGINE="ollama-${MODEL},Default"

    echo "###################################################"
    echo "# MODEL: $MODEL  LABEL: $LABEL  N: $N"
    echo "###################################################"

    OUT_DIR="bird/results/$LABEL"
    LOG_DIR="bird/logs/$LABEL"
    mkdir -p "$OUT_DIR" "$LOG_DIR"

    for i in $(seq 0 $((N-1))); do
        OUT_FILE="$OUT_DIR/output_$i.json"
        if [ -f "$OUT_FILE" ]; then
            echo "  [$i] already done, skip"
            continue
        fi
        echo "  [$i/$((N-1))] running..."
        timeout 180 python solve.py \
            --index $i \
            --task "bird" \
            --data_file "$DATA_FILE" \
            --llm_engine_name "ollama-${MODEL}" \
            --model_engine "$MODEL_ENGINE" \
            --enabled_tools "$TOOLS" \
            --tool_engine "$TOOL_ENGINE" \
            --output_types direct \
            --max_steps 3 \
            --max_time 180 \
            --temperature 0.0 \
            --output_json_dir "$OUT_DIR" \
            2>&1 | tee "$LOG_DIR/$i.log" || {
            EXIT=$?
            if [ $EXIT -eq 124 ]; then
                echo "  [$i] timed out, skipping"
                python3 -c "import json; json.dump({'pid':'$i','direct_output':'timeout','step_count':0,'execution_time':180,'query_analysis':'timed_out','memory':{}}, open('$OUT_FILE','w'))"
            fi
        }
    done

    echo "--- Scoring $LABEL ---"
    python calculate_score_bird.py \
        --data_file "$DATA_FILE" \
        --result_dir "$OUT_DIR" \
        --bird_db_dir "$BIRD_DB_DIR" \
        --output_file "bird_scores.json" \
        | tee "$OUT_DIR/finalscore.log"

    echo "### Done: $LABEL ###"
    echo ""
done

echo "All done! Results in AgentFlow/test/bird/results/*/"
