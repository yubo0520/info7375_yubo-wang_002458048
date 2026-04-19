#!/bin/bash
# Bird 27b continuation — Serper tool setup (matches output_2/3/4)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/AgentFlow"
export OPENAI_API_KEY="${DASHSCOPE_API_KEY:?set DASHSCOPE_API_KEY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_BASE_URL="${OPENAI_API_BASE}"
export SERPER_API_KEY="${SERPER_API_KEY:-6e7f9ba58720edf774a19b307083bdad6e278961}"

MODEL="qwen3.5-27b"
LABEL="qwen3.5-27b"
N="${N:-30}"
TEST_DIR="${SCRIPT_DIR}/AgentFlow/test"
DATA_FILE="bird/data/data.json"
TOOLS="Base_Generator_Tool,Serper_Search_Tool"
TOOL_ENGINE="${MODEL},Default"
MODEL_ENGINE="trainable,trainable,trainable,trainable"

cd "$TEST_DIR"

OUT_DIR="bird/results/$LABEL"
LOG_DIR="bird/logs/$LABEL"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "MODEL=$MODEL  TOOLS=$TOOLS  N=$N"

for i in $(seq 0 $((N-1))); do
    OUT_FILE="$OUT_DIR/output_$i.json"
    if [ -f "$OUT_FILE" ]; then
        echo "  [$i] already done, skip"
        continue
    fi
    echo "  [$i/$((N-1))] running..."

    EXIT=0
    timeout 300 python solve.py \
        --index "$i" \
        --task "bird" \
        --data_file "$DATA_FILE" \
        --llm_engine_name "$MODEL" \
        --model_engine "$MODEL_ENGINE" \
        --enabled_tools "$TOOLS" \
        --tool_engine "$TOOL_ENGINE" \
        --output_types direct \
        --max_steps 3 \
        --max_time 300 \
        --temperature 0.0 \
        --output_json_dir "$OUT_DIR" \
        2>&1 | tee "$LOG_DIR/$i.log" || EXIT=$?

    if [ "$EXIT" -eq 124 ]; then
        echo "  [$i] timed out"
        python3 -c "import json; json.dump({'pid':'${i}','direct_output':'timeout','step_count':0,'execution_time':300,'query_analysis':'timed_out','memory':{}}, open('${OUT_FILE}','w'))"
    elif [ "$EXIT" -ne 0 ]; then
        echo "  [$i] failed exit=$EXIT"
    fi
done

echo "--- Scoring $LABEL ---"
python calculate_score_bird.py \
    --data_file "$DATA_FILE" \
    --result_dir "$OUT_DIR" \
    --bird_db_dir "${BIRD_DB_DIR:-${SCRIPT_DIR}/dev/dev_data/dev_databases}" \
    --output_file "bird_scores.json" | tee "$OUT_DIR/finalscore.log"

echo "### Done ###"
