# Run GAIA baseline for Step 1 (qwen2.5-7b) and Step 2 (all 5 Qwen3.5)
# N=30 questions, consistent with other benchmarks
#
# Usage:
#   export DASHSCOPE_API_KEY="sk-xxxx"
#   bash run_gaia_baseline.sh

set -eo pipefail

export OLLAMA_HOST="http://172.19.208.1:11434"
export SERPER_API_KEY="6e7f9ba58720edf774a19b307083bdad6e278961"

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "Error: please set DASHSCOPE_API_KEY first"
    echo "  export DASHSCOPE_API_KEY=\"sk-xxxx\""
    exit 1
fi

N=30
TASK="gaia"
DATA_FILE="gaia/data/data.json"
PYTHON="$(pwd)/AgentFlow/.venv/bin/python"

cd AgentFlow/test

score_model() {
    local LABEL=$1
    local OUT_DIR="gaia/results/$LABEL"
    echo "--- scoring $LABEL ---"
    "$PYTHON" calculate_score_unified.py \
        --task_name "$TASK" \
        --data_file "$DATA_FILE" \
        --result_dir "$OUT_DIR" \
        --response_type "direct_output" \
        --output_file "finalresults_direct_output.json" \
        | tee "$OUT_DIR/finalscore.log"
    echo ""
}

run_ollama() {
    local MODEL=$1   # e.g. qwen2.5:7b
    local LABEL=$2

    export OPENAI_API_KEY="fake-key"
    unset OPENAI_BASE_URL 2>/dev/null || true

    OUT_DIR="gaia/results/$LABEL"
    LOG_DIR="gaia/logs/$LABEL"
    mkdir -p "$OUT_DIR" "$LOG_DIR"

    echo "========================================"
    echo "[Ollama] MODEL: $MODEL  LABEL: $LABEL  N: $N"
    echo "========================================"

    for i in $(seq 0 $((N-1))); do
        OUT_FILE="$OUT_DIR/output_$i.json"
        if [ -f "$OUT_FILE" ]; then echo "  [$i] skip"; continue; fi
        echo "  [$i/$((N-1))]..."
        timeout 120 "$PYTHON" solve.py \
            --index $i \
            --task "$TASK" \
            --data_file "$DATA_FILE" \
            --llm_engine_name "ollama-${MODEL}" \
            --model_engine "trainable,trainable,trainable,trainable" \
            --enabled_tools "Base_Generator_Tool,Serper_Search_Tool" \
            --tool_engine "ollama-${MODEL},Default" \
            --output_types direct \
            --max_steps 3 \
            --max_time 120 \
            --temperature 0.0 \
            --output_json_dir "$OUT_DIR" \
            2>&1 | tee "$LOG_DIR/$i.log" || {
            [ $? -eq 124 ] && python3 -c "import json; json.dump({'pid':'$i','direct_output':'timeout','step_count':0,'execution_time':120}, open('$OUT_DIR/output_$i.json','w'))"
        }
    done
    score_model "$LABEL"
}

run_dashscope() {
    local MODEL=$1   # e.g. qwen3.5-27b (DashScope model id)
    local LABEL=$2

    export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
    export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

    OUT_DIR="gaia/results/$LABEL"
    LOG_DIR="gaia/logs/$LABEL"
    mkdir -p "$OUT_DIR" "$LOG_DIR"

    echo "========================================"
    echo "[DashScope] MODEL: $MODEL  LABEL: $LABEL  N: $N"
    echo "========================================"

    for i in $(seq 0 $((N-1))); do
        OUT_FILE="$OUT_DIR/output_$i.json"
        if [ -f "$OUT_FILE" ]; then echo "  [$i] skip"; continue; fi
        echo "  [$i/$((N-1))]..."
        "$PYTHON" solve.py \
            --index $i \
            --task "$TASK" \
            --data_file "$DATA_FILE" \
            --llm_engine_name "$MODEL" \
            --model_engine "trainable,trainable,trainable,trainable" \
            --enabled_tools "Base_Generator_Tool,Serper_Search_Tool" \
            --tool_engine "$MODEL,Default" \
            --output_types direct \
            --max_steps 3 \
            --max_time 120 \
            --temperature 0.0 \
            --output_json_dir "$OUT_DIR" \
            2>&1 | tee "$LOG_DIR/$i.log"
    done
    score_model "$LABEL"
}

# Step 1: Qwen2.5-7B (Ollama)
run_ollama "qwen2.5:7b" "qwen2.5-7b"

# Step 2: Qwen3.5 scaling
run_ollama    "qwen3.5:0.8b"  "qwen3.5-qwen3.5-0.8b"
run_ollama    "qwen3.5:2b"    "qwen3.5-2b"
run_ollama    "qwen3.5:4b"    "qwen3.5-4b"
run_ollama    "qwen3.5:9b"    "qwen3.5-9b"
run_dashscope "qwen3.5-27b"   "qwen3.5-27b"   # 阿里云 DashScope

echo "========================================"
echo "All done! Results in AgentFlow/test/gaia/results/"
echo "========================================"
