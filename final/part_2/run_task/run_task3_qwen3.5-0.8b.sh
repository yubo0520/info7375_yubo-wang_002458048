# Task 3 local shared script for Qwen3.5 0.8b/2b/4b/9b
# Default is 0.8b; other local scripts call into this one

set -eo pipefail

export OLLAMA_HOST="http://172.19.208.1:11434"
export VLLM_BASE_URL="http://172.19.208.1:11434/v1"
export OPENAI_API_KEY="fake-key"
export SERPER_API_KEY="6e7f9ba58720edf774a19b307083bdad6e278961"

MODEL=${1:-"qwen3.5:0.8b"}   # pass model as arg, default 0.8b
# Keep label format consistent with other scripts: qwen3.5-0.8b
SIZE="${MODEL#qwen3.5:}"
LABEL="qwen3.5-${SIZE}"
N=30
TEST_DIR="AgentFlow/test"
BENCHMARKS=("bamboogle" "2wiki" "hotpotqa" "musique")
TOOLS="Base_Generator_Tool,Serper_Search_Tool"
TOOL_ENGINE="ollama-${MODEL},Default"
MODEL_ENGINE="trainable,trainable,trainable,trainable"

cd "$TEST_DIR"

for TASK in "${BENCHMARKS[@]}"; do
    DATA_FILE="$TASK/data/data.json"
    OUT_DIR="$TASK/results/$LABEL"
    LOG_DIR="$TASK/logs/$LABEL"
    mkdir -p "$OUT_DIR" "$LOG_DIR"

    echo "========================================"
    echo "TASK: $TASK  MODEL: $MODEL  N: $N"
    echo "========================================"

    for i in $(seq 0 $((N-1))); do
        OUT_FILE="$OUT_DIR/output_$i.json"
        if [ -f "$OUT_FILE" ]; then
            echo "  [$i] already done, skip"
            continue
        fi
        echo "  [$i/$((N-1))] running..."
        timeout 120 python solve.py \
            --index $i \
            --task "$TASK" \
            --data_file "$DATA_FILE" \
            --llm_engine_name "ollama-${MODEL}" \
            --model_engine "$MODEL_ENGINE" \
            --enabled_tools "$TOOLS" \
            --tool_engine "$TOOL_ENGINE" \
            --output_types direct \
            --max_steps 3 \
            --max_time 120 \
            --temperature 0.0 \
            --output_json_dir "$OUT_DIR" \
            2>&1 | tee "$LOG_DIR/$i.log" || {
            EXIT=$?
            if [ $EXIT -eq 124 ]; then
                echo "  [$i] timed out, skipping"
                python3 -c "import json; json.dump({'pid':'$i','direct_output':'timeout','step_count':0,'execution_time':120,'query_analysis':'timed_out','memory':{}}, open('$OUT_FILE','w'))"
            fi
        }
    done

    echo "--- done: $TASK/$LABEL ---"
    echo ""
done

echo "========================================"
echo "All done! Results in AgentFlow/test/*/results/$LABEL/"
echo "========================================"
