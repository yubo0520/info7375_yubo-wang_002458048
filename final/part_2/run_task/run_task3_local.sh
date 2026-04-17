# Task 3: Run AgentFlow on Qwen3.5 models via local Ollama
# Usage: bash run_task3_local.sh [model1] [model2] ...
# Example: bash run_task3_local.sh 0.8b 2b 4b
# Default: runs 0.8b, 2b, 4b in sequence

set -eo pipefail

export OLLAMA_HOST="http://172.19.208.1:11434"
export VLLM_BASE_URL="http://172.19.208.1:11434/v1"
export OPENAI_API_KEY="fake-key"
export SERPER_API_KEY="6e7f9ba58720edf774a19b307083bdad6e278961"

SIZES=("${@:-0.8b 2b 4b}")
if [ $# -eq 0 ]; then
    SIZES=(0.8b 2b 4b)
else
    SIZES=("$@")
fi

N=30
TEST_DIR="AgentFlow/test"
BENCHMARKS=("bamboogle" "2wiki" "hotpotqa" "musique")
TOOLS="Base_Generator_Tool,Serper_Search_Tool"
MODEL_ENGINE="trainable,trainable,trainable,trainable"

cd "$TEST_DIR"

for SIZE in "${SIZES[@]}"; do
    MODEL="qwen3.5:${SIZE}"
    LABEL="qwen3.5-${SIZE}"
    TOOL_ENGINE="ollama-${MODEL},Default"

    echo "###################################################"
    echo "# Starting MODEL: $MODEL  LABEL: $LABEL"
    echo "###################################################"

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

    echo "###################################################"
    echo "# Done MODEL: $MODEL  Results in AgentFlow/test/*/results/$LABEL/"
    echo "###################################################"
    echo ""
done

echo "All done!"
