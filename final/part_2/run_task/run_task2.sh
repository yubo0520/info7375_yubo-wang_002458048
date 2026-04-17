# Run the paper benchmarks with local Qwen2.5-7B.

set -e

export VLLM_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="fake-key"
export SERPER_API_KEY="your_serper_api_key"
export OLLAMA_HOST="http://172.19.208.1:11434"

MODEL="ollama-qwen2.5:7b"
LABEL="qwen2.5-7b"
N=30
TEST_DIR="AgentFlow/test"
BENCHMARKS=("bamboogle" "2wiki" "hotpotqa" "musique")
TOOLS="Base_Generator_Tool,Serper_Search_Tool,Wikipedia_Search_Tool"
TOOL_ENGINE="ollama-qwen2.5:7b,Default,ollama-qwen2.5:7b"
MODEL_ENGINE="trainable,trainable,trainable,trainable"

cd "$TEST_DIR"

for TASK in "${BENCHMARKS[@]}"; do
    DATA_FILE="$TASK/data/data.json"
    OUT_DIR="$TASK/results/$LABEL"
    LOG_DIR="$TASK/logs/$LABEL"
    mkdir -p "$OUT_DIR" "$LOG_DIR"

    echo "========================================"
    echo "TASK: $TASK  MODEL: $LABEL  N: $N"
    echo "========================================"

    for i in $(seq 0 $((N-1))); do
        OUT_FILE="$OUT_DIR/output_$i.json"
        if [ -f "$OUT_FILE" ]; then
            echo "  [$i] already done, skip"
            continue
        fi
        echo "  [$i/$((N-1))] running..."
        python solve.py \
            --index $i \
            --task "$TASK" \
            --data_file "$DATA_FILE" \
            --llm_engine_name "$MODEL" \
            --model_engine "$MODEL_ENGINE" \
            --enabled_tools "$TOOLS" \
            --tool_engine "$TOOL_ENGINE" \
            --output_types direct \
            --max_steps 3 \
            --max_time 120 \
            --temperature 0.0 \
            --output_json_dir "$OUT_DIR" \
            2>&1 | tee "$LOG_DIR/$i.log"
    done

    echo ""
    echo "--- scoring $TASK ---"
    python calculate_score_unified.py \
        --task_name "$TASK" \
        --data_file "$DATA_FILE" \
        --result_dir "$OUT_DIR" \
        --response_type "direct_output" \
        --output_file "finalresults_direct_output.json" \
        | tee "$OUT_DIR/finalscore.log"
    echo ""
done


echo "All done! Results in AgentFlow/test/*/results/$LABEL/"

