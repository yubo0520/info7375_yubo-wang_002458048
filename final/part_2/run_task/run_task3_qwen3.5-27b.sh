# Task 3 (Lightning only): Run AgentFlow with Qwen3.5-27B via DashScope OpenAI-compatible API
# Usage on Lightning:
#   export DASHSCOPE_API_KEY="your_dashscope_key"
#   export SERPER_API_KEY="6e7f9ba58720edf774a19b307083bdad6e278961"   # optional but recommended
#   bash run_task3_qwen3.5-27b.sh

set -euo pipefail

if [ "$(uname -s)" != "Linux" ]; then
    echo "Error: this script is for Lightning/Linux only."
    exit 1
fi

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "Error: DASHSCOPE_API_KEY is not set."
    echo "Please run: export DASHSCOPE_API_KEY=\"your_dashscope_key\""
    exit 1
fi

if [ -z "${SERPER_API_KEY:-}" ]; then
    echo "Warning: SERPER_API_KEY is not set. Serper tool may fail."
fi

# Use DashScope OpenAI-compatible endpoint (curl test path that already works for you).
# You can override manually:
#   export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
if [ -z "${OPENAI_BASE_URL:-}" ]; then
    if [ "${DASHSCOPE_REGION:-cn}" = "intl" ]; then
        export OPENAI_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    else
        export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    fi
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-$DASHSCOPE_API_KEY}"
echo "Using OpenAI-compatible base URL: ${OPENAI_BASE_URL}"

# We intentionally use plain model id to route through ChatOpenAI + compatible-mode endpoint.
MODEL="${MODEL:-qwen3.5-27b}"
LABEL="${LABEL:-qwen3.5-27b}"
N="${N:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${TEST_DIR:-$SCRIPT_DIR/AgentFlow/test}"
BENCHMARKS=("bamboogle" "2wiki" "hotpotqa" "musique")
TOOLS="Base_Generator_Tool,Serper_Search_Tool"
TOOL_ENGINE="${TOOL_ENGINE:-$MODEL,Default}"
MODEL_ENGINE="${MODEL_ENGINE:-trainable,trainable,trainable,trainable}"

if [ ! -d "$TEST_DIR" ]; then
    echo "Error: TEST_DIR not found: $TEST_DIR"
    echo "Hint: export TEST_DIR=\"/path/to/AgentFlow/test\""
    exit 1
fi

cd "$TEST_DIR"

for TASK in "${BENCHMARKS[@]}"; do
    DATA_FILE="$TASK/data/data.json"
    OUT_DIR="$TASK/results/$LABEL"
    LOG_DIR="$TASK/logs/$LABEL"
    mkdir -p "$OUT_DIR" "$LOG_DIR"

    echo "TASK: $TASK  MODEL: $MODEL  LABEL: $LABEL  N: $N"

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

    echo "done: $TASK/$LABEL"
done

echo "All done. Results in AgentFlow/test/*/results/$LABEL/"
