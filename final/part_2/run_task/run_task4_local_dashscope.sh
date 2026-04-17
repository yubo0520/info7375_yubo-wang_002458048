set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 避免 AgentFlow 内部的 types.py 遮蔽 Python stdlib
export PYTHONPATH="${SCRIPT_DIR}/AgentFlow"

# ---------- DashScope API 配置 ----------
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "Error: DASHSCOPE_API_KEY 未设置。请运行: export DASHSCOPE_API_KEY=\"sk-xxxx\""
    exit 1
fi

# qwen3.5-27b 只在 DashScope OpenAI 兼容接口可用，走 litellm-openai/ 前缀
export OPENAI_API_KEY="${DASHSCOPE_API_KEY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
# LiteLLM 同时读 OPENAI_BASE_URL（兼容两种变量名）
export OPENAI_BASE_URL="${OPENAI_API_BASE}"
echo "DashScope API Key        : ${DASHSCOPE_API_KEY:0:8}..."
echo "OpenAI-compat base URL   : ${OPENAI_API_BASE}"

# ---------- BIRD 数据库路径 ----------
if [ -z "${BIRD_DB_DIR:-}" ]; then
    LOCAL_DB="${SCRIPT_DIR}/dev/dev_data/dev_databases"
    if [ -d "$LOCAL_DB" ]; then
        export BIRD_DB_DIR="$LOCAL_DB"
    else
        echo "Error: BIRD_DB_DIR 未设置且本地路径不存在: $LOCAL_DB"
        echo "请运行: export BIRD_DB_DIR=/path/to/dev_databases"
        exit 1
    fi
fi

echo "Using BIRD_DB_DIR         : ${BIRD_DB_DIR}"

# ---------- 运行参数 ----------
# litellm- 前缀走 ChatLiteLLM，openai/ 子前缀让 LiteLLM 用 OPENAI_API_BASE
MODEL="${MODEL:-litellm-openai/qwen3.5-27b}"
LABEL="${LABEL:-qwen3.5-27b}"
N="${N:-30}"
TEST_DIR="${SCRIPT_DIR}/AgentFlow/test"
DATA_FILE="bird/data/data.json"
TOOLS="Base_Generator_Tool,SQL_Executor_Tool"
TOOL_ENGINE="${MODEL},Default"
MODEL_ENGINE="trainable,trainable,trainable,trainable"

if [ ! -f "${TEST_DIR}/${DATA_FILE}" ]; then
    echo "Error: ${TEST_DIR}/${DATA_FILE} 不存在。"
    echo "请先运行 prepare_bird_data.py 生成数据文件。"
    exit 1
fi

cd "$TEST_DIR"

echo "###################################################"
echo "# MODEL: $MODEL  LABEL: $LABEL  N: $N"
echo "###################################################"

OUT_DIR="bird/results/$LABEL"
LOG_DIR="bird/logs/$LABEL"
mkdir -p "$OUT_DIR" "$LOG_DIR"

for i in $(seq 0 $((N-1))); do
    OUT_FILE="${OUT_DIR}/output_${i}.json"
    if [ -f "$OUT_FILE" ]; then
        echo "  [$i] already done, skip"
        continue
    fi
    echo "  [$i/$((N-1))] running..."

    EXIT=0
    timeout 180 python solve.py \
        --index "$i" \
        --task "bird" \
        --data_file "$DATA_FILE" \
        --llm_engine_name "$MODEL" \
        --model_engine "$MODEL_ENGINE" \
        --enabled_tools "$TOOLS" \
        --tool_engine "$TOOL_ENGINE" \
        --output_types direct \
        --max_steps 3 \
        --max_time 180 \
        --temperature 0.0 \
        --output_json_dir "$OUT_DIR" \
        2>&1 | tee "${LOG_DIR}/${i}.log" || EXIT=$?

    if [ "$EXIT" -eq 124 ]; then
        echo "  [$i] timed out, writing placeholder and skipping"
        python3 -c "import json; json.dump({'pid':'${i}','direct_output':'timeout','step_count':0,'execution_time':180,'query_analysis':'timed_out','memory':{}}, open('${OUT_FILE}','w'))"
    elif [ "$EXIT" -ne 0 ]; then
        echo "  [$i] failed with exit code $EXIT, skipping"
    fi
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
echo "All done! Results in AgentFlow/test/bird/results/${LABEL}/"
