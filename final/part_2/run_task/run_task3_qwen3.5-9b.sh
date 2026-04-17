# Thin wrapper around the shared local Qwen3.5 task3 script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_task3_qwen3.5-0.8b.sh" "qwen3.5:9b"
