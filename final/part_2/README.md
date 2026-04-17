# Part 2 README

This directory contains my Part 2 reproduction work for the AgentFlow-based assignment pipeline. It includes:

- baseline inference runs for Qwen2.5 and Qwen3.5 models,
- BIRD Text-to-SQL evaluation scripts,
- a separate `task56_iso_new/` lane for the later GRPO + LoRA experiments,
- helper scripts for scoring


## Directory summary

- `AgentFlow/`: upstream project code used as the base system
- `run_task2.sh`: Step 1 baseline with local Qwen2.5-7B
- `run_task3_*.sh`: Step 2 Qwen3.5 scaling runs
- `run_task3_local.sh`: batch entrypoint for local Qwen3.5 sizes
- `run_task4_local.sh`: Step 3 BIRD evaluation for local Ollama models
- `run_task4_local_dashscope.sh`: Step 3 BIRD evaluation for DashScope Qwen3.5-27B
- `run_task4_bird27b_serper.sh`: optional one-off BIRD continuation script with a different tool setup; not part of the standard rerun path
- `run_gaia_baseline.sh`: GAIA baseline runs
- `score.sh`: optional rescoring helper for QA-style benchmarks; not required for the main rerun
- `task56_iso_new/`: separate implementation used for the later training stages
- `figures/`: plotting scripts and exported figures
- `train/`: local training helpers and checkpoints

## Environment

The work was split across two execution modes:

1. Local / WSL inference runs
   - Python 3.11
   - Ollama for local Qwen models
   - `AgentFlow/.venv` for Step 1/2/3 inference
   - activate the environment before using the local scripts, because most of them call plain `python`

2. Modal-based training runs
   - `.venv_modal` for the Modal entry scripts
   - used for the later GRPO + LoRA pipeline in `task56_iso_new/`

`requirements.txt` at this level is for the local Part 2 helper scripts. The full `AgentFlow/` environment is managed separately inside `AgentFlow/.venv`.

## Step mapping

### Step 1: Qwen2.5-7B baseline

Runs the baseline QA-style benchmarks with local Ollama:

```bash
export SERPER_API_KEY=xxx
bash run_task2.sh
```

Outputs are written under:

```text
AgentFlow/test/<benchmark>/results/qwen2.5-7b/
```

GAIA is handled separately by:

```bash
export DASHSCOPE_API_KEY=your_dashscope_key
bash run_gaia_baseline.sh
```

That script requires `DASHSCOPE_API_KEY` because it also includes the DashScope 27B run at the end.

`score.sh` is only an optional helper if I want to rescore existing QA outputs with a compatible scoring endpoint. It is not required for the main pipeline described here.

### Step 2: Qwen3.5 scaling

For local models `0.8b`, `2b`, `4b`, and `9b`, the single-model scripts now share one local implementation. These entry points still work:

```bash
bash run_task3_qwen3.5-0.8b.sh
bash run_task3_qwen3.5-2b.sh
bash run_task3_qwen3.5-4b.sh
bash run_task3_qwen3.5-9b.sh
```

Or the local batch entry script:

```bash
bash run_task3_local.sh 0.8b 2b 4b 9b
```

If you run `run_task3_local.sh` with no arguments, it defaults to `0.8b 2b 4b`. Pass `9b` explicitly if you want that model as well.

For the 27B model, use the separate DashScope path:

```bash
export DASHSCOPE_API_KEY=your_dashscope_key
export SERPER_API_KEY=your_serper_key
bash run_task3_qwen3.5-27b.sh
```

This script is intended for Linux / WSL shell use.

For the four local QA benchmarks, the `0.8b` output label is:

```text
qwen3.5-0.8b
```

For the GAIA baseline script, the `0.8b` label is the legacy form:

```text
qwen3.5-qwen3.5-0.8b
```

That double prefix is only kept for the existing GAIA results layout.

### Step 3: BIRD Text-to-SQL

Prepare the BIRD dev split first:

```bash
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip
python prepare_bird_data.py \
  --bird_dev_json dev/dev_data/dev.json \
  --bird_db_dir   dev/dev_data/dev_databases \
  --output_dir    AgentFlow/test/bird/data \
  --n 30
```

Then run the evaluations:

```bash
export BIRD_DB_DIR=$(pwd)/dev/dev_data/dev_databases
bash run_task4_local_dashscope.sh
bash run_task4_local.sh 0.8b 2b 4b 9b
```

The order does not matter. In practice, it is safer to set `BIRD_DB_DIR` explicitly instead of relying on per-script defaults.

Outputs go to:

```text
AgentFlow/test/bird/results/<label>/
```

Manual rescoring example:

```bash
cd AgentFlow/test
python calculate_score_bird.py \
  --data_file bird/data/data.json \
  --result_dir bird/results/<label> \
  --bird_db_dir "$BIRD_DB_DIR" \
  --output_file bird_scores.json
```

## Steps 5 and 6 lane

The later GRPO + LoRA experiments are implemented in `task56_iso_new/`.

This is intentionally separated from the earlier full AgentFlow path. It uses:

- a simplified agent loop,
- a separate training/evaluation entrypoint,
- separate Modal app and volume paths,
- separate result summaries.

Main commands:

```bash
cd final/part_2
source .venv_modal/bin/activate
export SERPER_API_KEY=xxx

python -m modal run task56_iso_new/modal_agentflow_dev.py::smoke
python -m modal run task56_iso_new/modal_agentflow_dev.py::train_step5_dev
python -m modal run task56_iso_new/modal_agentflow_dev.py::train_step6_dev
python -m modal run -d task56_iso_new/modal_agentflow_dev.py::train_step5_full
python -m modal run -d task56_iso_new/modal_agentflow_dev.py::train_step6_full
python -m modal run task56_iso_new/modal_agentflow_dev.py::eval_step5_bench_dev
python -m modal run task56_iso_new/modal_agentflow_dev.py::eval_step6_bird_dev
```

## Notes on comparison

The `task56_iso_new/` pipeline is not a drop-in replacement for the earlier Step 1/2/3 AgentFlow path. It is a separate implementation used for the later training stages, so result deltas should be interpreted carefully.

The baseline file used by `compare_step3.py` is:

```text
task56_iso_new/step3_baselines.json
```

