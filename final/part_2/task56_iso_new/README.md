# task56_iso_new

Independent lane for part2 task5/task6 only.

This subdirectory covers the later assignment steps only. For the full Part 2 overview, see `../README.md`.

- No edits to task1-4 code.
- Independent app/volume/output paths.
- Simple agent flow: planner output -> tool action -> reward -> GRPO update.
- Step5: Serper/Wiki search reward.
- Step6: rough SQLite executor reward.

## Run (WSL)

```bash
cd /mnt/d/GitHub/info7375_yubo-wang_002458048/final/part_2
source .venv_modal/bin/activate
export SERPER_API_KEY="your_serper_key"   # optional but recommended for step5
```

## Smoke

```bash
python -m modal run task56_iso_new/modal_agentflow_dev.py::smoke
```

## Dev

```bash
python -m modal run task56_iso_new/modal_agentflow_dev.py::train_step5_dev
python -m modal run task56_iso_new/modal_agentflow_dev.py::train_step6_dev
python -m modal run task56_iso_new/modal_agentflow_dev.py::eval_step5_bench_dev
python -m modal run task56_iso_new/modal_agentflow_dev.py::eval_step6_bird_dev
```

## Full (optional)

```bash
python -m modal run -d task56_iso_new/modal_agentflow_dev.py::train_step5_full
python -m modal run -d task56_iso_new/modal_agentflow_dev.py::train_step6_full
```

Outputs:

- `/runs/task56_new/step5_dev/summary.json`
- `/runs/task56_new/step6_dev/summary.json`
- `/runs/task56_new/step5_full/summary.json`
- `/runs/task56_new/step6_full/summary.json`
- `/runs/task56_new/bench_step5_dev/final_scores.json`
- `/runs/task56_new/bench_step5_dev/compare_step3.json`
- `/runs/task56_new/bench_step6_dev/final_scores.json`
- `/runs/task56_new/bench_step6_dev/compare_step3.json`
- `/runs/task56_new/logs/cmd_*.log`

These `/runs/...` paths are Modal volume paths rather than normal local folders.

Notes:

- Command retry is enabled (`--retry 2`).
- If training crashes, error summary is saved to `summary_error.json`.
- If checkpoint exists, training auto-resumes from latest checkpoint.
- Fill `step3_baselines.json` before running compare if you want real deltas.
- This lane is separate from the earlier Step 1/2/3 AgentFlow path, so the comparison files are only approximate references.

