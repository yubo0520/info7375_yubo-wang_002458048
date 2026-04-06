# Part 1: TinyZero Countdown with LoRA

This repo contains Final Part 1 for reproducing the TinyZero countdown task with LoRA.

Setup used:
- `TRL + PEFT + Transformers`
- model: `Qwen/Qwen2.5-3B-Instruct`
- dataset: `Jiayi-Pan/Countdown-Tasks-3to4`
- TinyZero countdown prompt format (with `<think>` and `<answer>`)

## Changes compared to TinyZero

TinyZero uses `verl` and full RL training setup.  
In this assignment, the setup uses:
- `GRPOTrainer` from TRL
- LoRA adapters (not full fine-tuning)

## How to run

Install dependencies:

```bash
pip install -U peft trl transformers datasets accelerate
```

Run:

```bash
python train.py
```

You can also run in Colab using `colab_run.ipynb`.

## What the script does

`train.py` runs 3 phases:
1. Baseline evaluation on the base model
2. GRPO + LoRA training
3. Evaluate saved checkpoints

## Run setup

- train samples: `1000`
- eval samples: `100`
- LoRA: `r=16`, `alpha=64`, `dropout=0.05`
- `num_generations=4`
- `max_completion_length` was changed from `512` to `256` to make the run finish in Colab time limits.

## Results from this run

- baseline accuracy: `16.0%`
- checkpoint results:
  - step 50: `17.0%`
  - step 100: `12.0%`
  - step 150: `16.0%`
  - step 200: `16.0%`
  - step 250: `13.0%`
  - step 300: `8.0%`
  - step 350: `8.0%`
  - step 400: `13.0%`
  - step 450: `12.0%`
  - step 500: `12.0%`
- best checkpoint: `step 50` with `17.0%`

This run shows a small improvement over baseline, with high variance across checkpoints.

## Notes

- Long runtime is expected for GRPO on 3B model.
- Warnings like unauthenticated HF requests and generation flag notices did not stop training.
- Checkpoints are saved during training, so interrupted runs can be resumed.

## References

- TinyZero repo: https://github.com/Jiayi-Pan/TinyZero
- TinyZero countdown prompt source: https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py
