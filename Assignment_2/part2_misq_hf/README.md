# Part 2 Replication: MISQ-HF

Paper: [Feedback-Aware Monte Carlo Tree Search for Efficient Information Seeking in Goal-Oriented Conversations](https://arxiv.org/abs/2501.15056)

Original repo: [https://github.com/harshita-chopra/misq-hf](https://github.com/harshita-chopra/misq-hf)

This part reproduces DP / UoT / MISQ / MISQ-HF on local LLMs (Ollama) with the DX dataset, with MISQ-HF as the main focus.

## Setup

1. Install Ollama and pull models:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Init the database (dataset `DX.json` is already in `data/`):

```bash
python setup_db.py
```

## How to run

Run each method:

```bash
python run_experiment.py --method dp --dataset DX --model llama3.1:8b
python run_experiment.py --method uot --dataset DX --model llama3.1:8b
python run_experiment.py --method misq --dataset DX --model llama3.1:8b
python run_experiment.py --method misq-hf --dataset DX --model llama3.1:8b
```

Use `--limit 5` for quick testing.  
MISQ/MISQ-HF parameters can be adjusted with `--n_iter`, `--mcts_c`, `--beta`, `--gamma`, `--tau`.

Compare results:

```bash
python compare_results.py --dataset DX
```

## What's DP / UoT / MISQ / MISQ-HF

- **DP (Direct Prompting)**: the baseline that asks yes/no questions directly.
- **UoT**: tree-based information-seeking with expected reward.
- **MISQ**: MCTS-based question selection without feedback.
- **MISQ-HF**: MISQ with cluster-based hierarchical feedback bonus.

Detailed analysis and discussion are provided in the report.

## File structure

- `db.py` — SQLite read/write
- `setup_db.py` — load JSON data into database
- `llm.py` — Ollama API wrapper
- `uot.py` — tree expansion, question generation, LLM response parsing
- `tree.py` — tree node classes and reward formulas
- `conversation.py` — DP, UoT, MISQ, and MISQ-HF conversation loops
- `mcts.py` — MCTS search and feedback propagation for MISQ/MISQ-HF
- `embedding.py` — embedding-based clustering for hierarchical feedback
- `run_experiment.py` — main entry, runs experiments
- `compare_results.py` — calculate and export metrics

## References

Chopra & Shah, "Feedback-Aware Monte Carlo Tree Search for Efficient Information Seeking in Goal-Oriented Conversations", 2025. [https://arxiv.org/abs/2501.15056](https://arxiv.org/abs/2501.15056)

Hu et al., "Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models", NeurIPS 2024. [https://arxiv.org/abs/2402.03271](https://arxiv.org/abs/2402.03271)

## Record

[Watch the recording here](https://drive.google.com/file/d/1LtDqivEklshqJpbtCZ42xWF458xPDDfL/view?usp=sharing)

