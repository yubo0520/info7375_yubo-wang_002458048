# Part 1 Baseline Replication: UoT (Uncertainty of Thoughts)

Paper: [Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models](https://arxiv.org/abs/2402.03271)

Original repo: https://github.com/zhiyuanhubj/UoT

This is a replication of the DP and UoT baselines from the paper, running on local LLMs via Ollama instead of GPT-4/3.5.

## Setup

1. Install Ollama and pull the models:
```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Init the database (dataset `DX.json` is already in `data/`):
```bash
python setup_db.py
```

## How to run

Run DP baseline:
```bash
python run_baseline.py --method dp --dataset DX --model llama3.1:8b
```

Run UoT baseline:
```bash
python run_baseline.py --method uot --dataset DX --model llama3.1:8b
```

Use `--limit 5` to do a quick test with only 5 samples. Drop `--model` to run all models.

Compare results:
```bash
python compare_results.py --dataset DX
python compare_results.py --dataset DX --export results/baseline_DX_all.csv
```

## What's DP and UoT

**DP (Direct Prompting)**: The LLM directly asks yes/no questions in a multi-turn conversation. Basically just prompt the model with the disease list and let it figure out what to ask. Simple but not very effective.

**UoT**: Uses a tree-search approach. Each turn it generates m candidate questions, splits the possibility space into YES/NO groups, calculates information gain to pick the best question. The first 60% of turns are for information gathering (tree search), the rest for directly guessing.

The key formulas are in `tree.py` — information gain (Eq.3), accumulated reward (Eq.4), and expected reward (Eq.5-6) from the paper.

## Results on DX dataset (closed-set)

Note: I only did the **closed-set** scenario (DPCS in the paper), where the questioner knows the full list of possible diseases.

My DP = the paper's DPCS (Direct Prompting, Closed Set).

| Method | Model | SR (%) | MSC | QGC |
|--------|-------|--------|-----|-----|
| DP | llama3.1:8b | 74.04 | 4.99 | 0 |
| DP | qwen2.5:7b | 16.35 | 4.12 | 0 |
| UoT | llama3.1:8b | 80.77 | 3.00 | 8.0 |
| UoT | qwen2.5:7b | 80.77 | 3.24 | 4.0 |

- SR = success rate, MSC = avg turns for successful cases, QGC = question generation calls (UoT only)
- 104 samples total, max 6 turns per conversation

### Comparison with paper (Table 1, DX column)

| Method | Model | SR (%) | MSC |
|--------|-------|--------|-----|
| DPCS (paper) | GPT-4 | 91.3 | 3.0 |
| DPCS (paper) | GPT-3.5 | 49.5 | 2.7 |
| UoT (paper) | GPT-4 | **97.0** | 2.0 |
| UoT (paper) | GPT-3.5 | 92.1 | 2.1 |
| DP (ours) | llama3.1:8b | 74.04 | 4.99 |
| DP (ours) | qwen2.5:7b | 16.35 | 4.12 |
| UoT (ours) | llama3.1:8b | 80.77 | 3.00 |
| UoT (ours) | qwen2.5:7b | 80.77 | 3.24 |

A few things stand out:

1. **UoT still helps a lot on small models.** qwen2.5 goes from 16% to 81% with UoT — the tree search basically saves it. The paper saw the same thing: GPT-3.5 went from 49.5% (DPCS) to 92.1% (UoT).

2. **Our SR is lower than the paper's.** GPT-4 UoT gets 97% on DX, we get ~81%. Makes sense — we're using 8b models vs GPT-4, and we do self-play (same model as both doctor and patient) while the paper uses GPT-4 as the answerer. A smarter answerer gives more consistent yes/no responses which makes the whole thing work better.

3. **MSC is higher for us.** Paper gets MSC around 2.0-2.1 for UoT, we get 3.0-3.2. Smaller models need more turns to narrow things down, probably because the generated questions don't split the possibility space as cleanly.

## File structure


- `db.py` — SQLite read/write
- `setup_db.py` — load JSON data into database
- `llm.py` — Ollama API wrapper
- `uot.py` — tree expansion, question generation, LLM response parsing
- `tree.py` — tree node classes and reward formulas
- `conversation.py` — DP and UoT conversation loops
- `run_baseline.py` — main entry, runs experiments
- `compare_results.py` — calculate and export metrics

## References

Hu et al., "Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models", NeurIPS 2024. https://arxiv.org/abs/2402.03271
