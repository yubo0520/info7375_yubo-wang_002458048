# Assignment 6 -- Polychromic Objectives for RL

Replication of results from [Polychromic Objectives for Reinforcement Learning](https://arxiv.org/abs/2509.25424) (Hamid et al., 2026) on the BabyAI GoTo environment (Table 1).

## Setup

```
pip install -r requirements.txt
python train.py --algo all
```

To re-pretrain from scratch (optional, existing pretrained.pt works):
```
rm pretrained.pt demos_goto.pkl
python pretrain.py
```

## Algorithms

1. **REINFORCE with baseline** (Williams 1992) -- policy gradient with MC returns and learned value baseline.
2. **PPO** (Schulman et al. 2017) -- proximal policy optimization with clipped surrogate, GAE, and per-state KL penalty.
3. **Poly-PPO** (this paper) -- PPO adapted for the polychromic set-RL objective via vine sampling and modified advantage.

## Environment

BabyAI GoTo: multi-room 22x22 grid, agent navigates to a target object specified by a natural language mission (e.g. "go to the red ball"). 50 fixed configurations for training and evaluation. Reward is 1 - 0.5*(t/H) on success, 0 on failure (H=100).

## Paper results (Table 1, Goto)

| Method | Avg Reward | Success (%) |
|--------|-----------|-------------|
| Pretrained | 0.246 | 34.2 |
| REINFORCE | 0.533 | 73.0 |
| PPO | 0.406 | 46.2 |
| Poly-PPO | 0.575 | 80.2 |

Results are averaged over 100 rollouts across 50 configurations and 3 random seeds.

## Our results

Run `python train.py --algo all` to reproduce. Results are written to `results.txt`.


