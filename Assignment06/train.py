"""
main entry: pretrain then fine-tune with REINFORCE / PPO / Poly-PPO
replicates Table 1 (Goto) from Hamid et al. 2026

usage:
    python train.py --algo reinforce
    python train.py --algo ppo
    python train.py --algo poly_ppo
    python train.py --algo all        # run all three sequentially
"""

import argparse
import copy
import os
import numpy as np
import torch

from env import GoToEnv
from model import make_policy, build_vocab_from_demos
from eval import eval_policy, print_results

# --- config

CFG = {
    "n_train_configs": 50,
    "n_eval_rollouts": 100,
    "n_iters": 1000,
    "n_seeds": 3,
    "max_steps": 100,
    "lr_trunk": 0,
    "lr_actor": 1e-5,
    "lr_critic": 3e-5,
    "gamma": 1.0,
    "lam": 0.95,
    "eps": 0.2,
    "kl_coef": 0.01,
    "eps_per_iter": 32,
    "critic_warmup_iters": 100,
    "reinforce_iters": 400,
    "ppo_iters": 400,
    "poly_iters": 250,
    "reinforce_eps_per_iter": 32,
    "ppo_eps_per_iter": 32,
    # poly-ppo specific
    "N": 4,
    "n": 4,
    "M": 4,
    "p": 1,
    "W": 5,
}

PRETRAIN_PATH = "pretrained.pt"
RESULTS_PATH  = "results.txt"
DEMO_PATH = "demos_goto.pkl"

TRAIN_CONFIGS = list(range(50))
EVAL_CONFIGS  = list(range(50))


def load_pretrained(n_actions, device):
    build_vocab_from_demos(DEMO_PATH)
    policy = make_policy(n_actions, device)
    if os.path.exists(PRETRAIN_PATH):
        policy.load_state_dict(torch.load(PRETRAIN_PATH, map_location=device))
        print(f"loaded pretrained from {PRETRAIN_PATH}")
    else:
        print("no pretrained checkpoint found -- running from scratch")
        print("hint: run pretrain.py first for better results")
    return policy


def run_seed(algo, pretrained_state, seed, device):
    from env import GoToEnv
    env = GoToEnv()
    env.reset()
    n_actions = env.n_actions

    policy = make_policy(n_actions, device)
    policy.load_state_dict(copy.deepcopy(pretrained_state))
    n_iters = CFG["n_iters"]
    if algo == "reinforce":
        n_iters = CFG["reinforce_iters"]
    elif algo == "ppo":
        n_iters = CFG["ppo_iters"]
    elif algo == "poly_ppo":
        n_iters = CFG["poly_iters"]

    if algo == "reinforce":
        from reinforce import train_reinforce
        policy = train_reinforce(policy, TRAIN_CONFIGS, device,
                                 n_iters=n_iters, lr=CFG["lr_actor"],
                                 lr_trunk=CFG["lr_trunk"], lr_critic=CFG["lr_critic"],
                                 eps_per_iter=CFG["reinforce_eps_per_iter"])

    elif algo == "ppo":
        from ppo import train_ppo
        policy = train_ppo(policy, TRAIN_CONFIGS, device,
                           n_iters=n_iters, lr=CFG["lr_actor"],
                           lr_trunk=CFG["lr_trunk"], lr_critic=CFG["lr_critic"],
                           kl_coef=CFG["kl_coef"],
                           eps_per_iter=CFG["ppo_eps_per_iter"],
                           critic_warmup_iters=CFG["critic_warmup_iters"])

    elif algo == "poly_ppo":
        from poly_ppo import train_poly_ppo
        policy = train_poly_ppo(policy, TRAIN_CONFIGS, device,
                                n_iters=n_iters, lr=CFG["lr_actor"],
                                lr_trunk=CFG["lr_trunk"], lr_critic=CFG["lr_critic"],
                                kl_coef=CFG["kl_coef"],
                                N=CFG["N"], n=CFG["n"], M=CFG["M"],
                                p=CFG["p"], W=CFG["W"])

    avg_r, sr = eval_policy(policy, EVAL_CONFIGS, device,
                            n_rollouts=CFG["n_eval_rollouts"],
                            max_steps=CFG["max_steps"])
    return avg_r, sr


def run_algo(algo, pretrained_state, device):
    print(f"\n[{algo.upper()}] {CFG['n_seeds']} seeds")

    rewards, srs = [], []
    for s in range(CFG["n_seeds"]):
        print(f"\n  seed {s+1}/{CFG['n_seeds']}")
        r, sr = run_seed(algo, pretrained_state, s, device)
        rewards.append(r)
        srs.append(sr)
        print(f"  seed {s+1} done -- reward={r:.3f}, success={sr:.1f}%")

    mean_r = np.mean(rewards)
    mean_sr = np.mean(srs)
    print(f"\n  {algo} final: reward={mean_r:.3f}, success={mean_sr:.1f}%")
    return mean_r, mean_sr


def save_result(line):
    with open(RESULTS_PATH, "a") as f:
        f.write(line + "\n")
    print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["reinforce", "ppo", "poly_ppo", "all"],
                        default="all")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--kl", type=float, default=None,
                        help="KL coefficient, paper sweeps {0.005, 0.01, 0.05, 0.1}")
    parser.add_argument("--lr_actor", type=float, default=None)
    parser.add_argument("--lr_critic", type=float, default=None)
    parser.add_argument("--lr_trunk", type=float, default=None)
    parser.add_argument("--ppo_eps_per_iter", type=int, default=None)
    parser.add_argument("--critic_warmup_iters", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--skip_pretrain", action="store_true")
    args = parser.parse_args()

    if args.iters:
        CFG["n_iters"] = args.iters
        CFG["reinforce_iters"] = args.iters
        CFG["ppo_iters"] = args.iters
        CFG["poly_iters"] = args.iters
    if args.seeds:
        CFG["n_seeds"] = args.seeds
    if args.kl:
        CFG["kl_coef"] = args.kl
    if args.lr_actor is not None:
        CFG["lr_actor"] = args.lr_actor
    if args.lr_critic is not None:
        CFG["lr_critic"] = args.lr_critic
    if args.lr_trunk is not None:
        CFG["lr_trunk"] = args.lr_trunk
    if args.ppo_eps_per_iter is not None:
        CFG["ppo_eps_per_iter"] = args.ppo_eps_per_iter
    if args.critic_warmup_iters is not None:
        CFG["critic_warmup_iters"] = args.critic_warmup_iters
    if args.N is not None:
        CFG["N"] = args.N
    if args.M is not None:
        CFG["M"] = args.M
    if args.p is not None:
        CFG["p"] = args.p

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if not args.skip_pretrain and not os.path.exists(PRETRAIN_PATH):
        from pretrain import pretrain
        pretrain(device=device, save_path=PRETRAIN_PATH)

    env = GoToEnv()
    env.reset()
    n_actions = env.n_actions
    pretrained_policy = load_pretrained(n_actions, device)
    pretrained_state = copy.deepcopy(pretrained_policy.state_dict())

    pre_r, pre_sr = eval_policy(pretrained_policy, EVAL_CONFIGS, device,
                                n_rollouts=CFG["n_eval_rollouts"])
    save_result(f"\nGoto (BabyAI) | n_iters={CFG['n_iters']}")
    save_result(f"Pretrained:  reward={pre_r:.3f}, success={pre_sr:.1f}%")

    algos = ["reinforce", "ppo", "poly_ppo"] if args.algo == "all" else [args.algo]

    for algo in algos:
        mean_r, mean_sr = run_algo(algo, pretrained_state, device)
        save_result(f"{algo:<12} reward={mean_r:.3f}, success={mean_sr:.1f}%")

    print(f"\ndone. results in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
