import numpy as np
import torch

from env import GoToEnv, obs_to_arrays
from model import obs_to_tensors, tokenize


def eval_policy(policy, configs, device, n_rollouts=100, max_steps=100):
    # evaluate over fixed configs -- Table 1 metrics: avg reward + success rate
    env = GoToEnv()
    policy.eval()

    total_reward = 0.0
    total_success = 0
    total = 0

    for seed in configs:
        for _ in range(n_rollouts // len(configs)):
            raw_obs = env.reset(seed=seed)
            img, d, mission = obs_to_arrays(raw_obs)

            ep_reward = 0.0
            done = False
            steps = 0
            success = False

            while not done and steps < max_steps:
                img_t, dir_t, tok_t = obs_to_tensors(img, d, mission, device)
                with torch.no_grad():
                    logits, _ = policy(img_t, dir_t, tok_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()

                next_raw, reward, done, _ = env.step(action)
                ep_reward += reward
                if reward > 0:
                    success = True

                img = next_raw["image"].astype(np.float32) / 10.0
                d = int(next_raw["direction"])
                mission = next_raw["mission"]
                steps += 1

            total_reward += ep_reward
            total_success += int(success)
            total += 1

    avg_reward = total_reward / total
    success_rate = total_success / total * 100

    policy.train()
    return avg_reward, success_rate


def print_results(label, avg_reward, success_rate):
    print(f"\n  [{label}] reward={avg_reward:.3f}, success={success_rate:.1f}%")


def eval_passk(policy, configs, device, k=10, max_steps=100):
    # pass@k: fraction of configs solved in at least 1 of k attempts
    env = GoToEnv()
    policy.eval()

    solved = 0
    for seed in configs:
        for attempt in range(k):
            raw_obs = env.reset(seed=seed)
            img, d, mission = obs_to_arrays(raw_obs)
            done = False
            steps = 0
            success = False

            while not done and steps < max_steps:
                img_t, dir_t, tok_t = obs_to_tensors(img, d, mission, device)
                with torch.no_grad():
                    logits, _ = policy(img_t, dir_t, tok_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()   # stochastic for diversity

                next_raw, reward, done, _ = env.step(action)
                if reward > 0:
                    success = True
                img = next_raw["image"].astype(np.float32) / 10.0
                d = int(next_raw["direction"])
                mission = next_raw["mission"]
                steps += 1

            if success:
                solved += 1
                break   # at least one success -- count config as solved

    pass_rate = solved / len(configs) * 100
    policy.train()
    return pass_rate


if __name__ == "__main__":
    # quick sanity: random policy
    import sys
    from env import GoToEnv
    from model import make_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = GoToEnv(seed=0)
    env.reset()
    policy = make_policy(env.n_actions, device)

    configs = list(range(10))
    r, sr = eval_policy(policy, configs, device, n_rollouts=10)
    print(f"random policy: reward={r:.3f}, success={sr:.1f}%")
