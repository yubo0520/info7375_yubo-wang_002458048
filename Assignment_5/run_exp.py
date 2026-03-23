import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import os

from a2c import (
    GaussianPolicy, ValueEstimator, VectorizedEnvWrapper,
    a2c
)


ENVS = [
    "Swimmer-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
]

# lam=0 -> 1-step TD, lam=1 -> monte carlo, lam=0.95 -> GAE
ALGORITHMS = {
    "1-step TD (λ=0)":   0.0,
    "Monte Carlo (λ=1)":  1.0,
    "GAE (λ=0.95)":       0.95,
}

# reasonable defaults for mujoco, can tune later
DEFAULT_CFG = {
    "num_envs": 8,
    "pi_lr": 3e-4,
    "v_lr": 1e-3,
    "gamma": 0.99,
    "epochs": 300,
    "train_v_iters": 25,
    "rollout_len": 2048,
    "hidden": 64,
}

SAVE_DIR = "results"


def run_single(env_name, lam, cfg):
    # run one experiment: env + lambda -> returns list
    env = VectorizedEnvWrapper(
        gym.make(env_name), num_envs=cfg["num_envs"]
    )
    agent = GaussianPolicy(env, lr=cfg["pi_lr"], hidden=cfg["hidden"])
    critic = ValueEstimator(env, lr=cfg["v_lr"], hidden=cfg["hidden"])

    print(f"\nRunning experiment: env={env_name}, lam={lam}, epochs={cfg['epochs']}")

    returns = a2c(
        env, agent, critic,
        gamma=cfg["gamma"], lam=lam,
        epochs=cfg["epochs"],
        train_v_iters=cfg["train_v_iters"],
        rollout_traj_len=cfg["rollout_len"],
        verbose=True,
    )
    return returns


def smooth(data, window=10):
    # simple moving average for cleaner plots
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_env(env_name, results, save_path=None):
    # plot learning curves for one env, 3 algorithms
    plt.figure(figsize=(10, 6))
    for alg_name, rets in results.items():
        y = smooth(rets, window=10)
        x = range(len(y))
        sns.lineplot(x=list(x), y=list(y), label=alg_name)

    plt.title(f"A2C Learning Curves: {env_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.close()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    cfg = DEFAULT_CFG.copy()

    all_results = {}

    for env_name in ENVS:
        env_results = {}
        for alg_name, lam in ALGORITHMS.items():
            rets = run_single(env_name, lam, cfg)
            env_results[alg_name] = rets

            # save raw data
            fname = f"{env_name}_{alg_name.split('(')[0].strip().replace(' ', '_')}.npy"
            np.save(os.path.join(SAVE_DIR, fname), np.array(rets))

        all_results[env_name] = env_results

        # plot per env
        safe_name = env_name.replace("-", "_")
        plot_env(env_name, env_results,
                 save_path=os.path.join(SAVE_DIR, f"{safe_name}_curves.png"))

    print("\nFinished. Check the results/ folder for plots.")


if __name__ == "__main__":
    main()