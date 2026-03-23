import numpy as np
import gymnasium as gym
import itertools
import json
import os
import time

from a2c import (
    GaussianPolicy, ValueEstimator, VectorizedEnvWrapper,
    a2c
)


# search space 

PARAM_GRID = {
    "num_envs":  [4, 8],
    "pi_lr":     [3e-4, 1e-3],
    "v_lr":      [1e-3, 1e-2],
    "gamma":     [0.99],
    "lam":       [0.0, 0.95, 1.0],
}

# fixed during grid search
FIXED = {
    "epochs": 150,        # shorter than full run, just enough to compare
    "train_v_iters": 25,
    "rollout_len": 2048,
    "hidden": 64,
}

ENVS = [
    "Swimmer-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
]

SAVE_DIR = "grid_results"


def eval_config(env_name, params):
    # run one config, return mean of last 20% returns as score
    try:
        env = VectorizedEnvWrapper(
            gym.make(env_name), num_envs=params["num_envs"], env_name=env_name
        )
        agent = GaussianPolicy(env, lr=params["pi_lr"], hidden=FIXED["hidden"])
        critic = ValueEstimator(env, lr=params["v_lr"], hidden=FIXED["hidden"])

        rets = a2c(
            env, agent, critic,
            gamma=params["gamma"], lam=params["lam"],
            epochs=FIXED["epochs"],
            train_v_iters=FIXED["train_v_iters"],
            rollout_traj_len=FIXED["rollout_len"],
            verbose=False,
        )

        # score = mean of last 20% epochs
        tail = max(1, len(rets) // 5)
        score = np.mean(rets[-tail:])
        return score, rets

    except Exception as e:
        print(f"  error: {e}")
        return float("-inf"), []


def grid_search(env_name):
    # enumerate all combos, track best
    keys = list(PARAM_GRID.keys())
    vals = [PARAM_GRID[k] for k in keys]
    combos = list(itertools.product(*vals))

    print(f"\nRunning grid search for {env_name}.")
    print(f"Testing {len(combos)} parameter combinations.")

    best_score = float("-inf")
    best_params = None
    results_log = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        t0 = time.time()

        score, rets = eval_config(env_name, params)
        dt = time.time() - t0

        results_log.append({
            "params": params,
            "score": float(score),
            "time": round(dt, 1),
        })

        tag = ""
        if score > best_score:
            best_score = score
            best_params = params.copy()
            tag = " (best so far)"

        print(f"  [{i+1}/{len(combos)}] score={score:.1f} ({dt:.0f}s) "
              f"envs={params['num_envs']} pi_lr={params['pi_lr']} "
              f"v_lr={params['v_lr']} gamma={params['gamma']} "
              f"lam={params['lam']}{tag}")

    return best_params, best_score, results_log


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    summary = {}

    for env_name in ENVS:
        best_p, best_s, log = grid_search(env_name)

        summary[env_name] = {
            "best_params": best_p,
            "best_score": float(best_s),
        }

        # save log
        safe = env_name.replace("-", "_")
        with open(os.path.join(SAVE_DIR, f"{safe}_grid.json"), "w") as f:
            json.dump(log, f, indent=2, default=str)

        print(f"\nBest score for {env_name}: {best_s:.1f}")
        print(f"Parameters: {best_p}")

    # save summary
    with open(os.path.join(SAVE_DIR, "best_params.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nSummary")
    for env, info in summary.items():
        print(f"\n{env}:")
        print(f"  score: {info['best_score']:.1f}")
        for k, v in info['best_params'].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()