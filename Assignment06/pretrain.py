import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from env import GoToEnv, obs_to_arrays
from model import make_policy, obs_to_tensors, tokenize, build_vocab_from_demos


DEMO_PATH = "demos_goto.pkl"
N_DEMOS = 5000
EPOCHS = 30
LR = 1e-3
BATCH = 64
ENTROPY_COEF = 0.01


# 

def gen_demos(n, seed_start=0):
    # generate demos using BotAgent 
    try:
        from minigrid.utils.baby_ai_bot import BabyAIBot as BotAgent
    except ImportError:
        raise ImportError("BotAgent not found -- check minigrid version")

    demos = []
    for i in range(n):
        env = GoToEnv()
        obs = env.reset(seed=seed_start + i)
        bot = BotAgent(env.env.unwrapped)
        traj = []
        done = False
        while not done:
            action = bot.replan()
            img, d, mission = obs_to_arrays(obs)
            traj.append((img.copy(), d, mission, action))
            obs, r, done, _ = env.step(action)
        if traj:
            demos.append(traj)
        if (i + 1) % 100 == 0:
            print(f"  generated {i+1}/{n} demos")
    return demos


def load_or_gen_demos():
    if os.path.exists(DEMO_PATH):
        print(f"loading demos from {DEMO_PATH}")
        with open(DEMO_PATH, "rb") as f:
            return pickle.load(f)
    print(f"generating {N_DEMOS} demos...")
    demos = gen_demos(N_DEMOS)
    with open(DEMO_PATH, "wb") as f:
        pickle.dump(demos, f)
    print(f"saved to {DEMO_PATH}")
    return demos


def demos_to_tensors(demos, device):
    imgs, dirs, toks, actions = [], [], [], []
    for traj in demos:
        for img, d, mission, action in traj:
            imgs.append(img)
            dirs.append(d)
            toks.append(tokenize(mission))
            actions.append(action)

    imgs = torch.tensor(np.array(imgs), dtype=torch.float32).to(device)
    dirs = torch.tensor(dirs, dtype=torch.long).to(device)
    toks = torch.tensor(toks, dtype=torch.long).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    return TensorDataset(imgs, dirs, toks, actions)


# training

def pretrain(device="cpu", save_path="pretrained.pt"):
    demos = load_or_gen_demos()
    build_vocab_from_demos(DEMO_PATH)
    print(f"total steps: {sum(len(d) for d in demos)} across {len(demos)} demos")

    # need n_actions -- peek from env
    env = GoToEnv()
    env.reset()
    n_actions = env.n_actions

    policy = make_policy(n_actions, device)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    dataset = demos_to_tensors(demos, device)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    print(f"\npretraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        n = 0
        for imgs, dirs, toks, acts in loader:
            logits, _ = policy(imgs, dirs, toks)
            loss = ce(logits, acts)

            # entropy bonus -- encourages exploration
            dist = torch.distributions.Categorical(logits=logits)
            loss = loss - ENTROPY_COEF * dist.entropy().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(acts)
            correct += (logits.argmax(-1) == acts).sum().item()
            n += len(acts)

        acc = correct / n * 100
        avg_loss = total_loss / n
        print(f"  epoch {epoch+1}/{EPOCHS} -- loss={avg_loss:.4f}, acc={acc:.1f}%")

    torch.save(policy.state_dict(), save_path)
    print(f"\nsaved to {save_path}")
    return policy


if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    pretrain(device=device)
