import numpy as np
import torch
from env import obs_to_arrays
from model import obs_to_tensors, tokenize


class RolloutBuffer:
    # stores one episode's worth of transitions for policy update
    def __init__(self):
        self.reset()

    def reset(self):
        self.imgs = []
        self.dirs = []
        self.toks = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def push(self, img, direction, tok, action, log_prob, reward, value, done):
        self.imgs.append(img)
        self.dirs.append(direction)
        self.toks.append(tok)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=1.0, lam=0.95):
        # generalized advantage estimation -- Schulman et al. 2018
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        values = [v if isinstance(v, float) else float(v) for v in self.values]
        values_next = values[1:] + [last_value]

        gae = 0.0
        for t in reversed(range(T)):
            delta = self.rewards[t] + gamma * values_next[t] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def to_tensors(self, device):
        imgs = torch.tensor(np.array(self.imgs), dtype=torch.float32).to(device)
        dirs = torch.tensor(self.dirs, dtype=torch.long).to(device)
        toks = torch.tensor(self.toks, dtype=torch.long).to(device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        log_probs = torch.stack(self.log_probs).to(device)
        return imgs, dirs, toks, actions, log_probs

    def __len__(self):
        return len(self.rewards)


def collect_episode(env, policy, seed, device, max_steps=100):
    # reset env with seed and collect one full episode
    raw_obs = env.reset(seed=seed)
    img, d, mission = obs_to_arrays(raw_obs)
    tok = tokenize(mission)

    buf = RolloutBuffer()
    done = False
    steps = 0

    while not done and steps < max_steps:
        img_t, dir_t, tok_t = obs_to_tensors(img, d, mission, device)
        with torch.no_grad():
            action, log_prob, value = policy.act(img_t, dir_t, tok_t)

        next_raw, reward, done, _ = env.step(action)
        buf.push(img.copy(), d, tok[:], action, log_prob.cpu(), reward, float(value.item()), float(done))

        img, d, mission = obs_to_arrays(next_raw)
        tok = tokenize(mission)
        steps += 1

    # bootstrap last value if episode didn't end naturally
    if not done:
        img_t, dir_t, tok_t = obs_to_tensors(img, d, mission, device)
        with torch.no_grad():
            _, last_val = policy(img_t, dir_t, tok_t)
        last_value = float(last_val.item())
    else:
        last_value = 0.0

    total_reward = sum(buf.rewards)
    success = total_reward > 0
    return buf, last_value, success
