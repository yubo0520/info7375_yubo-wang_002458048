import torch
import torch.nn as nn
import numpy as np
import pickle


# mission vocabulary 
_vocab = {"<pad>": 0, "<unk>": 1}


def reset_vocab():
    global _vocab
    _vocab = {"<pad>": 0, "<unk>": 1}


def build_vocab_from_demos(demo_path):
    # rebuild vocab in deterministic order from saved demos
    # this keeps mission token ids aligned with pretrained.pt
    if not demo_path:
        return
    try:
        with open(demo_path, "rb") as f:
            demos = pickle.load(f)
    except Exception:
        return
    reset_vocab()
    for traj in demos:
        for _img, _d, mission, _action in traj:
            for w in mission.lower().split():
                if w not in _vocab:
                    _vocab[w] = len(_vocab)

def tokenize(mission, max_len=10):
    words = mission.lower().split()
    ids = []
    for w in words[:max_len]:
        if w not in _vocab:
            _vocab[w] = len(_vocab)
        ids.append(_vocab[w])
    # pad
    ids += [0] * (max_len - len(ids))
    return ids


def vocab_size():
    return max(len(_vocab), 50)   # leave room for new words


class MissionEncoder(nn.Module):
    # GRU over word embeddings
    def __init__(self, vocab_sz, emb_dim=16, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, token_ids):
        # token_ids: (B, L)
        x = self.embed(token_ids)
        _, h = self.gru(x)
        return h.squeeze(0)   # (B, hidden)


class ImageEncoder(nn.Module):
    # small CNN for 7x7x3 partial obs
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.out_dim = 32 * 7 * 7

    def forward(self, img):
        # img: (B, H, W, C) numpy or tensor -- we permute to (B, C, H, W)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).float()
        return self.net(img)


class Policy(nn.Module):
    # actor-critic: CNN image + direction emb + GRU mission -> action logits + value
    def __init__(self, n_actions, vocab_sz=100, hidden=128):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.dir_emb = nn.Embedding(4, 8)
        self.mission_enc = MissionEncoder(vocab_sz)

        feat_dim = self.img_enc.out_dim + 8 + self.mission_enc.out_dim
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def encode(self, img, direction, token_ids):
        img_feat = self.img_enc(img)
        dir_feat = self.dir_emb(direction)
        mis_feat = self.mission_enc(token_ids)
        x = torch.cat([img_feat, dir_feat, mis_feat], dim=-1)
        return self.trunk(x)

    def forward(self, img, direction, token_ids):
        h = self.encode(img, direction, token_ids)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(self, img, direction, token_ids):
        logits, value = self.forward(img, direction, token_ids)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def trunk_parameters(self):
        return list(self.img_enc.parameters()) + list(self.dir_emb.parameters()) + \
               list(self.mission_enc.parameters()) + list(self.trunk.parameters())


def make_policy(n_actions, device="cpu"):
    vocab_sz = 200   # enough for BabyAI missions
    policy = Policy(n_actions, vocab_sz=vocab_sz).to(device)
    return policy


def obs_to_tensors(img, direction, mission, device="cpu"):
    # convert raw obs arrays to tensors for policy
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)
    dir_t = torch.tensor([direction], dtype=torch.long).to(device)
    tok = tokenize(mission)
    tok_t = torch.tensor(tok, dtype=torch.long).unsqueeze(0).to(device)
    return img_t, dir_t, tok_t


if __name__ == "__main__":
    import torch
    p = make_policy(n_actions=7)
    img = np.random.rand(7, 7, 3).astype(np.float32)
    img_t, dir_t, tok_t = obs_to_tensors(img, 0, "go to the red ball")
    logits, val = p(img_t, dir_t, tok_t)
    print(f"logits: {logits.shape}, value: {val.item():.4f}")
    action, lp, v = p.act(img_t, dir_t, tok_t)
    print(f"action={action}, log_prob={lp.item():.4f}, value={v.item():.4f}")
