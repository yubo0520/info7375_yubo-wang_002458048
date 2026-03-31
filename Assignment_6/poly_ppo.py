import numpy as np
import torch
import torch.nn.functional as F

from env import obs_to_arrays
from model import obs_to_tensors, tokenize
from buffer import RolloutBuffer, collect_episode

ENTROPY_COEF = 0.002
DIVERSITY_BONUS = 0.3


# diversity / polychromic objective

def diversity(rooms_list):
    """
    Fraction of semantically distinct trajectories in the set.
    d = 0 when all trajectories visit the same rooms (paper Sec 4).
    rooms_list: list of frozensets of room coordinates.
    """
    n = len(rooms_list)
    if n <= 1:
        return 0.0
    n_distinct = len(set(rooms_list))
    return (n_distinct - 1) / (n - 1)


def poly_score(rewards, rooms_list):
    """
    f_poly = mean_reward * diversity  (Eq. 7).
    Both R and d normalised in [0,1] by construction.
    A small diversity bonus keeps exploration gradients alive before the
    policy learns to succeed.
    """
    mean_r = float(np.mean(rewards))
    d      = diversity(rooms_list)
    return mean_r * d + DIVERSITY_BONUS * d


# vine rollout helpers


def rollout_from_state(env, policy, saved_state, device, max_steps=100, temperature=2.0):
    env.restore_state(saved_state)

    raw_obs = env.env.unwrapped.gen_obs()
    img, d, mission = obs_to_arrays(raw_obs)
    tok = tokenize(mission)

    buf  = RolloutBuffer()
    done = False
    steps = 0

    while not done and steps < max_steps:
        img_t, dir_t, tok_t = obs_to_tensors(img, d, mission, device)
        with torch.no_grad():
            logits, value = policy.forward(img_t, dir_t, tok_t)
            dist = torch.distributions.Categorical(logits=logits / temperature)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=device))

        next_raw, reward, done, _ = env.step(action)
        buf.push(img.copy(), d, tok[:], action, log_prob.cpu(),
                 reward, float(value.item()), float(done))

        img, d, mission = obs_to_arrays(next_raw)
        tok = tokenize(mission)
        steps += 1

    total_reward = sum(buf.rewards)
    rooms = env.room_sequence()
    return buf, total_reward, rooms


def collect_vines(env, policy, seed, device, N=8, p=2, max_steps=100):
    """
    Collect seed rollout then generate N vine rollouts from p equally-spaced
    states along the seed trajectory (paper Sec A.1.1).
    """
    seed_buf, last_val, _ = collect_episode(env, policy, seed, device, max_steps)

    T = len(seed_buf.rewards)
    if T < 2:
        return seed_buf, last_val, []

    vine_data = []
    for k in range(1, p + 1):
        t_idx = int(k * T / (p + 1))

        # replay seed actions 
        env.reset(seed=seed)
        for t in range(t_idx):
            env.step(seed_buf.actions[t])
        saved = env.save_state()

        vines = []
        for _ in range(N):
            remaining = max(1, max_steps - t_idx)
            vine_buf, r, rooms = rollout_from_state(
                env, policy, saved, device, remaining)
            vines.append((vine_buf, r, rooms))
        vine_data.append((saved, vines))

    return seed_buf, last_val, vine_data


def make_groups(vines, n=4, M=4):
    groups = []
    if len(vines) < n:
        return groups
    seen = set()
    max_trials = M * 20
    trials = 0
    while len(groups) < M and trials < max_trials:
        idx = tuple(sorted(np.random.choice(len(vines), size=n, replace=False)))
        if idx not in seen:
            seen.add(idx)
            groups.append([vines[i] for i in idx])
        trials += 1
    return groups

# PPO minibatch update (shared kernel)


def _ppo_minibatch_update(policy, opt, imgs, dirs, toks, acts, old_lp, advs, rets,
                          eps=0.2, kl_coef=0.01, epochs=2, batch=64):
    T   = len(acts)
    idx = np.arange(T)
    total_loss = 0.0

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, T, batch):
            mb = torch.tensor(idx[start:start + batch], dtype=torch.long)
            logits, values = policy(imgs[mb], dirs[mb], toks[mb])
            dist      = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(acts[mb])
            entropy   = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_lp[mb])
            adv   = advs[mb]
            surr  = torch.min(ratio * adv,
                              torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
            actor_loss  = -surr.mean()
            critic_loss = F.smooth_l1_loss(values, rets[mb])
            kl = (old_lp[mb].exp() * (old_lp[mb] - log_probs)).mean()

            loss = (actor_loss
                    + 0.5 * critic_loss
                    + kl_coef * kl
                    - ENTROPY_COEF * entropy)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

    return total_loss

def poly_ppo_update(policy, opt, seed_buf, last_val, vine_data,
                    extra_bufs=None,
                    gamma=1.0, lam=0.95, eps=0.2, kl_coef=0.01,
                    n=4, M=4, W=5, epochs=2, batch=64, device="cpu"):

    gae_imgs, gae_dirs, gae_toks, gae_acts, gae_lp, gae_advs, gae_rets = \
        [], [], [], [], [], [], []
    poly_imgs, poly_dirs, poly_toks, poly_acts, poly_lp, poly_advs, poly_rets = \
        [], [], [], [], [], [], []

    # --- seed rollout + extra episodes: standard GAE
    gae_bufs = [(seed_buf, last_val)]
    if extra_bufs:
        gae_bufs.extend(extra_bufs)

    for buf, lv in gae_bufs:
        advantages, returns = buf.compute_gae(lv, gamma, lam)
        imgs_s, dirs_s, toks_s, acts_s, lp_s = buf.to_tensors(device)
        gae_imgs.append(imgs_s);  gae_dirs.append(dirs_s);  gae_toks.append(toks_s)
        gae_acts.append(acts_s);  gae_lp.append(lp_s.detach())
        gae_advs.append(torch.tensor(advantages, dtype=torch.float32).to(device))
        gae_rets.append(torch.tensor(returns,    dtype=torch.float32).to(device))

    # --- vine rollouts: polychromic advantage (Algorithm 2, lines 4-5)
    for _, vines in vine_data:
        groups = make_groups(vines, n=n, M=M)
        scores  = [poly_score([g[1] for g in grp], [g[2] for g in grp])
                   for grp in groups]
        baseline = float(np.mean(scores))

        for gi, grp in enumerate(groups):
            adv_val = scores[gi] - baseline
            for vine_buf, r, _rooms in grp:
                if len(vine_buf.rewards) == 0:
                    continue
                # apply polychromic advantage only within the window W
                W_len      = min(W + 1, len(vine_buf.rewards))
                vine_advs  = [adv_val] * W_len
                vine_rets_ = [r]       * W_len   # sparse return = total reward

                imgs_v, dirs_v, toks_v, acts_v, lp_v = vine_buf.to_tensors(device)
                # truncate to window -- cleaner than padding with zeros
                poly_imgs.append(imgs_v[:W_len]);   poly_dirs.append(dirs_v[:W_len])
                poly_toks.append(toks_v[:W_len]);   poly_acts.append(acts_v[:W_len])
                poly_lp.append(lp_v[:W_len].detach())
                poly_advs.append(
                    torch.tensor(vine_advs,  dtype=torch.float32).to(device))
                poly_rets.append(
                    torch.tensor(vine_rets_, dtype=torch.float32).to(device))

    # --- normalise GAE and poly advantages independently, then concatenate
    all_imgs = gae_imgs + poly_imgs
    all_dirs = gae_dirs + poly_dirs
    all_toks = gae_toks + poly_toks
    all_acts = gae_acts + poly_acts
    all_lp   = gae_lp   + poly_lp
    all_rets = gae_rets  + poly_rets

    norm = lambda t: (t - t.mean()) / (t.std() + 1e-8)

    gae_adv_cat  = norm(torch.cat(gae_advs))
    if poly_advs:
        poly_adv_cat = torch.cat(poly_advs)
        if poly_adv_cat.std() > 1e-8:
            poly_adv_cat = norm(poly_adv_cat)
        all_advs_cat = torch.cat([gae_adv_cat, poly_adv_cat])
    else:
        all_advs_cat = gae_adv_cat

    return _ppo_minibatch_update(
        policy, opt,
        torch.cat(all_imgs), torch.cat(all_dirs), torch.cat(all_toks),
        torch.cat(all_acts), torch.cat(all_lp),
        all_advs_cat, torch.cat(all_rets),
        eps=eps, kl_coef=kl_coef, epochs=epochs, batch=batch)

def train_poly_ppo(policy, configs, device, n_iters=1000, lr=1e-5,
                   lr_trunk=1e-5, lr_critic=1e-4, kl_coef=0.01,
                   N=8, n=4, M=8, p=4, W=5):
    from env import GoToEnv
    opt = torch.optim.Adam([
        {"params": policy.trunk_parameters(), "lr": lr_trunk},
        {"params": policy.actor.parameters(), "lr": lr},
        {"params": policy.critic.parameters(), "lr": lr_critic},
    ])
    env = GoToEnv()

    print(f"  Poly-PPO | {n_iters} iters, N={N} n={n} M={M} p={p} W={W} "
          f"kl={kl_coef} ent={ENTROPY_COEF}")

    for it in range(n_iters):
        seed = int(configs[np.random.randint(len(configs))])
        seed_buf, last_val, vine_data = collect_vines(
            env, policy, seed, device, N=N, p=p)

        # additional episodes with standard GAE (matches paper's trajectory budget)
        extra_bufs = []
        for _ in range(3):
            s = int(configs[np.random.randint(len(configs))])
            b, lv, _ = collect_episode(env, policy, s, device)
            extra_bufs.append((b, lv))

        if vine_data:
            loss = poly_ppo_update(
                policy, opt, seed_buf, last_val, vine_data,
                extra_bufs=extra_bufs,
                kl_coef=kl_coef, n=n, M=M, W=W, device=device)
        else:
            # fallback: plain PPO if vine sampling failed
            from ppo import train_ppo as _ppo
            # single-step PPO update on seed + extra
            bufs = [(seed_buf, last_val)] + extra_bufs
            all_imgs, all_dirs, all_toks, all_acts, all_lp, all_advs, all_rets = \
                [], [], [], [], [], [], []
            for buf, lv in bufs:
                advantages, returns = buf.compute_gae(lv)
                imgs_s, dirs_s, toks_s, acts_s, lp_s = buf.to_tensors(device)
                all_imgs.append(imgs_s); all_dirs.append(dirs_s)
                all_toks.append(toks_s); all_acts.append(acts_s)
                all_lp.append(lp_s.detach())
                all_advs.append(
                    torch.tensor(advantages, dtype=torch.float32).to(device))
                all_rets.append(
                    torch.tensor(returns,    dtype=torch.float32).to(device))
            adv_cat = torch.cat(all_advs)
            adv_cat = (adv_cat - adv_cat.mean()) / (adv_cat.std() + 1e-8)
            loss = _ppo_minibatch_update(
                policy, opt,
                torch.cat(all_imgs), torch.cat(all_dirs), torch.cat(all_toks),
                torch.cat(all_acts), torch.cat(all_lp),
                adv_cat, torch.cat(all_rets),
                kl_coef=kl_coef)

        if (it + 1) % 100 == 0:
            ep_reward = sum(seed_buf.rewards)
            n_vines   = sum(len(v) for _, v in vine_data) if vine_data else 0
            print(f"  iter {it+1}/{n_iters} -- reward={ep_reward:.3f}, "
                  f"loss={loss:.4f}, vines={n_vines}")

    return policy
