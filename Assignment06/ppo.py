import numpy as np
import torch
import torch.nn.functional as F

from buffer import collect_episode

ENTROPY_COEF = 0.002


def train_ppo(policy, configs, device, n_iters=1000, lr=1e-5,
              lr_trunk=1e-5, lr_critic=1e-4, kl_coef=0.01,
              eps_per_iter=64, critic_warmup_iters=0):
    from env import GoToEnv
    opt = torch.optim.Adam([
        {"params": policy.trunk_parameters(), "lr": lr_trunk},
        {"params": policy.actor.parameters(), "lr": lr},
        {"params": policy.critic.parameters(), "lr": lr_critic},
    ])
    env = GoToEnv()

    print(f"  PPO | {n_iters} iters, kl={kl_coef}, {eps_per_iter} eps/iter")

    for it in range(n_iters):
        all_imgs, all_dirs, all_toks, all_acts, all_lp, all_advs, all_rets = \
            [], [], [], [], [], [], []
        ep_reward = 0.0

        for _ in range(eps_per_iter):
            seed = int(configs[np.random.randint(len(configs))])
            buf, last_val, _ = collect_episode(env, policy, seed, device)
            ep_reward += sum(buf.rewards)

            advantages, returns = buf.compute_gae(last_val, gamma=1.0, lam=0.95)
            imgs, dirs, toks, acts, lp = buf.to_tensors(device)

            all_imgs.append(imgs);  all_dirs.append(dirs);  all_toks.append(toks)
            all_acts.append(acts);  all_lp.append(lp.detach())
            all_advs.append(torch.tensor(advantages, dtype=torch.float32).to(device))
            all_rets.append(torch.tensor(returns, dtype=torch.float32).to(device))

        imgs   = torch.cat(all_imgs)
        dirs   = torch.cat(all_dirs)
        toks   = torch.cat(all_toks)
        acts   = torch.cat(all_acts)
        old_lp = torch.cat(all_lp)
        advs   = torch.cat(all_advs)
        rets   = torch.cat(all_rets)
        advs   = (advs - advs.mean()) / (advs.std() + 1e-8)

        T   = len(acts)
        idx = np.arange(T)
        total_loss = 0.0

        for _ in range(2):
            np.random.shuffle(idx)
            for start in range(0, T, 64):
                mb = torch.tensor(idx[start:start + 64], dtype=torch.long)
                logits, values = policy(imgs[mb], dirs[mb], toks[mb])
                dist     = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(acts[mb])
                entropy   = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_lp[mb])
                adv   = advs[mb]
                surr  = torch.min(ratio * adv,
                                  torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv)
                actor_loss  = -surr.mean()
                critic_loss = F.smooth_l1_loss(values, rets[mb])
                kl = (old_lp[mb].exp() * (old_lp[mb] - log_probs)).mean()

                warmup = it < critic_warmup_iters
                if warmup:
                    loss = 0.5 * critic_loss
                else:
                    loss = (actor_loss
                            + 0.5 * critic_loss
                            + kl_coef * kl
                            - ENTROPY_COEF * entropy)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()

        if (it + 1) % 100 == 0:
            tag = " (critic warmup)" if (it + 1) <= critic_warmup_iters else ""
            print(f"  iter {it+1}/{n_iters} -- "
                  f"reward={ep_reward/eps_per_iter:.3f}, loss={total_loss:.4f}{tag}")

    return policy
