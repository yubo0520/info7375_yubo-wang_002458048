import numpy as np
import torch

from buffer import collect_episode

ENTROPY_COEF = 0.005


def update_reinforce(policy, opt, imgs, dirs, toks, acts, returns):
    logits, values = policy(imgs, dirs, toks)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(acts)

    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss  = -(log_probs * advantages).mean()
    critic_loss = ((values - returns) ** 2).mean()
    entropy     = dist.entropy().mean()   # encourages exploration

    loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    return loss.item()


def train_reinforce(policy, configs, device, n_iters=1000, lr=1e-5,
                    lr_trunk=1e-5, lr_critic=1e-4, eps_per_iter=64):
    from env import GoToEnv
    opt = torch.optim.Adam([
        {"params": policy.trunk_parameters(), "lr": lr_trunk},
        {"params": policy.actor.parameters(), "lr": lr},
        {"params": policy.critic.parameters(), "lr": lr_critic},
    ])
    env = GoToEnv()

    print(f"  REINFORCE | {n_iters} iters, {eps_per_iter} eps/iter")

    for it in range(n_iters):
        batch_imgs, batch_dirs, batch_toks, batch_acts, batch_rets = [], [], [], [], []
        ep_reward = 0.0

        for _ in range(eps_per_iter):
            seed = int(configs[np.random.randint(len(configs))])
            buf, last_val, _ = collect_episode(env, policy, seed, device)
            ep_reward += sum(buf.rewards)

            T = len(buf.rewards)
            rets = np.zeros(T, dtype=np.float32)
            G = last_val
            for t in reversed(range(T)):
                G = buf.rewards[t] + G * (1 - buf.dones[t])
                rets[t] = G

            imgs, dirs, toks, acts, _ = buf.to_tensors(device)
            batch_imgs.append(imgs)
            batch_dirs.append(dirs)
            batch_toks.append(toks)
            batch_acts.append(acts)
            batch_rets.append(torch.tensor(rets, dtype=torch.float32).to(device))

        loss = update_reinforce(
            policy, opt,
            torch.cat(batch_imgs), torch.cat(batch_dirs), torch.cat(batch_toks),
            torch.cat(batch_acts), torch.cat(batch_rets))

        if (it + 1) % 100 == 0:
            print(f"  iter {it+1}/{n_iters} -- "
                  f"reward={ep_reward/eps_per_iter:.3f}, loss={loss:.4f}")

    return policy
