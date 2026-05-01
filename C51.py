import math, random, collections, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
from envs.light_dark_navigation_env import make_env
import gymnasium as gym
from gymnasium import Env, Wrapper
from gymnasium.spaces import Discrete
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
import cv2

# ========= import your env =========
# from your_env_file import make_env   # <-- if your env is in another file
# For clarity, I’ll assume make_env is available in scope.

# ---------- Discretize continuous actions ----------
# ---------- Discretize continuous actions ----------
class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env, step: float | None = None, actions: np.ndarray | None = None):
        super().__init__(env)
        if actions is None:
            if step is None:
                step = float(self.unwrapped.cfg.max_speed * self.unwrapped.cfg.dt)
            s = float(step)
            actions = np.array([
                [0.0, 0.0],
                [0.0,  s], [0.0, -s], [-s, 0.0], [ s, 0.0],
                [-s,  s], [ s,  s],  [-s,-s],  [ s,-s],
            ], dtype=np.float32)
        self._A = actions.astype(np.float32)
        self.action_space = spaces.Discrete(len(self._A))      # real Discrete(9)
        self.observation_space = env.observation_space         # unchanged

    def action(self, a: int) -> np.ndarray:                    # now it’s used
        box_a = self._A[int(a)]
        return np.clip(box_a, self.env.action_space.low, self.env.action_space.high).astype(np.float32)
    

def evaluate_policy(env, net, episodes=5, seed=0, render=False, filename="C51_eval.mp4", fps=15):
    #env = DiscreteActions(env)

    rng = np.random.default_rng(seed)
    device = next(net.parameters()).device
    world_radius = env.unwrapped.cfg.world_radius

    returns, steps_list, successes = [], [], 0

    obs, info = env.reset(seed=rng.integers(0, 1_000_000))
    if render:
        frame = env.render()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    for ep in range(episodes):
        if ep > 0:
            obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
            if render:
                frame = env.render()
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        ep_ret, steps = 0.0, 0
        while True:
            with torch.no_grad():
                o = torch.as_tensor(obs/world_radius, dtype=torch.float32, device=device).unsqueeze(0)
                logits = net(o)                          # [1, A, K]
                q = net.q_from_logits(logits)             # [1, A]
                a = q.argmax(dim=1).detach().cpu().numpy()[0]

            obs, r, terminated, truncated, info = env.step(a)

            ep_ret += r; steps += 1
            if render: 
                frame = env.render()
                if frame is not None: writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if terminated or truncated:
                successes += int(terminated)
                break
        returns.append(np.round(ep_ret))
        steps_list.append(steps)


    env.close()
    # duration is seconds per frame
    if render:
        writer.release()
    return returns, steps_list, successes

# ---------- C51 Network ----------
class C51Net(nn.Module):
    def __init__(self, obs_dim:int, n_actions:int, num_atoms:int=51, v_min:float=-5.0, v_max:float=5.0):
        super().__init__()
        self.num_atoms = num_atoms
        self.n_actions = n_actions
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms)  # [K]
        hid = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, n_actions * num_atoms)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits: [B, A, K]
        B = x.shape[0]
        logits = self.net(x).view(B, self.n_actions, self.num_atoms)
        return logits

    def q_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # returns expected values per action: [B, A]
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.support.to(logits.device), dim=-1)

# ---------- C51 distribution projection ----------
def projection_distribution(next_dist_logits: torch.Tensor,
                            rewards: torch.Tensor,
                            dones: torch.Tensor,
                            gamma: float,
                            support: torch.Tensor) -> torch.Tensor:
    """
    next_dist_logits: [B, A, K] (from target net on next_obs)
    rewards, dones: [B]
    returns projected target probs: [B, K] corresponding to the greedy next-action
    """
    device = next_dist_logits.device
    B, A, K = next_dist_logits.shape
    support = support.to(device)  # [K]
    delta_z = (support[-1] - support[0]) / (K - 1)
    # greedy next action on expectation
    next_probs = F.softmax(next_dist_logits, dim=-1)         # [B, A, K]
    next_q = torch.sum(next_probs * support.view(1,1,-1), dim=-1)  # [B, A]
    next_a = torch.argmax(next_q, dim=-1)                    # [B]
    # pick that action's logits
    idx = next_a.view(-1,1,1).expand(-1,1,K)
    next_logits_astar = next_dist_logits.gather(1, idx).squeeze(1) # [B, K]
    next_probs_astar = F.softmax(next_logits_astar, dim=-1)         # [B, K]

    Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma * support.view(1, -1)  # [B, K]
    Tz = Tz.clamp(min=support[0].item(), max=support[-1].item())

    b = (Tz - support[0]) / delta_z  # [B, K]
    l = b.floor().to(torch.int64)
    u = b.ceil().to(torch.int64)

    B_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)

    m = torch.zeros(B, K, device=device)
    m.index_put_((B_idx, l.clamp(0, K-1)), next_probs_astar * (u.float() - b), accumulate=True)
    m.index_put_((B_idx, u.clamp(0, K-1)), next_probs_astar * (b - l.float()), accumulate=True)
    return m  # [B, K]

def rb_add(buf, o, no, a, r, d, info):
    o  = np.asarray(o,  dtype=np.float32)[None, ...]
    no = np.asarray(no, dtype=np.float32)[None, ...]
    a  = np.array([[a]], dtype=np.int64)          # (1,1) <- key
    r  = np.array([r], dtype=np.float32)
    d  = np.array([d], dtype=np.float32)
    buf.add(o, no, a, r, d, infos=[info])

# ---------- Training ----------
def train_c51_env1(
    env,
    total_steps:int = 200_000,
    gamma:float = 0.99,
    lr:float = 5e-4,
    batch_size:int = 128,
    buffer_size:int = 200_000,
    start_steps:int = 1_000,
    train_after:int = 5_000,
    train_every:int = 1,
    target_update:int = 2_000,
    eps_start:float = 1.0,
    eps_end:float = 0.05,
    eps_decay_steps:int = 50_000,
    num_atoms:int = 51,
    v_min:float = -5.0,
    v_max:float = 5.0,
    eval_every:int = 10_000,
    do_render_eval_every: int = 25_000,
    seed:int = 0,
):
    rng = np.random.default_rng(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Wrap discrete actions (use env.unwrapped.cfg.max_speed as primitive step)
    step_mag = env.unwrapped.cfg.max_speed * env.unwrapped.cfg.dt
    world_radius = env.unwrapped.cfg.world_radius
    env = DiscreteActions(env, step=step_mag)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = C51Net(obs_dim, n_actions, num_atoms, v_min, v_max).to(device)
    tgt = C51Net(obs_dim, n_actions, num_atoms, v_min, v_max).to(device)
    tgt.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=lr)

    support = net.support.to(device)

    buf = ReplayBuffer(
        buffer_size=100_000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    def epsilon(t):
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    obs, info = env.reset(seed=seed)
    obs = obs.astype(np.float32)/world_radius
    total = 0
    next_eval_at = eval_every
    next_render_eval_at = do_render_eval_every
    ep_ret, ep_len = 0.0, 0

    with tqdm(total=total_steps, desc="C51-Env1 no-noise") as pbar:
        while total < total_steps:
            # act
            if total < start_steps or random.random() < epsilon(total):
                a = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    ot = torch.tensor(obs/world_radius, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = net(ot)                          # [1, A, K]
                    q = net.q_from_logits(logits)             # [1, A]
                    a = int(torch.argmax(q, dim=1).detach().cpu().numpy()[0])

            next_obs, r, term, trunc, info = env.step(a)
            next_obs = next_obs.astype(np.float32)/world_radius
            d = float(term or trunc)

            rb_add(buf, obs, next_obs, a, r, d, info)

            obs = next_obs
            ep_ret += r; ep_len += 1; total += 1; pbar.update(1)

            # train
            if total >= train_after and total % train_every == 0 and buf.size() >= batch_size:
                batch = buf.sample(batch_size)

                # SB3 tensors already on the right device
                o    = batch.observations              # [B, obs_dim]
                no   = batch.next_observations         # [B, obs_dim]
                a_b  = batch.actions.long().squeeze(-1)   # [B]
                r_b  = batch.rewards.squeeze(-1)          # [B]
                d_b  = batch.dones.squeeze(-1).float()    # [B]

                # current logits for chosen actions
                logits = net(o)  # [B, A, K]
                logits_a = logits.gather(1, a_b.view(-1,1,1).expand(-1,1,net.num_atoms)).squeeze(1)  # [B, K]

                with torch.no_grad():
                    next_logits_tgt = tgt(no)  # [B, A, K]
                    target_probs = projection_distribution(next_logits_tgt, r_b, d_b, gamma, support)  # [B, K]

                logp = F.log_softmax(logits_a, dim=-1)
                loss = -(target_probs * logp).sum(dim=-1).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()

            # target update
            if total % target_update == 0:
                tgt.load_state_dict(net.state_dict())

            # reset episode
            if d:
                o, info = env.reset(seed=rng.integers(0, 1_000_000))
                o = o.astype(np.float32)/world_radius
                ep_ret, ep_len = 0.0, 0

            """# quick eval print
            if total % eval_every == 0:
                rets, succ = evaluate(env, net, device, episodes=5)
                pbar.set_postfix({"eval_return_mean": f"{np.mean(rets):.2f}",
                                  "success": f"{succ}/{len(rets)}"})"""
            
            # evaluation by steps
            if total >= next_eval_at:
                render_eval = total >= next_render_eval_at
                rets, steps, succ = evaluate_policy(
                    env, net,
                    episodes=1,
                    seed=rng.integers(0, 1_000_000),
                    render=render_eval,
                )
                tqdm.write(f"[{total_steps}] Eval | Return {rets} | Steps {steps} | Success {succ}")
                next_eval_at += eval_every
                if render_eval:
                    next_render_eval_at += do_render_eval_every

        #pbar.update(1)

    return net

# ---------- Evaluation ----------
def evaluate(env, net, device, episodes=5, seed=0) -> Tuple[list, int]:
    rng = np.random.default_rng(seed)
    # (env is already wrapped with DiscreteActions)
    world_radius = env.unwrapped.cfg.world_radius
    returns = []; successes = 0
    for _ in range(episodes):
        o, info = env.reset(seed=rng.integers(0,1_000_000))
        o = o.astype(np.float32)
        ep_ret = 0.0
        while True:
            with torch.no_grad():
                ot = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                logits = net(ot)
                q = net.q_from_logits(logits)
                a = int(torch.argmax(q, dim=1).item())
            o, r, term, trunc, info = env.step(a)
            o = o.astype(np.float32); ep_ret += r
            if term or trunc:
                successes += int(term)
                break
        returns.append(ep_ret)
    return returns, successes

# ---------- Run ----------
if __name__ == "__main__":
    # Build Env-1 with NO NOISE
    env = make_env(
        render_mode="rgb_array",
        world_radius=10.0,
        band_width=2.0,
        band_angle_deg=90.0,
        band_center=(-8.0 + 2.0/2, 0.0),
        sigma_dark=0.0,     # <-- no noise
        sigma_light=0.0,    # <-- no noise
        include_goal_in_obs=True,
        randomize_start=False,
        randomize_goal=False,
        noisy_goal_obs=False,
        step_cost=-0.01,
        success_reward=1.0,
        goal_radius=0.5,
        max_steps=200,
    )

    trained = train_c51_env1(env,
                             total_steps=200_000,
                             gamma=0.99,
                             lr=5e-4,
                             batch_size=128,
                             buffer_size=200_000,
                             start_steps=2_000,
                             train_after=5_000,
                             target_update=2_000,
                             eps_decay_steps=50_000,
                             num_atoms=51,
                             v_min=-5.0, v_max=5.0,
                             eval_every=10_000,
                             seed=0)

    """rets, succ = evaluate(DiscreteActions(env, step=env.unwrapped.cfg.max_speed * env.unwrapped.cfg.dt),
                          trained,
                          torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                          episodes=10, seed=1)
    print(f"Eval returns: {rets} | Successes: {succ}/{len(rets)}")"""
