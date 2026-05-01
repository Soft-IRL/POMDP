from envs.light_dark_navigation_env import make_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gymnasium import Wrapper
from gymnasium.spaces import Discrete
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
from tempfile import NamedTemporaryFile
import imageio.v2 as imageio


# -------- Discretizer (unchanged) -------------------------------------------------
class DiscreteActions(Wrapper):
    """Map a 2D Box action env to a 9-action discrete grid: (dx,dy) in {-1,0,1}*max_speed."""
    def __init__(self, env):
        super().__init__(env)
        ms = env.unwrapped.cfg.max_speed
        grid = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                grid.append([dx * ms, dy * ms])
        self._grid = np.asarray(grid, dtype=np.float32)
        self.action_space = Discrete(len(self._grid))
        self.observation_space = env.observation_space

    def step(self, a_idx):
        a = self._grid[int(a_idx)]
        return self.env.step(a)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# -------- Video helpers ------------------------------------------------------------
def _crop_to_mb(frames: np.ndarray, mb: int = 16) -> np.ndarray:
    """Crop frames so H and W are divisible by macroblock size (default 16)."""
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4:
        return frames
    t, h, w, c = frames.shape
    h2 = h - (h % mb)
    w2 = w - (w % mb)
    if h2 <= 0 or w2 <= 0:
        return frames
    return frames[:, :h2, :w2, :]

def log_video_to_wandb(frames: np.ndarray, fps: int, key: str = "eval/video"):
    """
    Encode frames to an H.264 mp4 with yuv420p pixel format (Firefox-compatible),
    then log to W&B.
    """
    frames = _crop_to_mb(frames, mb=16)

    tmp = NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        writer = imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",          # avoids "multiple -pix_fmt" warning
            ffmpeg_params=["-movflags", "+faststart"],
            macro_block_size=16,
        )
        for f in frames:
            writer.append_data(f)
        writer.close()

        # With sync_tensorboard=True, avoid wandb.log(..., step=...) if you use it.
        wandb.log({key: wandb.Video(tmp_path, format="mp4")})

    except Exception as e:
        # Fallback: GIF
        try:
            gif_path = tmp_path[:-4] + ".gif"
            imageio.mimsave(gif_path, frames, fps=fps)
            wandb.log({key: wandb.Video(gif_path, fps=fps, format="gif")})
        finally:
            raise RuntimeError(f"Video encoding failed (mp4->gif fallback attempted). Original error: {e}")


# -------- Evaluation (extrinsic-only; unchanged) ----------------------------------
def evaluate_policy(env, net, *, episodes=5, seed=0, record_video=False, fps=15):
    """
    Returns (returns_list, steps_list, successes, optional_frames).
    Important: returns are EXTRINSIC ONLY (whatever env.step returns).
    """
    rng = np.random.default_rng(seed)
    device = next(net.parameters()).device
    world_radius = env.unwrapped.cfg.world_radius

    returns, steps_list, successes = [], [], 0
    frames = [] if record_video else None

    obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
    if record_video:
        frame = env.render()
        if frame is not None:
            frames.append(frame.astype(np.uint8))

    for ep in range(episodes):
        if ep > 0:
            obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
            if record_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame.astype(np.uint8))

        ep_ret, steps = 0.0, 0
        while True:
            with torch.no_grad():
                o = torch.as_tensor(obs / world_radius, dtype=torch.float32, device=device).unsqueeze(0)
                q = net.q_values(o)                 # (1, A)
                a = int(q.argmax(dim=1).item())

            obs, r, terminated, truncated, info = env.step(a)
            ep_ret += float(r)
            steps += 1

            if record_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame.astype(np.uint8))

            if terminated or truncated:
                successes += int(terminated)
                break

        returns.append(float(ep_ret))
        steps_list.append(int(steps))

    env.close()
    if record_video and frames:
        vid = np.stack(frames, axis=0)
        return returns, steps_list, successes, vid
    return returns, steps_list, successes, None


# -------- C51 network --------------------------------------------------------------
class C51Net(nn.Module):
    """
    Outputs logits over atoms for each action: (B, A, N).
    Q(s,a) is computed as expectation over the support.
    """
    def __init__(self, obs_dim: int, n_actions: int, n_atoms: int, v_min: float, v_max: float):
        super().__init__()
        self.n_actions = int(n_actions)
        self.n_atoms = int(n_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        support = torch.linspace(self.v_min, self.v_max, self.n_atoms)
        self.register_buffer("support", support)  # (N,)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, self.n_actions * self.n_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.n_actions, self.n_atoms)  # (B,A,N)

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)                  # (B,A,N)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        p = self.dist(x)                                           # (B,A,N)
        return (p * self.support.view(1, 1, -1)).sum(dim=-1)        # (B,A)


@torch.no_grad()
def c51_projection(p_next, rewards, dones, v_min, v_max, support, gamma):
    """
    p_next:  (B, N)  probs for next-state chosen action
    rewards: (B,)
    dones:   (B,) in {0,1}
    support: (N,) atoms
    returns: (B, N) projected distribution
    """
    device = p_next.device
    B, N = p_next.shape
    delta_z = (v_max - v_min) / (N - 1)

    tz = rewards[:, None] + gamma * (1.0 - dones[:, None]) * support[None, :]
    tz = tz.clamp(v_min, v_max)

    b = (tz - v_min) / delta_z
    l = torch.floor(b).long().clamp(0, N - 1)
    u = torch.ceil(b).long().clamp(0, N - 1)

    eq = (u == l)
    ne = ~eq

    m = torch.zeros(B, N, device=device)

    offset = (torch.arange(B, device=device) * N)[:, None]
    l_idx = (l + offset).reshape(-1)
    u_idx = (u + offset).reshape(-1)

    # fractional mass where l != u
    m.view(-1).index_add_(0, l_idx, (p_next * (u.float() - b) * ne.float()).reshape(-1))
    m.view(-1).index_add_(0, u_idx, (p_next * (b - l.float()) * ne.float()).reshape(-1))

    # integer case: dump full mass to l
    m.view(-1).index_add_(0, l_idx, (p_next * eq.float()).reshape(-1))

    m = m.clamp_min(0)
    m = m / (m.sum(dim=1, keepdim=True) + 1e-8)
    return m


# -------- Entropy helpers ----------------------------------------------------------
@torch.no_grad()
def entropy_of_action_dist(logits_action: torch.Tensor) -> torch.Tensor:
    """
    logits_action: (N,) logits over atoms for one (s,a)
    returns: scalar entropy in nats
    """
    p = F.softmax(logits_action, dim=-1)
    return -(p * p.clamp_min(1e-8).log()).sum()

@torch.no_grad()
def entropy_drop_bonus(net: C51Net,
                       obs_t: np.ndarray,
                       obs_tp1: np.ndarray,
                       a_t: int,
                       done: bool,
                       device: torch.device) -> tuple[float, float, float]:
    """
    Returns (H_t, H_tp1, bonus) where bonus = H_t - H_{t+1} (0 if done).
    Uses greedy a* at t+1 for H_{t+1}.
    """
    if done:
        return 0.0, 0.0, 0.0

    o_t = torch.from_numpy(obs_t).float().to(device).unsqueeze(0)    # (1,obs_dim)
    o_tp1 = torch.from_numpy(obs_tp1).float().to(device).unsqueeze(0)

    logits_t = net(o_t)[0, a_t]                                     # (N,)
    H_t = entropy_of_action_dist(logits_t)

    q_tp1 = net.q_values(o_tp1)                                     # (1,A)
    a_star = int(q_tp1.argmax(dim=1).item())
    logits_tp1 = net(o_tp1)[0, a_star]                              # (N,)
    H_tp1 = entropy_of_action_dist(logits_tp1)

    bonus = (H_t - H_tp1)
    return float(H_t.item()), float(H_tp1.item()), float(bonus.item())


# -------- Training: C51 + entropy-drop intrinsic ----------------------------------
def train_c51(
    env,
    total_training_steps: int = 300_000,
    max_episode_steps: int = 200,
    gamma: float = 0.99,
    lr: float = 5e-4,
    batch_size: int = 128,
    start_steps: int = 1_000,
    train_after: int = 5_000,
    train_every: int = 1,
    target_update: int = 2_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 100_000,
    seed: int = 0,
    eval_every_steps: int = 10_000,
    render_eval_every: int = 50_000,
    track: bool = True,
    project: str = "light-dark-c51",
    run_name: str | None = None,

    # C51 params
    n_atoms: int = 51,
    v_min: float = -7.0,   # widened default: safer once intrinsic reward is added
    v_max: float = 7.0,
    double_c51: bool = True,

    # Intrinsic: entropy reduction
    use_entropy_bonus: bool = True,
    alpha_max: float = 3e-3,        # start small: 1e-3 .. 1e-2
    alpha_warmup_steps: int = 20_000,
    intrinsic_clip: float = 0.05,   # clip to prevent early instability
):
    if track:
        wandb.login()
        run = wandb.init(
            project=project,
            name=run_name,
            sync_tensorboard=True,
            config=dict(
                total_training_steps=total_training_steps,
                max_episode_steps=max_episode_steps,
                gamma=gamma, lr=lr, batch_size=batch_size,
                start_steps=start_steps, train_after=train_after,
                train_every=train_every, target_update=target_update,
                eps_start=eps_start, eps_end=eps_end, eps_decay_steps=eps_decay_steps,
                eval_every_steps=eval_every_steps, render_eval_every=render_eval_every,
                n_atoms=n_atoms, v_min=v_min, v_max=v_max, double_c51=double_c51,
                use_entropy_bonus=use_entropy_bonus,
                alpha_max=alpha_max, alpha_warmup_steps=alpha_warmup_steps,
                intrinsic_clip=intrinsic_clip,
            ),
            monitor_gym=True,
            save_code=True,
        )
        run.log_code(root=".")

    writer = SummaryWriter(f"runs/{run_name}")

    env.unwrapped.cfg.max_steps = max_episode_steps
    env = DiscreteActions(env)
    world_radius = env.unwrapped.cfg.world_radius

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    net = C51Net(obs_dim, n_actions, n_atoms, v_min, v_max).to(device)
    tgt = C51Net(obs_dim, n_actions, n_atoms, v_min, v_max).to(device)
    tgt.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=lr)

    def epsilon(t):
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    def alpha_schedule(t):
        if not use_entropy_bonus:
            return 0.0
        return alpha_max * min(1.0, t / max(1, alpha_warmup_steps))

    total_steps = 0
    next_eval_at = eval_every_steps
    next_video_eval_at = render_eval_every

    obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
    obs = obs / world_radius

    ep_ret_ext = 0.0
    ep_ret_total = 0.0
    ep_len = 0

    pbar = tqdm(total=total_training_steps, desc="C51 + entropy-drop (steps)")
    running_loss = None

    while total_steps < total_training_steps:
        total_steps += 1
        ep_len += 1

        eps = epsilon(total_steps)
        alpha = alpha_schedule(total_steps)

        # ---- act (ε-greedy on expected Q) ----
        if total_steps < start_steps or rng.random() < eps:
            a = int(env.action_space.sample())
        else:
            with torch.no_grad():
                o_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                q = net.q_values(o_t)  # (1,A)
                a = int(q.argmax(dim=1).item())

        # ---- env step (extrinsic reward) ----
        next_obs, r_ext, term, trunc, info = env.step(a)
        next_obs = next_obs / world_radius
        done = bool(term or trunc)

        # ---- intrinsic: entropy drop H_t - H_{t+1} (0 if done) ----
        H_t = H_tp1 = bonus = 0.0
        r_int = 0.0
        if use_entropy_bonus and (not done):
            H_t, H_tp1, bonus = entropy_drop_bonus(net, obs, next_obs, a, done=False, device=device)
            r_int = alpha * bonus
            if intrinsic_clip is not None and intrinsic_clip > 0:
                r_int = float(np.clip(r_int, -intrinsic_clip, intrinsic_clip))

        r_total = float(r_ext) + float(r_int)

        # ---- store transition (use TOTAL reward for learning) ----
        a_np = np.array([a], dtype=np.int64)
        r_np = np.array([r_total], dtype=np.float32)
        d_np = np.array([done], dtype=np.bool_)
        buf.add(obs, next_obs, a_np, r_np, d_np, infos=[info])

        # ---- bookkeeping: log extrinsic vs total episode returns ----
        ep_ret_ext += float(r_ext)
        ep_ret_total += float(r_total)

        # log per-step intrinsic stats (helpful to debug)
        writer.add_scalar("train/epsilon", eps, total_steps)
        writer.add_scalar("train/alpha", alpha, total_steps)
        writer.add_scalar("train/reward_ext", float(r_ext), total_steps)
        writer.add_scalar("train/reward_int", float(r_int), total_steps)
        writer.add_scalar("train/reward_total", float(r_total), total_steps)
        if use_entropy_bonus and (not done):
            writer.add_scalar("train/entropy_t", float(H_t), total_steps)
            writer.add_scalar("train/entropy_tp1", float(H_tp1), total_steps)
            writer.add_scalar("train/entropy_drop", float(bonus), total_steps)

        # ---- advance obs ----
        obs = next_obs

        # ---- learn ----
        step_loss = None
        if total_steps >= train_after and total_steps % train_every == 0 and buf.size() >= batch_size:
            batch = buf.sample(batch_size)
            o = batch.observations                         # (B, obs_dim)
            no = batch.next_observations                   # (B, obs_dim)
            a_b = batch.actions.long().squeeze(-1)         # (B,)
            r_b = batch.rewards.squeeze(-1)                # (B,)  <-- total reward
            d_b = batch.dones.squeeze(-1).float()          # (B,)

            logits = net(o)                                 # (B,A,N)
            logp = F.log_softmax(logits, dim=-1)             # (B,A,N)
            logp_a = logp[torch.arange(batch_size, device=device), a_b]  # (B,N)

            with torch.no_grad():
                if double_c51:
                    q_next_online = net.q_values(no)         # (B,A)
                    a_star = q_next_online.argmax(dim=1)     # (B,)
                    p_next = tgt.dist(no)[torch.arange(batch_size, device=device), a_star]  # (B,N)
                else:
                    q_next_tgt = tgt.q_values(no)
                    a_star = q_next_tgt.argmax(dim=1)
                    p_next = tgt.dist(no)[torch.arange(batch_size, device=device), a_star]

                m = c51_projection(
                    p_next=p_next,
                    rewards=r_b,
                    dones=d_b,
                    support=net.support,
                    v_min=net.v_min,
                    v_max=net.v_max,
                    gamma=gamma,
                )

            loss = -(m * logp_a).sum(dim=1).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()

            step_loss = float(loss.item())
            running_loss = 0.95 * running_loss + 0.05 * step_loss if running_loss is not None else step_loss

        if total_steps % target_update == 0:
            tgt.load_state_dict(net.state_dict())

        if step_loss is not None:
            writer.add_scalar("train/loss", step_loss, total_steps)
        if running_loss is not None:
            writer.add_scalar("train/loss_ema", running_loss, total_steps)

        # ---- evaluation (extrinsic-only) ----
        if total_steps >= next_eval_at:
            rets, steps_list, succ, _ = evaluate_policy(
                env, net, episodes=5, seed=rng.integers(0, 1_000_000), record_video=False
            )
            writer.add_scalar("eval/return_mean_ext", float(np.mean(rets)), total_steps)
            writer.add_scalar("eval/return_std_ext", float(np.std(rets)), total_steps)
            writer.add_scalar("eval/success", int(succ), total_steps)
            next_eval_at += eval_every_steps

        if total_steps >= next_video_eval_at:
            rets, steps_list, succ, vid = evaluate_policy(
                env, net, episodes=5, seed=rng.integers(0, 1_000_000), record_video=True
            )
            if track and vid is not None:
                log_video_to_wandb(vid, fps=15, key="eval/video")
            next_video_eval_at += render_eval_every

        pbar.update(1)

        # ---- episode end ----
        if done or ep_len >= max_episode_steps:
            writer.add_scalar("episode/return_ext", float(ep_ret_ext), total_steps)
            writer.add_scalar("episode/return_total", float(ep_ret_total), total_steps)
            obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
            obs = obs / world_radius
            ep_ret_ext = 0.0
            ep_ret_total = 0.0
            ep_len = 0

    pbar.close()
    env.close()
    writer.close()
    if track:
        wandb.finish()


# -------- Main --------------------------------------------------------------------
if __name__ == "__main__":
    env = make_env(
        render_mode="rgb_array",
        world_radius=10.0,
        max_speed=0.5,
        goal_radius=1.0,
        band_center=(-9.0, 0.0),
        band_angle_deg=90.0,
        band_width=2.0,

        # Start with noise to test the hypothesis:
        # sigma_dark high, sigma_light low
        sigma_dark=2.0,
        sigma_light=0.1,

        include_goal_in_obs=True,
        randomize_start=True,
        randomize_goal=True,
        min_start_goal_dist=6.0,
        require_opposite_band_side=False,
    )

    train_c51(
        env,
        project="light-dark-c51-entropy-drop",
        run_name=None,
        n_atoms=51,
        v_min=-7.0,
        v_max=7.0,
        double_c51=True,

        use_entropy_bonus=True,
        alpha_max=3e-3,
        alpha_warmup_steps=20_000,
        intrinsic_clip=0.05,
    )
