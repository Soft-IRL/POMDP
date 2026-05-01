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


# -------- Video helpers (same idea, small cleanup to avoid pix_fmt warning) --------
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

def log_video_to_wandb(frames: np.ndarray, fps: int, step: int, key: str = "eval/video"):
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
        # Logging without step is fine; TB provides the x-axis.
        wandb.log({key: wandb.Video(tmp_path, format="mp4")})

    except Exception as e:
        # Fallback: GIF
        try:
            gif_path = tmp_path[:-4] + ".gif"
            imageio.mimsave(gif_path, frames, fps=fps)
            wandb.log({key: wandb.Video(gif_path, fps=fps, format="gif")})
        finally:
            raise RuntimeError(f"Video encoding failed (mp4->gif fallback attempted). Original error: {e}")


# -------- Evaluation (adapted to C51) ---------------------------------------------
def evaluate_policy(env, net, *, episodes=5, seed=0, record_video=False, video_name="eval_rollout", fps=15):
    """
    Returns (returns_list, steps_list, successes, optional_frames).
    If record_video=True, returns frames as a numpy array (T, H, W, 3) uint8.
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
                a = q.argmax(dim=1).item()

            obs, r, terminated, truncated, info = env.step(a)
            ep_ret += r
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

        # fixed support z_i
        support = torch.linspace(self.v_min, self.v_max, self.n_atoms)
        self.register_buffer("support", support)  # (N,)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, self.n_actions * self.n_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # logits: (B, A, N)
        logits = self.net(x).view(-1, self.n_actions, self.n_atoms)
        return logits

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        # probs: (B, A, N)
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        # (B, A)
        probs = self.dist(x)
        q = (probs * self.support.view(1, 1, -1)).sum(dim=-1)
        return q


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

    # Bellman-shifted atoms
    tz = rewards[:, None] + gamma * (1.0 - dones[:, None]) * support[None, :]
    tz = tz.clamp(v_min, v_max)

    b = (tz - v_min) / delta_z  # in [0, N-1] ideally
    l = torch.floor(b).long()
    u = torch.ceil(b).long()

    l = l.clamp(0, N - 1)
    u = u.clamp(0, N - 1)

    eq = (u == l)          # true integer case (after numeric + clamp safety)
    ne = ~eq

    m = torch.zeros(B, N, device=device)

    offset = (torch.arange(B, device=device) * N)[:, None]  # (B,1)

    l_idx = (l + offset).reshape(-1)
    u_idx = (u + offset).reshape(-1)

    # non-integer mass split
    m.view(-1).index_add_(0, l_idx, (p_next * (u.float() - b) * ne.float()).reshape(-1))
    m.view(-1).index_add_(0, u_idx, (p_next * (b - l.float()) * ne.float()).reshape(-1))

    # exact-integer: dump full mass at l
    m.view(-1).index_add_(0, l_idx, (p_next * eq.float()).reshape(-1))

    # normalize (numerical safety)
    m = m.clamp_min(0)
    m = m / (m.sum(dim=1, keepdim=True) + 1e-8)
    return m


# -------- Training: C51 with W&B logging ------------------------------------------
def train_c51_zero_uncertainty(
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
    render_eval_every: int = 50_000,     # log a video this often
    save_model_every: int = 500_000,      # save checkpoint artifact this often
    track: bool = True,
    project: str = "light-dark-c51",
    run_name: str | None = None,

    # C51 params
    n_atoms: int = 51,
    v_min: float = -5.0,
    v_max: float = 5.0,
    double_c51: bool = True,            # argmax by online net, dist by target net
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

    total_steps = 0
    next_eval_at = eval_every_steps
    next_video_eval_at = render_eval_every
    next_save_at = save_model_every

    obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
    obs = obs / world_radius
    ep_ret = 0.0
    ep_len = 0
    ep_idx = 0

    pbar = tqdm(total=total_training_steps, desc="C51 training (steps)")
    running_loss = None

    while total_steps < total_training_steps:
        total_steps += 1
        ep_len += 1

        eps = epsilon(total_steps)

        # act
        if total_steps < start_steps or rng.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                o = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                q = net.q_values(o)  # (1,A)
                a = int(q.argmax(dim=1).item())

        next_obs, r, term, trunc, info = env.step(a)
        next_obs = next_obs / world_radius
        d = float(term or trunc)

        a_np = np.array([a], dtype=np.int64)          # (1,)
        r_np = np.array([r], dtype=np.float32)        # (1,)
        d_np = np.array([bool(d)], dtype=np.bool_)
        buf.add(obs, next_obs, a_np, r_np, d_np, infos=[info])


        ep_ret += r
        obs = next_obs

        # learn
        step_loss = None
        if total_steps >= train_after and total_steps % train_every == 0 and buf.size() >= batch_size:
            batch = buf.sample(batch_size)
            o = batch.observations                         # (B, obs_dim)
            no = batch.next_observations                   # (B, obs_dim)
            a_b = batch.actions.long().squeeze(-1)         # (B,)
            r_b = batch.rewards.squeeze(-1)                # (B,)
            d_b = batch.dones.squeeze(-1).float()          # (B,)

            # Predicted log-prob for taken actions
            logits = net(o)                                # (B,A,N)
            logp = F.log_softmax(logits, dim=-1)            # (B,A,N)
            logp_a = logp[torch.arange(batch_size, device=device), a_b]  # (B,N)

            with torch.no_grad():
                # Get next-state distributions
                if double_c51:
                    # a* from online net expectation
                    q_next_online = net.q_values(no)        # (B,A)
                    a_star = q_next_online.argmax(dim=1)    # (B,)
                    p_next = tgt.dist(no)[torch.arange(batch_size, device=device), a_star]  # (B,N)
                else:
                    # greedy by target net expectation
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
                )  # (B,N)

            # Cross-entropy loss: - sum m * log p(s,a)
            loss = -(m * logp_a).sum(dim=1).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()

            step_loss = float(loss.item())
            running_loss = 0.95 * running_loss + 0.05 * step_loss if running_loss is not None else step_loss

        if total_steps % target_update == 0:
            tgt.load_state_dict(net.state_dict())

        # ----- log scalars -----
        writer.add_scalar("train/epsilon", eps, total_steps)
        if step_loss is not None:
            writer.add_scalar("train/loss", step_loss, total_steps)
        if running_loss is not None:
            writer.add_scalar("train/loss_ema", running_loss, total_steps)

        # --- Periodic evaluation (no video) ---
        if total_steps >= next_eval_at:
            rets, steps_list, succ, _ = evaluate_policy(
                env, net, episodes=5, seed=rng.integers(0, 1_000_000), record_video=False
            )
            writer.add_scalar("eval/return_mean", float(np.mean(rets)), total_steps)
            writer.add_scalar("eval/return_std", float(np.std(rets)), total_steps)
            writer.add_scalar("eval/success", int(succ), total_steps)
            next_eval_at += eval_every_steps

        # --- Less frequent evaluation WITH video upload ---
        if total_steps >= next_video_eval_at:
            rets, steps_list, succ, vid = evaluate_policy(
                env, net, episodes=5, seed=rng.integers(0, 1_000_000),
                record_video=True, video_name=f"eval_{total_steps}"
            )
            if track and vid is not None:
                log_video_to_wandb(vid, fps=15, step=total_steps, key="eval/video")
            next_video_eval_at += render_eval_every

        # --- Save model weights to W&B artifacts periodically ---
        if total_steps >= next_save_at:
            ckpt = f"c51_{total_steps}.pt"
            torch.save(net.state_dict(), ckpt)
            if track:
                art = wandb.Artifact("c51_net", type="model")
                art.add_file(ckpt)
                wandb.log_artifact(art)
            next_save_at += save_model_every

        pbar.update(1)

        # episode end
        if term or trunc or ep_len >= max_episode_steps:
            writer.add_scalar("episode/return", float(ep_ret), total_steps)
            ep_idx += 1
            obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
            obs = obs / world_radius
            ep_ret = 0.0
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
        sigma_dark=0.0,     # zero uncertainty baseline
        sigma_light=0.0,
        include_goal_in_obs=True,
        randomize_start=True,
        randomize_goal=True,
        min_start_goal_dist=6.0,
        require_opposite_band_side=False,
    )

    train_c51_zero_uncertainty(
        env,
        project="light-dark-c51",
        run_name=None,
        # You can tighten these once you see return ranges:
        n_atoms=51,
        v_min=-5.0,
        v_max=5.0,
        double_c51=True,
    )
