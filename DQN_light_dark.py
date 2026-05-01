from envs.light_dark_navigation_env import make_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env, Wrapper
from gymnasium.spaces import Discrete
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm
import wandb 
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
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

# -------- Q-network (unchanged) ---------------------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        return self.net(x)

def _ensure_even_hw(frames: np.ndarray) -> np.ndarray:
    """yuv420p requires even height/width. Crop last row/col if needed."""
    if frames.ndim != 4:
        return frames
    t, h, w, c = frames.shape
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 != h or w2 != w:
        frames = frames[:, :h2, :w2, :]
    return frames

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
    #frames = np.asarray(frames, dtype=np.uint8)
    #frames = _ensure_even_hw(frames)
    frames = _crop_to_mb(frames, mb=16)

    # Write an mp4 with yuv420p so browsers can decode it reliably
    
    tmp = NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        writer = imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
            macro_block_size=16,
        )
        for f in frames:
            writer.append_data(f)
        writer.close()

        wandb.log({key: wandb.Video(tmp_path, format="mp4")})
    
    except Exception as e:
        # Fallback: log as GIF if mp4 encoding isn't available
        try:
            gif_path = tmp_path[:-4] + ".gif"
            imageio.mimsave(gif_path, frames, fps=fps)
            wandb.log({key: wandb.Video(gif_path, fps=fps, format="gif")})
        finally:
            # Re-raise with context so you can see why mp4 failed
            raise RuntimeError(f"Video encoding failed (mp4->gif fallback attempted). Original error: {e}")
    finally:
        # Let W&B upload first; then you may clean up later if desired.
        pass

# -------- Evaluation with W&B video logging ---------------------------------------
def evaluate_policy(env, q, *, episodes=5, seed=0, record_video=False, video_name="eval_rollout", fps=15):
    """
    Returns (returns_list, steps_list, successes, optional_frames).
    If record_video=True, returns frames as a numpy array (T, H, W, 3) uint8.
    """
    rng = np.random.default_rng(seed)
    device = next(q.parameters()).device
    world_radius = env.unwrapped.cfg.world_radius

    returns, steps_list, successes = [], [], 0
    frames = [] if record_video else None

    obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
    # capture initial frame if requested
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
                o = torch.as_tensor(obs/world_radius, dtype=torch.float32, device=device).unsqueeze(0)
                a = q(o).argmax(dim=1).item()

            obs, r, terminated, truncated, info = env.step(a)
            ep_ret += r; steps += 1

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
    if record_video and len(frames) > 0:
        vid = np.stack(frames, axis=0)  # (T, H, W, 3), uint8
        return returns, steps_list, successes, vid
    return returns, steps_list, successes, None

# -------- Training with W&B logging -----------------------------------------------
def train_dqn_zero_uncertainty(
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
    render_eval_every: int = 50_000,    # log a video this often
    save_model_every: int = 500_000,     # save a checkpoint to W&B artifacts this often^
    track: bool = True,
    project: str = "light-dark-dqn",
    run_name: str | None = None,
):
    
    if track:
    # --- W&B init ---
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

    q = QNet(obs_dim, n_actions).to(device)
    qt = QNet(obs_dim, n_actions).to(device)
    qt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)

    def epsilon(t):
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    total_steps = 0
    next_eval_at = eval_every_steps
    next_video_eval_at = render_eval_every
    next_save_at = save_model_every

    # start first episode
    obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
    obs = obs / world_radius
    ep_ret = 0.0
    ep_len = 0
    ep_idx = 0

    pbar = tqdm(total=total_training_steps, desc="DQN training (steps)")

    running_loss = None
    while total_steps < total_training_steps:
        total_steps += 1
        ep_len += 1

        # act
        eps = epsilon(total_steps)
        if total_steps < start_steps or rng.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                o = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                a = q(o).argmax(dim=1).detach().cpu().numpy()[0]

        next_obs, r, term, trunc, info = env.step(a)
        next_obs = next_obs / world_radius
        d = float(term or trunc)

        buf.add(obs, next_obs, a, r, d, infos=[info])

        ep_ret += r
        obs = next_obs

        # learn
        step_loss = None
        if total_steps >= train_after and total_steps % train_every == 0 and buf.size() >= batch_size:
            batch = buf.sample(batch_size)
            o  = batch.observations
            no = batch.next_observations
            a_b = batch.actions.long().squeeze(-1)
            r_b = batch.rewards.squeeze(-1)
            d_b = batch.dones.squeeze(-1).float()

            q_pred = q(o).gather(1, a_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = qt(no).max(1)[0]
                target = r_b + gamma * (1.0 - d_b) * next_q

            loss = nn.functional.smooth_l1_loss(q_pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            step_loss = loss.item()
            running_loss = 0.95 * running_loss + 0.05 * step_loss if running_loss is not None else step_loss

        if total_steps % target_update == 0:
            qt.load_state_dict(q.state_dict())
        
        # ----- log scalars -----
        writer.add_scalar("train/epsilon", eps, total_steps)
        if step_loss is not None:
            writer.add_scalar("train/loss", step_loss, total_steps)
        if running_loss is not None:
            writer.add_scalar("train/loss_ema", running_loss, total_steps)

        """# --- Log per-step metrics to W&B (lightweight) ---
        wandb.log(
            {
                "train/epsilon": eps,
                "env/episode_length": ep_len,
                **({"train/loss": step_loss} if step_loss is not None else {}),
                **({"train/loss_ema": running_loss} if running_loss is not None else {}),
            },
            step=total_steps,
        )"""

        # --- Periodic evaluation (no video) ---
        if total_steps >= next_eval_at:
            rets, steps_list, succ, _ = evaluate_policy(
                env, q, episodes=5, seed=rng.integers(0, 1_000_000), record_video=False
            )
            writer.add_scalar("eval/return_mean", np.mean(rets), total_steps)
            writer.add_scalar("eval/return_std", np.std(rets), total_steps)
            writer.add_scalar("eval/success", succ, total_steps)
            """wandb.log(
                {
                    "eval/return_mean": float(np.mean(rets)),
                    "eval/return_std": float(np.std(rets)),
                    "eval/steps_mean": float(np.mean(steps_list)),
                    "eval/successes": int(succ),
                },
                step=total_steps,
            )"""
            next_eval_at += eval_every_steps

        # --- Less frequent evaluation WITH video upload ---
        if total_steps >= next_video_eval_at:
            rets, steps_list, succ, vid = evaluate_policy(
                env, q, episodes=5, seed=rng.integers(0, 1_000_000),
                record_video=True, video_name=f"eval_{total_steps}"
            )
            if track and vid is not None:
                """wandb.log(
                    {
                        "videos/eval": wandb.Video(vid, fps=15, format="mp4"),
                        "eval_video/return_mean": np.mean(rets),
                        "eval_video/success": succ,
                    },
                    step=total_steps,
                )"""
                log_video_to_wandb(vid, fps=15,step=total_steps + 1, key="eval/video")
            next_video_eval_at += render_eval_every

        # --- Save model weights to W&B artifacts periodically ---
        if total_steps >= next_save_at:
            ckpt = f"qnet_{total_steps}.pt"
            torch.save(q.state_dict(), ckpt)
            if track:
                art = wandb.Artifact("qnet", type="model")
                art.add_file(ckpt)
                wandb.log_artifact(art)
            next_save_at += save_model_every

        pbar.update(1)

        # episode end
        if term or trunc or ep_len >= max_episode_steps:
            writer.add_scalar("episode/return", ep_ret, total_steps)
            """wandb.log(
                {"episode/return": float(ep_ret), "episode/index": ep_idx},
                step=total_steps,
            )"""
            ep_idx += 1
            obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
            obs = obs / world_radius
            ep_ret = 0.0
            ep_len = 0

    pbar.close()
    env.close()
    writer.close()
    wandb.finish()
    return

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
        randomize_goal=False,
        min_start_goal_dist=6.0,
        require_opposite_band_side=False,
    )

    # Optional: set your run name via ENV or here
    train_dqn_zero_uncertainty(env, project="light-dark-dqn")
