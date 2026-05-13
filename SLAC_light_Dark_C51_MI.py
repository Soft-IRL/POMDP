import os
from dataclasses import dataclass
import tyro
import copy
import numpy as np
import pickle  as pkl
import time
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import torch
from torch.distributions import MultivariateNormal, Normal, Independent, Bernoulli, kl_divergence
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from gymnasium import Env, Wrapper
from gymnasium.spaces import Discrete

from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import imageio.v2 as imageio
import wandb

from SLAC_Agent_deterministic_tabular import ModelDistributionNetwork, MLPDecoder
from SLAC_Agent_D3QN_tabular import D3QNAgent
from SequenceReplayBuffer import SequenceReplayBuffer
from envs.light_dark_navigation_env import make_env

@dataclass
class Args:
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = False
    """if toggled, the trained model will be saved to disk"""
    from_scratch: bool = False
    """if toggled, the model will be trained from scratch"""
    ckpt_path = "checkpoints//light_dark/light_dark_full_pretrain_checkpoint//model_pretrained_kl_teacher.pth"
    """If not from scratch, path to the pretrained model"""

    env_id: str = "LightDarkNavigation-v0"
    """the id of the environment"""
    sigma_dark = 0.0
    """the standard deviation of the dark region noise"""
    sigma_light = 0.0
    """the standard deviation of the light region noise"""

    # Algorithm specific arguments
    total_timesteps: int = 500_000
    """total timesteps of the experiments"""
    max_episode_steps: int = 200
    """max timesteps per episode"""
    pretrain_steps: int = 100_000
    """number of pretraining steps for the world model"""
    seed : int = 42
    """random seed of the experiment"""
    num_envs: int = 1
    """number of parallel environments"""
    q_learning_rate: float = 5e-4
    """the learning rate of the q_network optimizer"""
    m_learning_rate: float = 1e-4
    """the learning rate of the model_network optimizer"""
    alpha_lr: float = 3e-4
    """the learning rate of the alpha optimizer"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.2
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    gamma: float = 0.99
    """the discount factor"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    target_network_frequency: int = 1000
    """the frequency of target network update"""
    tau: float = 0.005
    """the polyak averaging factor for target network update"""
    sequence_len : int = 8
    """the length of the sequence for training"""
    buffer_size: int = 100_000
    """the replay memory buffer size"""
    kl_analytic: bool = True
    """if toggled, the KL divergence will be computed analytically"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    base_depth: int = 32
    """the base depth of the model network"""
    latent1_size: int = 32
    """the size of the first latent variable"""
    latent2_size: int = 256
    """the size of the second latent variable"""
    hidden_dims: tuple = (256, 256)
    """the hidden dimensions of the Q-network"""

    # ========= C51 (distributional critic) =========
    N_atoms: int = 51
    """Number of atoms for the categorical return distribution (C51)."""
    Q_min: float = -1.0
    """Minimum value of the support for C51."""
    Q_max: float = 1.0
    """Maximum value of the support for C51."""

    # ========= Mutual-information intrinsic bonus =========
    mi_use: bool = True
    """If toggled, adds MI-based intrinsic reward."""
    mi_num_samples: int = 4
    """Posterior samples per timestep for MI estimation."""
    mi_beta: float = 0.1
    """Weight of MI bonus added to env reward: r_total_{t+1}=r_env_{t+1}+mi_beta*MI_t."""


    # Logging
    track: bool = True
    """If toggled, logs metrics & videos to Weights & Biases."""
    wandb_project_name: str = "light-dark-slac"
    """W&B project name"""
    wandb_entity: str | None = None
    """W&B entity (team/user). None = default."""
    disable_wandb_service: bool = True
    """Windows stability: disable W&B service process (helps avoid WinError 10053)."""
    video_fps: int = 15
    """FPS for encoded evaluation videos."""
    eval_every: int = 10_000
    """How often to run evaluation (steps)."""
    video_every: int = 50_000
    """How often to log an evaluation video (steps)."""


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
        # observation_space is the same as the base env
        self.observation_space = env.observation_space

    def step(self, a_idx):
        a = self._grid[int(a_idx)]
        #print(f"Discrete action {a_idx} -> {a}")  
        return self.env.step(a)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

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

def encode_mp4_yuv420p(frames: np.ndarray, fps: int, out_path: str) -> str:
    """
    Encode frames to H.264 mp4 with yuv420p pixel format (browser compatible).
    """
    frames = _crop_to_mb(frames, mb=16)
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",              # <-- avoids the "multiple -pix_fmt" warning
        ffmpeg_params=["-movflags", "+faststart"],
        macro_block_size=16,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    return out_path

def log_video_to_wandb(frames: np.ndarray, fps: int, key: str, step: int, run):
    """
    Encode mp4 then log to W&B.
    IMPORTANT: do NOT pass step=... to wandb when sync_tensorboard=True.
              Instead we log 'global_step' and use define_metric.
    """

    frames = _crop_to_mb(frames, mb=16)

    tmp = NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    encode_mp4_yuv420p(frames, fps=fps, out_path=tmp_path)

    # W&B ignores fps when you pass a file path; fps is already baked into the mp4.
    run.log({key: wandb.Video(tmp_path, format="mp4"), "global_step": step})


class Qagent():
    """
    Discrete-action soft-Q / DQN agent for SLAC latents.
    - Input  : concat(z1, z2)  (size = latent1 + latent2)
    - Output : Q-values for all actions
    """

    def __init__(self, n_actions, args):
        self.state_size = args.latent1_size + args.latent2_size
        self.hidden_dims = args.hidden_dims
        self.n_actions = int(n_actions)
        self.target_entropy = -0.98 * np.log(self.n_actions)
        self.device = args.device
        self.lr = args.q_learning_rate
        self.alpha_lr = args.alpha_lr
        self.epsilon = args.start_e
        self.gamma = args.gamma
        self.min_log_alpha = np.log(1e-4)

        # 1. Q-network & target network
        self.q_net        = self._build_mlp(self.state_size, self.n_actions, self.hidden_dims).to(self.device)
        self.q_target_net = copy.deepcopy(self.q_net).eval().requires_grad_(False)

        # ---------------- learnable log_alpha ------------
        init_alpha = 0.5
        self.log_alpha = torch.tensor(np.log(init_alpha),
                                      requires_grad=True,
                                      device=self.device)
        
        # ---------- optimizers -------------------------------------------
        self.q_opt     = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
    
    @staticmethod
    def _build_mlp(in_dim, out_dim, hidden_dims):
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        return nn.Sequential(*layers)
    
    @torch.no_grad()
    def act(self, z: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
        """
        ε-greedy over Q-values.
        Returns action indices of shape (B, 1)
        """
        z = F.layer_norm(z, z.shape[-1:])
        q = self.q_net(z)                                    # (B, A)
        greedy = q.argmax(dim=1, keepdim=True)               # (B, 1)
        return greedy

    def compute_loss(self, z1, z2, actions, rewards, dones):
        B, S, d1 = z1.shape
        d = d1 + z2.size(-1)

        z_all   = torch.cat([z1, z2], dim=-1)   # (B, S, d)
        z_all = F.layer_norm(z_all, z_all.shape[-1:])
        a_all   = actions.long()                # (B, S)
        r_all   = rewards
        done_all= dones.float()

        # --- regular transitions: t = 0..S-2
        z_t   = z_all[:, :-1]                   # (B, S-1, d)
        z_tp1 = z_all[:,  1:]                   # (B, S-1, d)
        a_t   = a_all[:, :-1]                   # (B, S-1)
        r_tp1   = r_all[:, 1:]                   # (B, S-1)
        d_tp1   = done_all[:, 1:]                # (B, S-1)

        BT = B*(S-1)
        z_t_f   = z_t.reshape(BT, d)
        z_tp1_f = z_tp1.reshape(BT, d)
        a_t_f   = a_t.reshape(BT)
        r_tp1_f  = r_tp1.reshape(BT)
        d_tp1_f  = d_tp1.reshape(BT)

        with torch.no_grad():
             # 1) online net chooses argmax action at s_{t+1}
            q_online_tp1 = self.q_net(z_tp1_f)                      # (BT, A)
            a_star     = q_online_tp1.argmax(dim=1, keepdim=True) # (BT,1)

            # 2) target net evaluates that action
            q_target_tp1 = self.q_target_net(z_tp1_f).gather(1, a_star).squeeze(-1)  # (BT,)

            # terminal masking via (1 - done)
            y_main = r_tp1_f + self.gamma * (1.0 - d_tp1_f) * q_target_tp1

        # Q(s_t, a_t) prediction
        q_main = self.q_net(z_t_f).gather(1, a_t_f.unsqueeze(-1)).squeeze(-1)  # (BT,)

        # actions must be 0..A-1
        assert a_t_f.min().item() >= 0 and a_t_f.max().item() < self.n_actions, \
            f"Bad action indices in buffer: [{a_t_f.min().item()}, {a_t_f.max().item()}]"

        # targets must be 1D and match BT
        assert y_main.ndim == 1 and y_main.shape[0] == z_t_f.shape[0]
        
        q_loss = 0.5 * F.mse_loss(q_main, y_main, reduction="mean")

        return q_loss, q_main
    
    def update(self, q_loss):
        self.q_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_opt.step()
    
    def update_target_model(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())
    
    @torch.no_grad()
    def get_td_error(self, z1, z2, actions, rewards, dones):
        B, S, d1 = z1.shape
        d = d1 + z2.size(-1)

        z_all   = torch.cat([z1, z2], dim=-1)   # (B, S, d)
        z_all = F.layer_norm(z_all, z_all.shape[-1:])
        a_all   = actions.long()                # (B, S)
        r_all   = rewards
        done_all= dones.float()

        # --- regular transitions: t = 0..S-2
        z_t   = z_all[:, :-1]                   # (B, S-1, d)
        z_tp1 = z_all[:,  1:]                   # (B, S-1, d)
        a_t   = a_all[:, :-1]                   # (B, S-1)
        r_tp1   = r_all[:, 1:]                   # (B, S-1)
        d_tp1   = done_all[:, 1:]                # (B, S-1)

        BT = B*(S-1)
        z_t_f   = z_t.reshape(BT, d)
        z_tp1_f = z_tp1.reshape(BT, d)
        a_t_f   = a_t.reshape(BT)
        r_tp1_f  = r_tp1.reshape(BT)
        d_tp1_f  = d_tp1.reshape(BT)

            # 1) online net chooses argmax action at s_{t+1}
        q_online_tp1 = self.q_net(z_tp1_f)                      # (BT, A)
        a_star     = q_online_tp1.argmax(dim=1, keepdim=True) # (BT,1)

        # 2) target net evaluates that action
        q_target_tp1 = self.q_target_net(z_tp1_f).gather(1, a_star).squeeze(-1)  # (BT,)

        # terminal masking via (1 - done)
        y_main = r_tp1_f + self.gamma * (1.0 - d_tp1_f) * q_target_tp1

        # Q(s_t, a_t) prediction
        q_main = self.q_net(z_t_f).gather(1, a_t_f.unsqueeze(-1)).squeeze(-1)  # (BT,)

        q_all_t = self.q_net(z_t_f)   # (BT, A)
        td_err_taken = torch.zeros_like(y_main)
        # This is the TD error for the action actually taken (what the loss uses)
        td_err_taken = (q_all_t.gather(1, a_t_f.unsqueeze(-1)).squeeze(-1) - y_main)

        mean_abs = []
        for a in range(self.n_actions):
            mask = (a_t_f == a)
            if mask.any():
                q_a  = q_all_t[mask, a]
                y_a  = y_main[mask]          # IMPORTANT: target must be matched to the taken action only
                td_a = (q_a - y_a).abs().mean().item()
            else:
                td_a = float('nan')
            mean_abs.append(td_a)
        return mean_abs
    
    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)
    
def save_world_model_ckpt(model, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
    }
    torch.save(ckpt, path)
    print(f"[✓] Saved world model checkpoint at {path}")

def load_from_newest_run(model, env_id, args, base_dir="checkpoints\\LightDarkNavigation",
                         filename="model_pretrained_kl_teacher.pth"):
    base = Path(base_dir)
    run_dirs = [p for p in base.glob(f"{env_id}__*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {base} for env_id={env_id}")
    latest_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    ckpt_path = latest_dir / filename
    print(f"[✓] Loaded world model checkpoint from {ckpt_path}")
    load_model_ckpt(model, args=args, ckpt_path=ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("step"), ckpt_path

def load_model_ckpt(Model, args, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=args.device)

    if Model.obs_kind == "image":
        # only image decoders have deconv1
        latent_dim = args.latent1_size + args.latent2_size
        if getattr(Model.decoder, "deconv1", None) is not None:
            if Model.decoder.deconv1.weight.shape[0] != latent_dim:
                Model.decoder.deconv1 = nn.ConvTranspose2d(
                    latent_dim, 8*args.base_depth, kernel_size=4, stride=1, padding=0
                ).to(args.device)
    else:
        # tabular path: make sure decoder exists (since it’s lazy normally)
        if Model.decoder is None:
            latent_dim = args.latent1_size + args.latent2_size
            Model.decoder = MLPDecoder(
                latent_dim, Model.tabular_dim, hidden=Model.decoder_mlp_hidden
            ).to(args.device)

    # finally load weights (strict=False in case ckpt has extra keys)
    missing, unexpected = Model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)


class FreezeParams:
    def __init__(self, params):
        self.params = list(params)
        self.prev = None
    def __enter__(self):
        self.prev = [p.requires_grad for p in self.params]
        for p in self.params:
            p.requires_grad_(False)
    def __exit__(self, *exc):
        for p, r in zip(self.params, self.prev):
            p.requires_grad_(r)

def huber(x, delta=1.0):
    a = x.abs()
    return torch.where(a < delta, 0.5 * a * a, delta * (a - 0.5 * delta))



@torch.no_grad()
def compute_mi_bonus(
    agent,
    q_z1,
    q_z2,
    actions,
    step_types,
    mi_num_samples: int,
):
    """
    Approximate I(Z_t ; R | a_t) for the *taken actions*, using the existing
    C51 critic.

    Inputs:
        q_z1, q_z2 : StackedNormal-like objects from Model.sample_posterior
                     (have .dists which is an Independent(Normal(...),1))
        actions    : (B, S) int64 tensors from replay buffer
        step_types : (B, S) int64 tensors (unused except for shape; kept for clarity)
        mi_num_samples: int, number of posterior samples for Monte-Carlo MI

    Output:
        mi_bonus : (B, S-1) tensor, MI per transition t (attached to r_{t+1})
    """
    device = agent.device
    B, S = actions.shape
    T = S - 1                   # number of transitions in each sequence

    # Sample multiple latent trajectories from the *posterior* distributions
    # q_z1, q_z2.  Shape: (K, B, S, d1/d2)
    K = mi_num_samples
    z1_samples = q_z1.dists.rsample((K,))    # (K, B, S, d1)
    z2_samples = q_z2.dists.rsample((K,))    # (K, B, S, d2)

    # We only need z_t (t=0..T-1) for transitions.
    z1_t = z1_samples[:, :, :-1, :]          # (K, B, T, d1)
    z2_t = z2_samples[:, :, :-1, :]          # (K, B, T, d2)
    z_t  = torch.cat([z1_t, z2_t], dim=-1)   # (K, B, T, d)

    # Actions at time t (aligned with z_t).
    a_t = actions[:, :-1].to(device)         # (B, T)
    B_, T_ = a_t.shape
    assert B_ == B and T_ == T
    BT = B * T

    a_t_flat = a_t.reshape(-1)               # (BT,)
    idx_bt   = torch.arange(BT, device=device)

    probs_list = []

    for k in range(K):
        # z_t[k] : (B, T, d)
        z_flat = z_t[k].reshape(BT, -1)                # (BT, d)
        # same normalisation as in act()/compute_loss
        z_flat = F.layer_norm(z_flat, z_flat.shape[-1:])

        logits = agent._logits(z_flat)                 # (BT, A, N_atoms)
        probs  = logits.softmax(dim=-1)                # (BT, A, N_atoms)
        p_taken = probs[idx_bt, a_t_flat]              # (BT, N_atoms)
        probs_list.append(p_taken)

    # Stack probs over the posterior samples
    # p_stack: (BT, K, N_atoms)
    p_stack = torch.stack(probs_list, dim=1)
    p_stack = p_stack.clamp(min=1e-8)

    # Marginal distribution over returns (average across z)
    # p_mean: (BT, N_atoms)
    p_mean = p_stack.mean(dim=1)

    def entropy(p):
        # p: (..., N_atoms)
        p = p.clamp(min=1e-8)
        return -(p * p.log()).sum(dim=-1)

    # H(R | a, "unknown z")  — entropy of the marginal
    H_total = entropy(p_mean)                 # (BT,)

    # H(R | a, z)  averaged over posterior samples
    H_each  = entropy(p_stack)               # (BT, K)
    H_cond  = H_each.mean(dim=-1)            # (BT,)

    # I(Z;R | a) = H_total - H_cond
    I_flat = (H_total - H_cond).clamp(min=0.0)   # (BT,)

    # Reshape back to (B, T)  and return
    mi_bonus = I_flat.view(B, T)
    return mi_bonus

def compute_loss(model, images, actions, step_types, step=None, rewards=None, discounts=None, latent_posterior_samples_and_dists=None, use_kl=False,  rollout_K=3):
    #If not provided, sample the latent variables and distributions from the encoder (inference model) conditioned on the current sequence.
    if latent_posterior_samples_and_dists is None:
        latent_posterior_samples_and_dists = model.sample_posterior(images, actions, step_types) # q(z1_0 | x0)  , q(z2_0 | z1_0), q(z1_t | x_t, z2_{t-1}, a_{t-1}), q(z2_t | z1_t, z2_{t-1}, a_{t-1})
        
    #Latent variables and their corresponding distributions for both z1 and z2.
    (z1_post, z2_post), (q_z1, q_z2) = latent_posterior_samples_and_dists
    model._ensure_tabular_decoder(z1_post, z2_post)
    preds_imgs   = model.decoder(z1_post, z2_post)
    mse = ((images - preds_imgs)**2).mean()
    output = {"mse": mse}

    if use_kl:
        p_z1, p_z2, p_z1_auto, p_z2_auto = model.get_prior(z1_post, z2_post, actions, step_types) # For every t=0…T−1: pψ(zt+1∣zt2,at) and pψ(zt+12∣zt+1,zt2,at)
        kl_z1 = kl_divergence(q_z1.dists, p_z1.dists).sum(-1)

        q1 = q_z1.dists.base_dist
        p1 = p_z1.dists.base_dist

        # KL balancing + free bits
        tau = 0.02; alpha = 0.8
        p1_det = torch.distributions.Normal(p1.loc.detach(), p1.scale.detach())
        q1_det = torch.distributions.Normal(q1.loc.detach(), q1.scale.detach())

        kl_q_raw = torch.distributions.kl_divergence(q1, p1_det)          # (B,T+1,D)
        kl_p_raw = torch.distributions.kl_divergence(q1_det, p1)

        kl_q = (kl_q_raw - tau).clamp_min(0).sum(-1).mean()               # scalar
        kl_p = (kl_p_raw - tau).clamp_min(0).sum(-1).mean()

        kl_bal = alpha * kl_q + (1 - alpha) * kl_p
        target = 0.2  # aim each auxiliary to be ~20% of recon
        kl_term   = (target * mse.detach() / (kl_bal.detach() + 1e-8)).clamp_(0, 1.0) * kl_bal
        #pred_term = (target * mse.detach() / (pred_loss.detach() + 1e-8)).clamp_(0, 1.0) * pred_loss


        # ----- 2) One-step prior consistency (GT inputs → predict t+1) ------------
        with torch.no_grad():
            z1_det, z2_det = z1_post.detach(), z2_post.detach()

        p_z1, p_z2, p_z1_auto, p_z2_auto = model.get_prior(z1_det, z2_det, actions, step_types)

        z1_next = z1_det[:, 1:]                 # (B,T,·)
        z2_next = z2_det[:, 1:]                 # (B,T,·)
        mu1 = p_z1_auto.base_dist.loc           # (B,T,·)
        mu2 = p_z2_auto.base_dist.loc           # (B,T,·)
        
        latent_tf_mse = ((mu1 - z1_next)**2).mean() + ((mu2 - z2_next)**2).mean()

        # Pixel one-step (decode predicted latents vs x_{t+1})
        x_pred_next = model.decoder(mu1, mu2)                      # (B,T,1,H,W)
        pix_tf_mse  = ((images[:, 1:] - x_pred_next) ** 2).mean()

        loss = mse + kl_term + 0.1*latent_tf_mse + pix_tf_mse

        output["kl_z1"] = kl_z1.mean()
        output["kl_q_raw"] = kl_q_raw.mean()
        output["kl_q"] = kl_q
        output["kl_term"] = kl_term
        output["latent_tf_mse"] = latent_tf_mse
        output["pix_tf_mse"] = pix_tf_mse

        return loss, output
    else:
        return mse, output

def evaluate_policy(
    env,
    model,
    agent,
    args,
    episodes: int = 5,
    seed: int = 0,
    record_video: bool = False,
):
    rng = np.random.default_rng(seed)
    device = args.device
    world_radius = env.unwrapped.cfg.world_radius

    returns, steps_list, successes = [], [], 0
    frames = [] if record_video else None

    obs, info = env.reset(seed=rng.integers(0, 1_000_000))
    if record_video:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    for ep in range(episodes):
        if ep > 0:
            obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
            if record_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        prev_action = torch.zeros(args.num_envs, dtype=torch.long, device=device)

        with torch.no_grad():
            imgs0 = torch.from_numpy(obs).reshape(1, 1, -1).to(device).float() / world_radius
            feat0 = model.encoder(imgs0)
            z1_bel = model.latent1_first_posterior(feat0).rsample()
            z2_bel = model.latent2_first_posterior(z1_bel).rsample()

        ep_ret, steps = 0.0, 0
        while True:
            # -------- Bayes filter: PREDICT (priors) --------
            with torch.no_grad():
                a_one = F.one_hot(prev_action, num_classes=model.action_dim).float()
                p1 = model.latent1_prior(z2_bel, a_one).base_dist
                z1_prd = p1.loc
                p2 = model.latent2_prior(z1_prd, z2_bel, a_one).base_dist
                _z2_prd = p2.loc  # kept for completeness (you weren't using it)

            # -------- Bayes filter: UPDATE (posteriors) --------
            with torch.no_grad():
                imgs = torch.from_numpy(obs).reshape(1, 1, -1).to(device).float() / world_radius
                feat = model.encoder(imgs)
                q1 = model.latent1_posterior(feat, z2_bel, a_one)
                z1_t = q1.rsample()
                q2 = model.latent2_posterior(z1_t, z2_bel, a_one)
                z2_t = q2.rsample()
                z1_bel, z2_bel = z1_t, z2_t

                z_cat = torch.cat([z1_bel, z2_bel], dim=1)
                action = agent.act(z_cat).squeeze(1).to(device)

            obs, r, terminated, truncated, info = env.step(action)
            prev_action = action
            ep_ret += float(r)
            steps += 1

            if record_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if terminated or truncated:
                successes += int(terminated)
                break

        returns.append(float(ep_ret))
        steps_list.append(int(steps))

    video = None
    if record_video and frames:
        video = np.stack(frames, axis=0)  # (T,H,W,3), uint8

    return returns, steps_list, successes, video


if __name__ == '__main__':
    args = tyro.cli(Args)
    #run_name = f"{args.env_id}__{int(time.time())}"
    run_name = None
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device
    print(f"Using device: {args.device}")

    run = None
    writer = None

    if args.track:
        if args.disable_wandb_service:
            os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
        
        wandb.login()
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=None,
            save_code=True,
        )

        # Make W&B use "global_step" as the step axis (avoid wandb.log(step=...))
        run.define_metric("global_step")
        run.define_metric("train/*", step_metric="global_step")
        run.define_metric("eval/*", step_metric="global_step")

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
        )


    env = make_env(
        render_mode="rgb_array",
        world_radius=10.0,
        max_speed=0.5,
        goal_radius=1.0,
        band_center=(-9.0, 0.0),
        band_angle_deg=90.0,
        band_width=2.0,
        sigma_dark=args.sigma_dark,     
        sigma_light=args.sigma_light,
        include_goal_in_obs=True,
        randomize_start=True,
        randomize_goal=True,
        min_start_goal_dist=6.0,
        require_opposite_band_side=False,
    )

    env.unwrapped.cfg.max_steps =  args.max_episode_steps
    env = DiscreteActions(env) 
    args.obs_kind = "tabular"
    world_radius = env.unwrapped.cfg.world_radius
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    obs, info = env.reset(seed=rng.integers(0, 1_000_000))
    args.tabular_dim = obs.shape[0]

    Model = ModelDistributionNetwork(env.action_space, args)
    agent = D3QNAgent(env.action_space.n, args)
    use_kl = False
    if not args.from_scratch:
        ckpt = load_from_newest_run(Model, args.env_id, args, base_dir="checkpoints\\LightDarkNavigation",
                             filename="model_pretrained_kl_teacher.pth")
    

    rb = SequenceReplayBuffer(
        capacity   = args.buffer_size,
        obs_shape  = (1,obs.shape[0]),
        act_shape  = (),
        seq_len    = args.sequence_len,
        device     = args.device,
        obs_dtype = torch.float32)
    
    episode_first = np.ones(args.num_envs, dtype=bool)   # True right after reset

    ############################# THIS IS THE MODEL PRETRAINING ###############################
    if args.from_scratch:
        # 0. collect bootstrap data ------------------------------------------------
        print("Collecting bootstrap data for model pretraining...")
        while rb.ptr < 10_000:                 
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_type = np.where(
                        [done],               2,
                        np.where(episode_first, 0, 1)
                    ).astype(np.int64)      # shape (n_envs,)

            rb.add(obs[None,...], action, reward, done, step_type[0])
            obs = next_obs
            episode_first = done
            if done:
                obs, info = env.reset(seed=rng.integers(0, 1_000_000))
                episode_first = np.ones(args.num_envs, dtype=bool)
        
        
        # 1. model-only optimisation loop -----------------------------------------
        print("Pretraining the model...")
        reconstruction_losses_mse = []
        reconstruction_losses_kl = []
        for pretrain_step in tqdm(range(args.pretrain_steps)):
            batch = rb.sample(args.batch_size)
            obs  = (batch["obs"].float() / world_radius)    # normalize to [-1,1]
            actions = batch["action"]
            step_ty = batch["step_type"]
            if pretrain_step < int(args.pretrain_steps / 2):
                model_loss, output = compute_loss(Model, obs, actions, step_ty, step=pretrain_step, use_kl=False)
                reconstruction_losses_mse.append(model_loss.item())
            else:
                model_loss, output = compute_loss(Model, obs, actions, step_ty, step=pretrain_step, use_kl=True)
                reconstruction_losses_kl.append(model_loss.item())
            Model.optimizer.zero_grad()
            model_loss.backward()
            Model.optimizer.step()
            #reconstruction_losses.append(model_loss.item())
        if args.save_model:
            path = f"checkpoints\\LightDarkNavigation\\{run_name}\\model_pretrained_kl_teacher.pth"
            save_world_model_ckpt(Model, pretrain_step+1, path)
            pkl.dump([reconstruction_losses_mse, reconstruction_losses_kl], open(f"light_dark_model_reconstruction_losses.pkl", "wb"))
            #log_checkpoint_to_wandb(path, pretrain_step+1, run, aliases=("pretrain", "latest"))

        ######################## END OF MODEL PRETRAINING ###############################
        print("Model pretraining completed.")

    #####################################################################################

    ep_return = 0.0
    ep_len = 0
    episodes = 0

    #start the game
    # ---- reset & init belief with FIRST posteriors (uses latent1_first_posterior) ----
    obs, _ = env.reset(seed=args.seed)
    episode_first = np.ones(args.num_envs, dtype=bool)   # True right after reset
    # previous action indices per env (start with NOOP index = 0)
    prev_action = torch.zeros(args.num_envs, dtype=torch.long, device=args.device) # NOOP
    start_time = time.time()

    with torch.no_grad():
        imgs0  = torch.from_numpy(obs).reshape(1,1,-1).to(args.device).float() / world_radius
        feat0  = Model.encoder(imgs0)                           # (N, feat)
        z1_bel = Model.latent1_first_posterior(feat0).rsample() # (N, d1)
        z2_bel = Model.latent2_first_posterior(z1_bel).rsample()# (N, d2)

    for global_step in tqdm(range(args.total_timesteps)+1):
        agent.epsilon = agent.linear_schedule(args.start_e, args.end_e, int(args.exploration_fraction * args.total_timesteps), global_step)

     # -------- Bayes filter: PREDICT (use PRIORS) --------
        with torch.no_grad():
            a_one  = F.one_hot(prev_action, num_classes=Model.action_dim).float()  # (N,A)
            # p(z^1_t | z^2_{t-1}, a_{t-1})
            p1     = Model.latent1_prior(z2_bel, a_one).base_dist
            z1_prd = p1.loc  # mean prediction (lower variance than sampling)
            # p(z^2_t | z^1_t, z^2_{t-1}, a_{t-1})
            p2     = Model.latent2_prior(z1_prd, z2_bel, a_one).base_dist
            z2_prd = p2.loc
        
        # -------- Bayes filter: UPDATE (use POSTERIORS with current frame) --------
        with torch.no_grad():
            imgs = torch.from_numpy(obs).reshape(1,1,-1).to(device).float() / world_radius
            feat = Model.encoder(imgs)  # (N, feat)
            # q(z^1_t | x_t, z^2_{t-1}, a_{t-1})
            q1   = Model.latent1_posterior(feat, z2_bel, a_one)
            z1_t = q1.rsample()
            # q(z^2_t | z^1_t, z^2_{t-1}, a_{t-1})
            q2   = Model.latent2_posterior(z1_t, z2_bel, a_one)
            z2_t = q2.rsample()

            z1_bel, z2_bel = z1_t, z2_t
        
        if random.random() < agent.epsilon:     
            action = torch.as_tensor([env.action_space.sample() for _ in range(args.num_envs)], device=device, dtype=torch.long) 
        else:
            z_cat = torch.cat([z1_bel, z2_bel], dim=1)     # (N, d1+d2)
            action = agent.act(z_cat).squeeze(1).to(device) # (N,)
        
        next_obs, reward, termination, truncation, info = env.step(action)
        done = termination | truncation

        # remember actions for next predict/update
        prev_action = action
        
        step_type = np.where(done, 2, np.where(episode_first, 0, 1)).astype(np.int64)  # shape (n_envs,)

        rb.add(obs[None,...], action, reward, done, step_type[0])

        ep_return += float(reward)
        ep_len += 1

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook


        # -------- RE-INIT belief on resets (uses latent1_first_posterior again) --------
        if done == True:
            obs, info = env.reset(seed=rng.integers(0, 1_000_000))
            episode_first = np.ones(args.num_envs, dtype=bool) 
            prev_action = torch.zeros(args.num_envs, dtype=torch.long, device=args.device) # NOOP
            with torch.no_grad():
                imgs0  = torch.from_numpy(obs).reshape(1,1,-1).to(args.device).float() / world_radius
                feat0  = Model.encoder(imgs0)                           # (N, feat)
                z1_bel = Model.latent1_first_posterior(feat0).rsample() # (N, d1)
                z2_bel = Model.latent2_first_posterior(z1_bel).rsample()# (N, d2)
            
            episodes += 1
            if writer is not None:
                writer.add_scalar("train/episode_return", ep_return, global_step)
                writer.add_scalar("train/episode_length", ep_len, global_step)
                writer.add_scalar("train/episodes", episodes, global_step)
            ep_return = 0.0
            ep_len = 0
        else:
            obs = next_obs
            episode_first = [done]


        if global_step > args.learning_starts and global_step % args.train_frequency == 0:            
            #with tic("sample"):
            data = rb.sample(args.batch_size)
            images  = data["obs"].to(dtype=torch.float32).div_(world_radius)
            actions = data["action"]
            step_ty = data["step_type"]
            rewards = data["reward"]
            dones   = data["done"]

            model_loss, output = compute_loss(Model, images, actions, step_ty, use_kl=True)
            Model.optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 20.0)
            Model.optimizer.step()
            
            with torch.no_grad():
                # Posterior latents & posterior distributions
                (z1, z2), (q_z1, q_z2) = Model.sample_posterior(images, actions, step_ty)

                # Optional MI bonus (aligned with transitions t=0..S-2, attached to r_{t+1})
                if args.mi_use and args.mi_beta > 0.0:
                    mi_bonus = compute_mi_bonus(
                        agent=agent,
                        q_z1=q_z1,
                        q_z2=q_z2,
                        actions=actions,
                        step_types=step_ty,
                        mi_num_samples=args.mi_num_samples,
                    )  # (B, S-1)

                    rewards_total = rewards.clone()
                    rewards_total[:, 1:] = rewards_total[:, 1:] + args.mi_beta * mi_bonus
                else:
                    rewards_total = rewards

            q_loss, q_pred, target_q = agent.compute_loss(z1, z2, actions, rewards_total, dones)

            agent.update(q_loss)

            if global_step % args.target_network_frequency == 0:
                agent.update_target_model()
            
            if global_step % args.video_every == 0 and global_step > 0:
                returns, steps_list, successes, video = evaluate_policy(
                    env, Model, agent, args,
                    episodes=5,
                    seed=args.seed,
                    record_video=True,
                )
                ret_mean = float(np.mean(returns)) if len(returns) else 0.0
                len_mean = float(np.mean(steps_list)) if len(steps_list) else 0.0
                succ_rate = float(successes)

                if writer is not None:
                    writer.add_scalar("eval_video/return_mean", ret_mean, global_step)
                    writer.add_scalar("eval_video/ep_len_mean", len_mean, global_step)
                    writer.add_scalar("eval_video/success_rate", succ_rate, global_step)

                if run is not None and video is not None:
                    log_video_to_wandb(video, fps=args.video_fps, key="eval/video", step=global_step, run=run)

                tqdm.write(f"[{global_step}] Eval(video) | Return {ret_mean:.3f} | Steps {len_mean:.1f} | Success {succ_rate:.2f}")


            if global_step % args.eval_every == 0 and global_step > 0:
                returns, steps_list, successes, _ = evaluate_policy(
                    env, Model, agent, args,
                    episodes=5,
                    seed=args.seed,
                    record_video=False,
                )
                ret_mean = float(np.mean(returns)) if len(returns) else 0.0
                len_mean = float(np.mean(steps_list)) if len(steps_list) else 0.0
                succ_rate = float(successes)

                if writer is not None:
                    writer.add_scalar("eval/return_mean", ret_mean, global_step)
                    writer.add_scalar("eval/ep_len_mean", len_mean, global_step)
                    writer.add_scalar("eval/success_rate", succ_rate, global_step)

                tqdm.write(f"[{global_step}] Eval | Return {ret_mean:.3f} | Steps {len_mean:.1f} | Success {succ_rate:.2f}")
   
            if global_step % 500 == 0 and writer is not None:
                writer.add_scalar("train/epsilon", agent.epsilon, global_step)
                writer.add_scalar("train/q_loss", float(q_loss.item()), global_step)
                writer.add_scalar("train/q_pred_mean", float(q_pred.mean().item()), global_step)
    
    if writer is not None:
        writer.close()
    if run is not None:
        run.finish()

