import os

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.spaces import Box, Discrete
import ale_py
import numpy as np
#import matplotlib.pyplot as plt
import random
import time
from dataclasses import dataclass
import tyro
#import cv2
import copy
from tqdm import tqdm
from contextlib import contextmanager
import collections, math

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.distributions import kl_divergence

from SequenceReplayBuffer import SequenceReplayBuffer

from huggingface_hub import hf_hub_download

import matplotlib.pyplot as plt

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from SLAC_Agent_deterministic_1 import ModelDistributionNetwork



script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_model: bool = True
    """if toggled, the trained model will be saved to disk"""
    from_scratch: bool = False
    """if toggled, the model will be trained from scratch"""
    ckpt_path = "checkpoints//ALE//Pong-v5__SLAC_PONG_deterministic__1__941_full_pretrain_checkpoint//model_pretrained_kl_teacher.pth"
    """If not from scratch, path to the pretrained model"""
    wandb_project_name: str = "SLAC_PONG"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "ALE/Pong-v5"
    """the id of the environment"""
    num_envs: int = 1
    """the number of parallel game environments"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    q_learning_rate: float = 3e-4
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
    target_network_frequency: int = 10000
    """the frequency of target network update"""
    tau: float = 0.005
    """the polyak averaging factor for target network update"""
    sequence_len : int = 8
    """the length of the sequence for training"""
    buffer_size: int = 100_000
    """the replay memory buffer size"""
    kl_analytic: bool = True
    """if toggled, the KL divergence will be computed analytically"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    base_depth: int = 32
    """the base depth of the model network"""
    latent1_size: int = 32
    """the size of the first latent variable"""
    latent2_size: int = 256
    """the size of the second latent variable"""
    hidden_dims: tuple = (256, 256)
    """the hidden dimensions of the Q-network"""


def make_env(env_id, seed, idx, capture_video=False, run_name=""):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", frameskip=1, full_action_space=False)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, frameskip=1, full_action_space=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = RallyDoneWrapper(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        env = gym.wrappers.GrayScaleObservation(env)


        env.action_space.seed(seed)
        return env

    return thunk

class RallyDoneWrapper(gym.Wrapper):
    """End episode after each rally (when a point is scored)."""
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """Modify step function to end the episode after each rally."""
        obs, reward, done, truncated, info = self.env.step(action)

        # End the episode when a point is scored (reward ≠ 0)
        if reward != 0:
            done = True  # Force episode to end

        return obs, reward, done, truncated, info


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
    
    """@torch.no_grad()
    def act(self, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            q = self.q_net(z).float()                           # (B, A)
            alpha = self.log_alpha.exp().float().clamp_min(1e-4)

            # numerically stable: subtract rowwise max before softmax
            logits = (q - q.max(dim=1, keepdim=True).values) / alpha
            pi = torch.softmax(logits, dim=1)                   # (B, A)

            if deterministic:
                a = pi.argmax(dim=1, keepdim=True)              # greedy
            else:
                a = torch.multinomial(pi, num_samples=1)        # stochastic

        return a    # int64 indices (B, 1)"""
    
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

        ############# PUT THIS BACK IN IF USING SOFT Q-LEARNING #############
        """with torch.no_grad():
            alpha  = self.log_alpha.exp()
            q_tp1  = self.q_target_net(z_tp1_f)                 # (BT, A)
            soft   = torch.logsumexp(q_tp1 / alpha, dim=-1)     # (BT,)
            y_main = r_t.reshape(BT) + self.gamma * (1 - d_t.reshape(BT)) * (alpha * soft)"""
        #######################################################################

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

        """with torch.no_grad():
            q_all_t = self.q_net(z_t_f)   # (BT, A)
            td_err_taken = torch.zeros_like(y_main)
            # This is the TD error for the action actually taken (what the loss uses)
            td_err_taken = (q_all_t.gather(1, a_t_f.unsqueeze(-1)).squeeze(-1) - y_main)

            mean_abs = []
            counts = []
            for a in range(self.n_actions):
                mask = (a_t_f == a)
                counts.append(mask.sum().item())
                if mask.any():
                    q_a  = q_all_t[mask, a]
                    y_a  = y_main[mask]          # IMPORTANT: target must be matched to the taken action only
                    td_a = (q_a - y_a).abs().mean().item()
                else:
                    td_a = float('nan')
                mean_abs.append(td_a)
            print(f"minibatch action counts: {counts}")
            print(f"mean|td_err| (per taken action): {mean_abs}")"""

        # actions must be 0..A-1
        assert a_t_f.min().item() >= 0 and a_t_f.max().item() < self.n_actions, \
            f"Bad action indices in buffer: [{a_t_f.min().item()}, {a_t_f.max().item()}]"

        # targets must be 1D and match BT
        assert y_main.ndim == 1 and y_main.shape[0] == z_t_f.shape[0]
        
        q_loss = 0.5 * F.mse_loss(q_main, y_main, reduction="mean")

        """# --- terminal last step: include only if this slice ends the episode
        last_done = (done_all[:, -1] > 0.5)
        print(last_done.shape)
        if last_done.any():
            idx    = last_done.nonzero(as_tuple=False).squeeze(-1)
            print(idx.shape)
            print(idx)
            z_last = z_all[idx, -1]                              # (B_term, d)
            print("z_last:", z_last.shape)
            a_last = a_all[idx, -1]                              # (B_term,)
            print("a_last:", a_last.shape)
            r_last = r_all[idx, -1]                              # (B_term,)
            print("r_last:", r_last.shape)
            q_last = self.q_net(z_last).gather(1, a_last.unsqueeze(-1)).squeeze(-1)
            print("q_last:", q_last.shape)
            bla
            q_loss = q_loss + 0.5 * F.mse_loss(q_last, r_last, reduction="mean")"""

        ############# PUT THIS BACK IN IF USING SOFT Q-LEARNING #############
        # --- alpha loss on the regular transitions
        """with torch.no_grad():
            q_all   = self.q_net(z_t_f)
            logits  = q_all / alpha
            log_pi  = F.log_softmax(logits, dim=-1)
            pi      = log_pi.exp()
            entropy = -(pi * log_pi).sum(-1)

        alpha_loss = (self.log_alpha.exp() * (entropy - self.target_entropy)).mean()"""
        #######################################################################

        #return q_loss, alpha_loss, q_main
        return q_loss, q_main


    def update(self, q_loss):
        self.q_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_opt.step()

        ############# PUT THIS BACK IN IF USING SOFT Q-LEARNING #############
        """self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.log_alpha.data.clamp_(min=self.min_log_alpha)"""
        #######################################################################

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

    def get_head_grads(self, q_loss):
        last = [m for m in self.q_net.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            # grads for each action head (row-wise)
            row_grad = last.weight.grad.norm(dim=1).cpu().numpy()   # shape (A,)
        return row_grad
        

    
    def get_grad_norm(self):
        total_norm = 0
        for p in self.q_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

def inspect_terminal_sequences(rb, num_sequences=20):
    """
    Sample `num_sequences` batches of size 1 from the buffer and
    print those that contain a done=True flag.
    """
    seq_len = rb.seq_len
    found = 0
    trial = 0

    while found < num_sequences and trial < 1000:
        trial += 1
        batch = rb.sample(1)                 # batch_size = 1
        r   = batch["reward"].cpu().numpy().squeeze()      # (S,)
        d   = batch["done"].cpu().numpy().squeeze()        # (S,)
        stp = batch["step_type"].cpu().numpy().squeeze()   # (S,)

        if d.any():                           # contains a terminal state?
            found += 1
            term_idx = np.where(d)[0][0]      # first occurrence
            print(f"\n=== Sequence {found}  "
                  f"(terminal at position {term_idx}) ===")
            print("idx | step_type | reward | done")
            print("--------------------------------")
            for i in range(seq_len):
                print(f"{i:3d} |     {stp[i]}     |  {r[i]:5.1f} |  {d[i]}")
            # simple assertions
            assert (d[:seq_len-1] == 0).all(),   "done in middle of slice!"
            if term_idx > 0:
                print("-- terminal reward is r[{}] = {:.1f}".format(
                      term_idx-1, r[term_idx-1]))
            else:
                print("-- episode ends exactly at first index (rare)")

    if found == 0:
        print("No terminal states found in 1000 samples – "
              "replay buffer may still be small.")

timings = collections.defaultdict(float)     # global dict

@contextmanager
def tic(name):
    t0 = time.perf_counter()
    yield
    timings[name] += time.perf_counter() - t0

def register_nan_hooks(module, name=""):
    for n, p in module.named_parameters():
        full_name = f"{name}.{n}" if name else n
        p.register_hook(
            lambda grad, n=full_name: (
                print(f"⚠️ NaN in grad of {n}") if torch.isnan(grad).any() else None
            )
        )

def visualize_rollout(images, title_prefix="Frame", cmap='gray'):
    """
    Visualize a sequence of grayscale images with shape (T, 1, H, W)
    
    :param images: Tensor or ndarray of shape (T, 1, H, W)
    :param title_prefix: Prefix for subplot titles
    :param cmap: Color map to use (default 'gray')
    """
    T = images.shape[0]
    plt.figure(figsize=(T * 2, 2))

    for t in range(T):
        img = images[t, 0].cpu().numpy()  # shape (H, W)
        plt.subplot(1, T, t + 1)
        plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"{title_prefix} {t}")

    plt.tight_layout()
    plt.show()

def log_rollout_grid(images, step, caption="Rollout"):
    """
    Log a rollout as a single grid image to Weights & Biases.
    
    :param images: Tensor of shape (T, 1, H, W)
    :param step: Global step or env step
    :param caption: Caption for the image
    """
    # Normalize and repeat channel to get (T, 3, H, W) for W&B
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)  # grayscale → RGB

    grid = vutils.make_grid(images, nrow=images.shape[0], pad_value=1)

    wandb.log({caption: wandb.Image(grid, caption=caption)}, step=step)

def save_world_model_ckpt(model, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
    }
    torch.save(ckpt, path)
    print(f"[✓] Saved world model checkpoint at {path}")

def log_checkpoint_to_wandb(path, step, run, *, aliases=("latest",), name="slac_world_model"):
    art = wandb.Artifact(name=name, type="model", metadata={"step": step})
    art.add_file(path, name=os.path.basename(path))             # optional 'name=' keeps a clean filename
    run.log_artifact(art, aliases=list(aliases))
    print(f"Uploaded {path} to W&B as artifact '{name}' with aliases {aliases}")

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

def compute_loss(model, images, actions, step_types, step=None, rewards=None, discounts=None, latent_posterior_samples_and_dists=None, use_kl=False,  rollout_K=3):
    #If not provided, sample the latent variables and distributions from the encoder (inference model) conditioned on the current sequence.
    if latent_posterior_samples_and_dists is None:
        latent_posterior_samples_and_dists = model.sample_posterior(images, actions, step_types) # q(z1_0 | x0)  , q(z2_0 | z1_0), q(z1_t | x_t, z2_{t-1}, a_{t-1}), q(z2_t | z1_t, z2_{t-1}, a_{t-1})
        
    #Latent variables and their corresponding distributions for both z1 and z2.
    (z1_post, z2_post), (q_z1, q_z2) = latent_posterior_samples_and_dists
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

        # tiny L2 on prior means to prevent blow-ups
        #prior_l2 = (mu1.pow(2).mean() + mu2.pow(2).mean())

        # ----- 3) Pure predicted K-step rollout (closed-loop, no mixing) ----------
        """lat_roll_mse = images.new_tensor(0.0)  # these will sum per-step errors, then we’ll average by rollout_K
        pix_roll_mse = images.new_tensor(0.0)

        B, S = images.shape[:2]
        T = S - 1

        if rollout_K > 0 and T >= rollout_K: # Only do a K-step unroll if the sequence has enough transitions (T = S-1) and the user asked for it.
            t0 = T // 2
            # Seed from posterior at t0 (just to initialize); after that, predictions only
            z2_t = z2_det[:, t0]   # (B,·)
            z1_t = z1_det[:, t0]

            for k in range(rollout_K):
                a_t  = actions[:, t0 + k]                         # (B,)
                a_in = Model._one_hot(a_t)      # (B, A)


                # Use predicted latent as input 
                z2_in = z2_t

                 # prior step: z1_{t+1} then z2_{t+1} (use means for stability)
                d1 = model.latent1_prior(z2_in, a_in).base_dist
                z1_tp1 = d1.loc
                d2 = model.latent2_prior(z1_tp1, z2_in, a_in).base_dist
                z2_tp1 = d2.loc

                # compare to GT (posterior) at t0+k+1
                # Latent K-step teacher forcing: compare  predicted latents to the posterior latents at the next timestep. We detach those targets earlier so this term updates the priors (and not the encoder).
                z1_gt = z1_det[:, t0 + k + 1]
                z2_gt = z2_det[:, t0 + k + 1]
                #lat_roll_mse = lat_roll_mse + ((z1_tp1 - z1_gt)**2).mean() + ((z2_tp1 - z2_gt)**2).mean()
                lat_roll_mse = lat_roll_mse + huber(z1_tp1 - z1_gt).mean(dim=-1).mean() \
                                + huber(z2_tp1 - z2_gt).mean(dim=-1).mean()

                # Pixel K-step teacher forcing: decode the predicted latents and match the actual next frame. This couples the priors with the decoder so the composed rollout produces the right pixels.
                with FreezeParams(model.decoder.parameters()):
                    x_pred = model.decoder(z1_tp1, z2_tp1)  # grads flow to z1_tp1/z2_tp1, not decoder weights
                pix_roll_mse = pix_roll_mse + ((images[:, t0 + k + 1] - x_pred)**2).mean()

                # advance using predictions (closed-loop)
                z1_t, z2_t = z1_tp1, z2_tp1

            lat_roll_mse = lat_roll_mse / rollout_K
            pix_roll_mse = pix_roll_mse / rollout_K

        w_lat     = 1.0
        w_pix     = 0.1
        w_lat_roll= 0.5
        w_pix_roll= 0.1
        #w_l2_prior = 1e-6

        loss = mse + kl_term + w_lat * latent_tf_mse + w_pix * pix_tf_mse + w_lat_roll * lat_roll_mse + w_pix_roll * pix_roll_mse """
        loss = mse + kl_term + 0.1*latent_tf_mse + pix_tf_mse

        output["kl_z1"] = kl_z1.mean()
        output["kl_q_raw"] = kl_q_raw.mean()
        output["kl_q"] = kl_q
        output["kl_term"] = kl_term
        output["latent_tf_mse"] = latent_tf_mse
        output["pix_tf_mse"] = pix_tf_mse
        #output["lat_roll_mse"] = lat_roll_mse
        #output["pix_roll_mse"] = pix_roll_mse

        return loss, output
    else:
        return mse, output


if __name__ == '__main__':
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.login()
        run = wandb.init(
            project=args.wandb_project_name,
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=False,
        )

        """if args.from_scratch:
            artifact = wandb.Artifact("source_code", type="code")
            run.log_artifact(artifact)
        else:"""
        run.log_code(root=script_dir)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    # env setup
    #envs = gym.vector.SyncVectorEnv(
    #    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    #)

    envs = SyncVectorEnv([make_env(args.env_id, args.seed + i, i,
              args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    ACTION_MAPPING = {0: 0, 1: 2, 2: 3}
    n_actions = len(ACTION_MAPPING.keys())
    action_space = Discrete(n_actions)
    #action_dim = int(np.prod(action_space.shape))

    """obs,_ = envs.reset()
    done = False
    total_reward = 0.0

    for i in range(1500):
        action = np.array([action_space.sample() for _ in range(envs.num_envs)])
        real_action = np.array([ACTION_MAPPING[a.item()] for a in action])
        obs, reward, done, truncation, infos = envs.step(real_action)
        total_reward += reward
        if done:
            print(f"Episode Reward: {total_reward}")
            plt.imshow(obs[0], cmap='gray')
            plt.axis('off')
            plt.show()
            done = False
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(infos)
                        print(f"episodic_return={info['episode']['r']}")
                        bla
        last_obs = obs
    
    bla"""

    obs,_ = envs.reset(seed=args.seed)


    Model = ModelDistributionNetwork(action_space, args)
    agent = Qagent(action_space.n, args)
    use_kl = False
    if not args.from_scratch:
        latent_dim = args.latent1_size + args.latent2_size
        if Model.decoder.deconv1.weight.shape[0] != latent_dim:
            Model.decoder.deconv1 = nn.ConvTranspose2d(latent_dim, 8*args.base_depth, kernel_size=4, stride=1, padding=0).to(args.device)
            Model.decoder.deconv1_initialized = True
            use_kl=True
        ckpt = torch.load(args.ckpt_path, map_location=args.device)
        Model.load_state_dict(ckpt["model_state"])

    eval_counter = 0
    #scaler = torch.amp.GradScaler("cuda")

    rb = SequenceReplayBuffer(
        capacity   = args.buffer_size,
        obs_shape  = (1,obs.shape[1],obs.shape[2]),
        act_shape  = (),
        seq_len    = args.sequence_len,
        device     = args.device,)

    #torch.autograd.set_detect_anomaly(True)
    #register_nan_hooks(Model.latent1_posterior, "latent1_post")

    #if args.from_scratch:
        #start the game
    obs, _ = envs.reset(seed=args.seed)
    episode_first = np.ones(args.num_envs, dtype=bool)   # True right after reset
        
    ############################# THIS IS THE MODEL PRETRAINING ###############################
    if args.from_scratch:
        # 0. collect bootstrap data ------------------------------------------------
        print("Collecting bootstrap data for model pretraining...")
        while rb.ptr < 10_000:                 # or 10 episodes for DM-Control
            actions = np.array([action_space.sample() for _ in range(envs.num_envs)])
            real_actions = np.array([ACTION_MAPPING[a.item()] for a in actions])
            #with tic("env.step"):
            next_obs, rewards, terminations, truncations, infos = envs.step(real_actions)
            done = terminations | truncations 
            step_type = np.where(
                        done,               2,
                        np.where(episode_first, 0, 1)
                    ).astype(np.int64)      # shape (n_envs,)
                
            for k in range(args.num_envs):
                    rb.add(obs[k][None],
                    actions[k],
                    rewards[k],
                    done[k],
                    step_type[k])      
            obs = next_obs
            episode_first = done.copy() 
            
        # 1. model-only optimisation loop -----------------------------------------
        print("Pretraining the model...")
        for pretrain_step in tqdm(range(100_000)):
            batch = rb.sample(args.batch_size)
            images  = (batch["obs"].float() / 255.)
                #print(images.dtype, images.min().item(), images.max().item())
            actions = batch["action"]
            step_ty = batch["step_type"]

            #model_loss, _, output = Model.compute_loss(images, actions, step_ty, step=pretrain_step)
            #model_loss, output = Model.compute_loss(images, actions, step_ty, step=pretrain_step)
            model_loss, output = compute_loss(Model, images, actions, step_ty, step=pretrain_step, use_kl=use_kl)
            Model.optimizer.zero_grad()
            model_loss.backward()

            Model.optimizer.step()

            if args.track:
                if (pretrain_step) % 10_000 == 0:
                    sequence = rb.sample(1)
                    #image = sequence["obs"][0][0].unsqueeze(0)  # Get the first image in the sequence
                    #rollout_images = Model.model_rollout(image, H=args.sequence_len-1)  # Generate a sequence of images
                    #log_rollout_grid(rollout_images, step=0, caption="Model Rollout")

                    #batch = rb.sample(args.batch_size)
                    images  = (sequence["obs"].float() / 255.)
                    actions = sequence["action"]
                    step_ty = sequence["step_type"]
                    #recon, img_cond, img_u = Model.visual_diagnostics(images[0].unsqueeze(0), actions[0].unsqueeze(0), step_ty)
                    #recon, img_cond, img_u = Model.visual_diagnostics(images[0].unsqueeze(0), actions[0].unsqueeze(0), step_ty)
                    #recon = Model.visual_diagnostics(images[0].unsqueeze(0), actions[0].unsqueeze(0), step_ty)
                    log_rollout_grid(images[0], step=0, caption="Sequence")
                    x_pred, aux = Model.one_step_prior_predict(images, actions, step_ty)
                    log_rollout_grid(x_pred[0], step=0, caption="One-step Prior Predict")
                    recon = Model.visual_diagnostics(images, actions, step_ty)
                    log_rollout_grid(recon, step=0, caption="Recon")
                    
                    preds_imgs, fd_pred, fd_true, mask = Model.build_motion_mask(images, actions, step_ty)
                    #log_rollout_grid(fd_pred.squeeze(0), step=0, caption="Frame Difference Prediction")
                    #log_rollout_grid(fd_true.squeeze(0), step=0, caption="Frame Difference Ground Truth")
                    #log_rollout_grid(mask.squeeze(0), step=0, caption="Motion Mask")
                    log_rollout_grid(preds_imgs.squeeze(0), step=0, caption="Predicted Images")

                    """log_rollout_grid(img_cond, step=0, caption="Image Condition")
                        log_rollout_grid(img_u, step=0, caption="Image U")"""
                if (pretrain_step) % 500 == 0:
                    writer.add_scalar("losses/model_loss", model_loss.item(), pretrain_step)
                    writer.add_scalar("losses/mse", output["mse"].item(), pretrain_step)
                    if use_kl:
                        writer.add_scalar("losses/kl_z1", output["kl_z1"].item(), pretrain_step)
                        writer.add_scalar("losses/kl_q_raw", output["kl_q_raw"].item(), pretrain_step)
                        writer.add_scalar("losses/kl_q", output["kl_q"].item(), pretrain_step)
                        writer.add_scalar("losses/kl_term", output["kl_term"].item(), pretrain_step)
                        writer.add_scalar("losses/latent_tf_mse", output["latent_tf_mse"].item(), pretrain_step)
                        writer.add_scalar("losses/pix_tf_mse", output["pix_tf_mse"].item(), pretrain_step)
                        #writer.add_scalar("losses/lat_roll_mse", output["lat_roll_mse"].item(), pretrain_step)
                        #writer.add_scalar("losses/pix_roll_mse", output["pix_roll_mse"].item(), pretrain_step)

        if args.save_model:
            path = f"checkpoints\\{run_name}\\model_pretrained_kl_teacher.pth"
            save_world_model_ckpt(Model, pretrain_step+1, path)
            log_checkpoint_to_wandb(path, pretrain_step+1, run, aliases=("pretrain", "latest"))

        ######################## END OF MODEL PRETRAINING ###############################
        print("Model pretraining completed.")


    #start the game
    # ---- reset & init belief with FIRST posteriors (uses latent1_first_posterior) ----
    obs, _ = envs.reset(seed=args.seed)
    episode_first = np.ones(args.num_envs, dtype=bool)   # True right after reset
    # previous action indices per env (start with NOOP index = 0)
    prev_actions = torch.zeros(args.num_envs, dtype=torch.long, device=args.device) # NOOP
    start_time = time.time()

    with torch.no_grad():
        imgs0  = torch.from_numpy(obs).unsqueeze(1).to(args.device).float() / 255.0
        feat0  = Model.encoder(imgs0)                         # (N, feat)
        z1_bel = Model.latent1_first_posterior(feat0).rsample() # (N, d1)
        z2_bel = Model.latent2_first_posterior(z1_bel).rsample()# (N, d2)

    for global_step in range(args.total_timesteps):
        agent.epsilon = agent.linear_schedule(args.start_e, args.end_e, int(args.exploration_fraction * args.total_timesteps), global_step)

     # -------- Bayes filter: PREDICT (use PRIORS) --------
        with torch.no_grad():
            a_one  = F.one_hot(prev_actions, num_classes=Model.action_dim).float()  # (N,A)
            # p(z^1_t | z^2_{t-1}, a_{t-1})
            p1     = Model.latent1_prior(z2_bel, a_one).base_dist
            z1_prd = p1.loc  # mean prediction (lower variance than sampling)
            # p(z^2_t | z^1_t, z^2_{t-1}, a_{t-1})
            p2     = Model.latent2_prior(z1_prd, z2_bel, a_one).base_dist
            z2_prd = p2.loc
            
        # -------- Bayes filter: UPDATE (use POSTERIORS with current frame) --------
        with torch.no_grad():
            imgs = torch.from_numpy(obs).unsqueeze(1).to(device).float() / 255.0
            feat = Model.encoder(imgs)  # (N, feat)
            # q(z^1_t | x_t, z^2_{t-1}, a_{t-1})
            q1   = Model.latent1_posterior(feat, z2_bel, a_one)
            z1_t = q1.rsample()
            # q(z^2_t | z^1_t, z^2_{t-1}, a_{t-1})
            q2   = Model.latent2_posterior(z1_t, z2_bel, a_one)
            z2_t = q2.rsample()

            z1_bel, z2_bel = z1_t, z2_t

        if random.random() < agent.epsilon:     
            actions = torch.as_tensor([action_space.sample() for _ in range(args.num_envs)], device=device, dtype=torch.long)  # (N,)
        else:
            z_cat = torch.cat([z1_bel, z2_bel], dim=1)     # (N, d1+d2)
            actions = agent.act(z_cat).squeeze(1).to(device) # (N,)
            # map to env actions
        
        a_np  = actions.detach().cpu().numpy()
        real_actions = np.array([ACTION_MAPPING[int(a)] for a in a_np], dtype=np.int64)

        #execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(real_actions)
        done = terminations | truncations 

        # remember actions for next predict/update
        prev_actions = actions

        # -------- RE-INIT belief on resets (uses latent1_first_posterior again) --------
        if done.any():
            ids = np.nonzero(done)[0]
            ids_t = torch.from_numpy(ids).to(device)
            with torch.no_grad():
                imgs_r  = torch.from_numpy(next_obs[ids]).unsqueeze(1).to(device).float() / 255.0
                feat_r  = Model.encoder(imgs_r)
                z1_0    = Model.latent1_first_posterior(feat_r).rsample()
                z2_0    = Model.latent2_first_posterior(z1_0).rsample()
                z1_bel[ids_t] = z1_0
                z2_bel[ids_t] = z2_0
                prev_actions[ids_t] = 0  # NOOP

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        step_type = np.where(done, 2, np.where(episode_first, 0, 1)).astype(np.int64)  # shape (n_envs,)

        #with tic("store_buffer"):
        for k in range(args.num_envs):
            rb.add(obs[k][None],
            actions[k].item(),
            rewards[k],
            done[k],
            step_type[k])      

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        episode_first = done.copy() 
        

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:            
            #with tic("sample"):
            data = rb.sample(args.batch_size)
            images  = data["obs"].to(dtype=torch.float32).div_(255.)
            actions = data["action"]
            step_ty = data["step_type"]
            rewards = data["reward"]
            dones   = data["done"]

            #with tic("model_forward"):
            #with torch.amp.autocast("cuda"):
            #model_loss, _ = Model.compute_loss(images,actions,step_ty)
            model_loss, output = compute_loss(Model, images, actions, step_ty, use_kl=True)
            #with tic("model_update"):
            #Model.update(model_loss)
            Model.optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 20.0)
            Model.optimizer.step()
                #with tic("sample_posterior"):
            with torch.no_grad():
                (z1, z2), _ = Model.sample_posterior(images, actions, step_ty)

                #with tic("q_forward"):
            #with torch.amp.autocast("cuda"):
            #q_loss, alpha_loss, q_pred = agent.compute_loss(z1, z2, actions, rewards, dones) 
            q_loss, q_pred = agent.compute_loss(z1, z2, actions, rewards, dones)
            #with tic("agent_update"):
            #agent.update(q_loss, alpha_loss)
            agent.update(q_loss)
            
            #scaler.update() 

            """total_loss = model_loss + q_loss + alpha_loss

            Model.optimizer.zero_grad()
            agent.q_opt.zero_grad()
            agent.alpha_opt.zero_grad()

            total_loss.backward()

            Model.optimizer.step()     
            agent.q_opt.step()              
            agent.alpha_opt.step()"""

            """if global_step % 2000 == 0 :
                tot = sum(timings.values())
                print(f"\nTiming breakdown after {global_step:,} env-steps")
                for k, v in sorted(timings.items(), key=lambda x: -x[1]):
                    print(f"  {k:15s}: {v:7.3f}s  {100*v/tot:5.1f}%")
                timings.clear()"""

            if global_step % 500 == 0:
                writer.add_scalar("losses/model_loss", model_loss.item(), global_step)
                writer.add_scalar("losses/q_loss", q_loss.item(), global_step)
                writer.add_scalar("losses/q_values", q_pred.mean().item(), global_step)
                #writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.add_scalar("losses/log_alpha", agent.log_alpha.item(), global_step)
                writer.add_scalar("losses/mse", output["mse"].item(), global_step)
                writer.add_scalar("losses/kl_z1", output["kl_z1"].item(), global_step)
                writer.add_scalar("losses/kl_q_raw", output["kl_q_raw"].item(), global_step)
                writer.add_scalar("losses/kl_q", output["kl_q"].item(), global_step)
                writer.add_scalar("losses/kl_term", output["kl_term"].item(), global_step)
                writer.add_scalar("losses/latent_tf_mse", output["latent_tf_mse"].item(), global_step )
                writer.add_scalar("losses/pix_tf_mse", output["pix_tf_mse"].item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/epsilon", agent.epsilon, global_step)

            if global_step % 5000 == 0:
                model_grad_norm = Model.get_grad_norm()
                agent_grad_norm = agent.get_grad_norm()
                writer.add_scalar("charts/grad_norm", agent_grad_norm, global_step)
                writer.add_scalar("charts/model_grad_norm", model_grad_norm, global_step)
                td_error = agent.get_td_error(z1, z2, actions, rewards, dones)
                writer.add_scalar("charts/td_error_head_0", td_error[0], global_step)
                writer.add_scalar("charts/td_error_head_1", td_error[1], global_step)
                writer.add_scalar("charts/td_error_head_2", td_error[2], global_step)
                agent_head_grads = agent.get_head_grads(q_loss)
                writer.add_scalar("charts/agent_head_0_grads", agent_head_grads[0], global_step)
                writer.add_scalar("charts/agent_head_1_grads", agent_head_grads[1], global_step)
                writer.add_scalar("charts/agent_head_2_grads", agent_head_grads[2], global_step)
            
             # update target network
            if global_step % args.target_network_frequency == 0:
                # -------------------------------------------------------------------------
                # 4. Target-net update (polyak)
                # -------------------------------------------------------------------------
                """with torch.no_grad():
                    for p, p_targ in zip(agent.q_net.parameters(), agent.q_target_net.parameters()):
                        p_targ.data.mul_(1.0 - args.tau).add_(args.tau * p.data)"""
                agent.update_target_model()

            
            if global_step % 100_000 == 0 and global_step > 0:
                sequence = rb.sample(1)
                #image = sequence["obs"][0][0].unsqueeze(0)  # Get the first image in the sequence
                #rollout_images = Model.model_rollout(image, H=args.sequence_len-1)  # Generate a sequence of images
                #log_rollout_grid(rollout_images, step=0, caption="Model Rollout")
                
                images  = (sequence["obs"].float() / 255.)
                actions = sequence["action"]
                step_ty = sequence["step_type"]
                #recon = Model.visual_diagnostics(images, actions, step_ty)
                #log_rollout_grid(recon, step=0, caption="Recon")
                log_rollout_grid(images[0], step=0, caption="Sequence")
                x_pred, aux = Model.one_step_prior_predict(images, actions, step_ty)
                log_rollout_grid(x_pred[0].detach().cpu(), step=0, caption="One-step Prior Predict")
                preds_imgs, fd_pred, fd_true, mask = Model.build_motion_mask(images, actions, step_ty)
                log_rollout_grid(preds_imgs.squeeze(0).detach().cpu(), step=0, caption="Predicted Images")

            
            if global_step % 1_000_000 == 0 and global_step > 0:
                if args.save_model:
                    save_dir   = f"runs/{run_name}"
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = f"{save_dir}/{args.exp_name}.pt"

                    ckpt = {
                        "global_step"   : global_step,

                        # ---------- generative model ----------------------------------
                        "model_state"   : Model.state_dict(),
                        "model_opt"     : Model.optimizer.state_dict(),

                        # ---------- Q-agent ------------------------------------------
                        "q_state"       : agent.q_net.state_dict(),
                        "q_target"      : agent.q_target_net.state_dict(),
                        "q_opt"         : agent.q_opt.state_dict(),

                        # temperature parameter α
                        "log_alpha"     : agent.log_alpha.detach().cpu(),
                        "alpha_opt"     : agent.alpha_opt.state_dict(),

                        # ---------- config -------------------------------------------
                        "args"          : vars(args),
                        }
                    
                    torch.save(ckpt, model_path)
                    print(f"Model saved to {model_path}")

                    # Upload the model to wandb
                    if args.track:
                        artifact = wandb.Artifact(f"model-{int(global_step/1_000_000)}M", type="model")
                        artifact.add_file(model_path)
                        run.log_artifact(artifact)
                        print(f"Model uploaded to wandb as artifact: model-{int(global_step/1_000_000)}M")
    
    envs.close()
    writer.close()



