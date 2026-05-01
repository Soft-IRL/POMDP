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
import imageio

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
    track: bool = True
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
        q = self.q_net(z)                                    # (B, A)
        greedy = q.argmax(dim=1, keepdim=True)               # (B, 1)

        if epsilon is None:
            epsilon = self.epsilon

        if epsilon <= 0.0:
            return greedy

        B = z.size(0)
        random_a = torch.randint(self.n_actions, (B, 1), device=z.device)
        take_rand = (torch.rand(B, 1, device=z.device) < epsilon)
        a = torch.where(take_rand, random_a, greedy)
        return a
    
    def compute_loss(self, z1, z2, actions, rewards, dones):
        B, S, d1 = z1.shape
        d = d1 + z2.size(-1)

        z_all   = torch.cat([z1, z2], dim=-1)   # (B, S, d)
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
        
        """#print("q_main:", q_main.shape)
        q_all  = self.q_net(z_t_f)                         # (BT, A)
        ##print("q_all:", q_all.shape)
        q_act  = q_all.gather(1, a_t_f.unsqueeze(-1)).squeeze(-1)
        #print("q_act:", q_act.shape)
        td_err = (q_act - y_main).detach()
        #print("td_err:", td_err.shape)"""

        """# per-action counts and mean |TD error|
        bins      = torch.bincount(a_t_f, minlength=self.n_actions)
        mean_abs  = torch.zeros(self.n_actions, device=z_t_f.device)
        for a in range(self.n_actions):
            mask = (a_t_f == a)
            mean_abs[a] = td_err[mask].abs().mean() if mask.any() else torch.tensor(0., device=z_t_f.device)

        print("counts:", bins.cpu().numpy(), "  mean|td_err|:", mean_abs.cpu().numpy())"""

        with torch.no_grad():
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
            print(f"mean|td_err| (per taken action): {mean_abs}")

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
    
    def compute_loss_old(self, z1, z2, actions, rewards, dones):
        B, S, d1 = z1.shape
        d = d1 + z2.size(-1)

        z_all   = torch.cat([z1, z2], dim=-1)   # (B, S, d)
        a_all   = actions.long()                # (B, S)
        r_all   = rewards
        done_all= dones.float()

        # --- regular transitions: t = 0..S-2
        z_t   = z_all[:, :-1]                   # (B, S-1, d)
        z_tp1 = z_all[:,  1:]                   # (B, S-1, d)
        a_t   = a_all[:, :-1]                   # (B, S-1)
        r_tp1   = r_all[:, :-1]                   # (B, S-1)
        d_tp1   = done_all[:, :-1]                # (B, S-1)

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
        
        """#print("q_main:", q_main.shape)
        q_all  = self.q_net(z_t_f)                         # (BT, A)
        ##print("q_all:", q_all.shape)
        q_act  = q_all.gather(1, a_t_f.unsqueeze(-1)).squeeze(-1)
        #print("q_act:", q_act.shape)
        td_err = (q_act - y_main).detach()
        #print("td_err:", td_err.shape)"""

        """# per-action counts and mean |TD error|
        bins      = torch.bincount(a_t_f, minlength=self.n_actions)
        mean_abs  = torch.zeros(self.n_actions, device=z_t_f.device)
        for a in range(self.n_actions):
            mask = (a_t_f == a)
            mean_abs[a] = td_err[mask].abs().mean() if mask.any() else torch.tensor(0., device=z_t_f.device)

        print("counts:", bins.cpu().numpy(), "  mean|td_err|:", mean_abs.cpu().numpy())"""

        with torch.no_grad():
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
            print(f"mean|td_err| (per taken action): {mean_abs}")
            if mean_abs[0] > 20.0:
                print("High TD error for action 0!")
                print("q_all_t for action 0:", q_all_t[:,0])

        # actions must be 0..A-1
        assert a_t_f.min().item() >= 0 and a_t_f.max().item() < self.n_actions, \
            f"Bad action indices in buffer: [{a_t_f.min().item()}, {a_t_f.max().item()}]"

        # targets must be 1D and match BT
        assert y_main.ndim == 1 and y_main.shape[0] == z_t_f.shape[0]
        
        q_loss = 0.5 * F.mse_loss(q_main, y_main, reduction="mean")

        # --- terminal last step: include only if this slice ends the episode
        last_done = (done_all[:, -1] > 0.5)
        if last_done.any():
            idx    = last_done.nonzero(as_tuple=False).squeeze(-1)
            z_last = z_all[idx, -1]                              # (B_term, d)
            a_last = a_all[idx, -1]                              # (B_term,)
            r_last = r_all[idx, -1]                              # (B_term,)
            q_last = self.q_net(z_last).gather(1, a_last.unsqueeze(-1)).squeeze(-1)
            q_loss = q_loss + 0.5 * F.mse_loss(q_last, r_last, reduction="mean")

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
        last = [m for m in self.q_net.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            # grads for each action head (row-wise)
            row_grad = last.weight.grad.norm(dim=1).cpu().numpy()   # shape (A,)
            print("head grad norms:", row_grad)

        #torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        #self.q_opt.step()

        ############# PUT THIS BACK IN IF USING SOFT Q-LEARNING #############
        """self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.log_alpha.data.clamp_(min=self.min_log_alpha)"""
        #######################################################################

    def update_target_model(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())
    
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

# ---------- helper: one-hot encode action index tensor (1,) ----------
def _one_hot_idx(a_idx: torch.Tensor, n_actions: int) -> torch.Tensor:
    # a_idx shape: (1,) int64
    return F.one_hot(a_idx.to(torch.long), num_classes=n_actions).to(torch.float32)

@torch.no_grad()
def play_one_episode(
    env,
    model,                      # your ModelDistributionNetwork
    agent,                      # your Qagent (ε-greedy)
    action_mapping: dict,       # e.g., {0:0, 1:2, 2:3}
    device: torch.device,
    *,
    epsilon: float = 0.05,      # exploration during eval (0.0 for greedy)
    max_steps: int = 10000,
    save_video_path: str | None = None,
    print_debug: bool = True
):
    # --- optional video writer ---
    #writer = imageio.get_writer(save_video_path, fps=30) if save_video_path else None
    #close_writer = (writer is not None)

    # --- reset env ---
    obs, _ = env.reset()
    total_reward, steps = 0.0, 0

    # --- init belief with FIRST posteriors (matches your loop) ---
    imgs0 = torch.from_numpy(obs).unsqueeze(1).to(device).float() / 255.0
    feat0  = model.encoder(imgs0)                           # (1, feat)
    z1_bel = model.latent1_first_posterior(feat0).rsample() # (1, d1)
    z2_bel = model.latent2_first_posterior(z1_bel).rsample()# (1, d2)
    prev_action_idx = torch.zeros(1, dtype=torch.long, device=device)  # NOOP index = 0

    # --- fetch first frame for video ---
    #if writer:
        #frame = env.render()  # rgb array
        #if frame is not None:
            #writer.append_data(frame)

    # --- loop until done/truncated or max_steps ---
    while steps < max_steps:
        steps += 1

        # 1) PREDICT with priors (uses prev action + current belief)
        a_one = _one_hot_idx(prev_action_idx, model.action_dim)  # (1, A)
        p1    = model.latent1_prior(z2_bel, a_one).base_dist
        z1_pr = p1.loc
        p2    = model.latent2_prior(z1_pr, z2_bel, a_one).base_dist
        z2_pr = p2.loc

        # 2) UPDATE with current observation → posterior belief (this is your Bayes filter)
        img_t = torch.from_numpy(obs).unsqueeze(1).to(device).float() / 255.0  # (1,1,64,64)
        feat  = model.encoder(img_t)
        q1    = model.latent1_posterior(feat, z2_bel, a_one)  # (uses z2_{t-1}, a_{t-1})
        z1_t  = q1.rsample()
        q2    = model.latent2_posterior(z1_t, z2_bel, a_one)
        z2_t  = q2.rsample()
        z1_bel, z2_bel = z1_t, z2_t

        # 3) Choose action from current belief
        z_cat = torch.cat([z1_bel, z2_bel], dim=1)  # (1, d1+d2)
        a_idx = agent.act(z_cat, epsilon=epsilon).squeeze(1)  # (1,)
        real_a = np.array([action_mapping[int(a_idx.item())]], dtype=np.int64)


        # 4) Step env
        obs, reward, done, truncated, info = env.step(real_a)
        total_reward += float(reward)

        # 5) Debug print
        if print_debug:
            q_vals = agent.q_net(z_cat).squeeze(0).cpu().numpy()  # (A,)
            print( f"t={steps:04d} | r={reward} | done={bool(done)} | a_idx={int(a_idx)}→{real_a} "
                  f"| Q={np.array2string(q_vals, precision=2, floatmode='fixed')}")

        # 6) Append video frame
        #if writer:
        #    frame = env.render()
        #    if frame is not None:
        #        writer.append_data(frame)

        # 7) Prepare for next step
        prev_action_idx = a_idx.to(device)

        if done or truncated:
            break

    #if close_writer:
    #    writer.close()

    print(f"\nEpisode finished: return={total_reward:.1f}, length={steps} steps")
    return total_reward, steps

def gather_experience(env, model, agent, rb, args, steps=1000):
    obs, _ = env.reset(seed=args.seed)
    episode_first = np.ones(args.num_envs, dtype=bool)   # True right after reset
    # previous action indices per env (start with NOOP index = 0)
    prev_actions = torch.zeros(args.num_envs, dtype=torch.long, device=args.device) # NOOP
    start_time = time.time()

    with torch.no_grad():
        imgs0  = torch.from_numpy(obs).unsqueeze(1).to(args.device).float() / 255.0
        feat0  = Model.encoder(imgs0)                           # (N, feat)
        z1_bel = Model.latent1_first_posterior(feat0).rsample() # (N, d1)
        z2_bel = Model.latent2_first_posterior(z1_bel).rsample()# (N, d2)
    
    for global_step in range(steps):
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
        next_obs, rewards, terminations, truncations, infos = env.step(real_actions)
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


args = tyro.cli(Args)
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
args.device = device
PATH = "C:\\Users\\Simo\\Documents\\Python Scripts\\SLAC\\runs\\ALE\\Pong-v5__SLAC_PONG_deterministic__1__1757940659\\SLAC_PONG_deterministic.pt"
ACTION_MAPPING = {0: 0, 1: 2, 2: 3}
n_actions = len(ACTION_MAPPING.keys())
action_space = Discrete(n_actions)


if __name__ == "__main__":
    envs = SyncVectorEnv([make_env(args.env_id, args.seed + i, i,
              args.capture_video) for i in range(args.num_envs)])
    agent = Qagent(action_space.n, args)
    Model = ModelDistributionNetwork(action_space, args)

    checkpoint = torch.load(PATH, map_location=args.device)

    latent_dim = args.latent1_size + args.latent2_size
    if Model.decoder.deconv1.weight.shape[0] != latent_dim:
        Model.decoder.deconv1 = nn.ConvTranspose2d(latent_dim, 8*args.base_depth, kernel_size=4, stride=1, padding=0).to(args.device)
        Model.decoder.deconv1_initialized = True
    
    Model.load_state_dict(checkpoint['model_state'])
    agent.q_net.load_state_dict(checkpoint['q_state'])

    obs,_ = envs.reset(seed=args.seed)

    rb = SequenceReplayBuffer(
        capacity   = args.buffer_size,
        obs_shape  = (1,obs.shape[1],obs.shape[2]),
        act_shape  = (),
        seq_len    = args.sequence_len,
        device     = args.device,)
    

    obs,_ = envs.reset()
    done = False
    total_reward = 0.0

    """"for i in range(1500):
        action = np.array([action_space.sample() for _ in range(envs.num_envs)])
        real_action = np.array([ACTION_MAPPING[a.item()] for a in action])
        obs, reward, done, truncation, infos = envs.step(real_action)
        total_reward += reward
        if done:
            print(f"Episode Reward: {total_reward}")
            done = False
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(infos)
                        print(f"episodic_return={info['episode']['r']}")
                        plt.imshow(last_obs[0], cmap='gray')
                        plt.axis('off')
                        plt.show()
                        bla
        last_obs = obs"""
    
    # Run one episode (set epsilon=0.0 for greedy)
    """play_one_episode(
        envs,
        Model, agent, ACTION_MAPPING, args.device,
        epsilon=0.05,
        save_video_path="eval_episode.mp4",   # or None
        print_debug=True
    )"""
    envs.close()
    gather_experience(envs, Model, agent, rb, args, steps=1000)
    print(f"Replay buffer size after gathering experience: {rb.ptr}")

    data = rb.sample(args.batch_size)
    images  = data["obs"].to(dtype=torch.float32).div_(255.)
    actions = data["action"]
    step_ty = data["step_type"]
    rewards = data["reward"]
    dones   = data["done"]
    last_done = np.bincount(dones[:, -1].cpu().numpy(), minlength=2)
    print("last step done counts (0,1):", last_done)

    a_batch = data["action"].cpu().numpy().reshape(-1)
    mb_counts = np.bincount(a_batch, minlength=agent.n_actions)
    print("minibatch action counts:", mb_counts)

    with torch.no_grad():
        (z1, z2), _ = Model.sample_posterior(images, actions, step_ty)
    
    print("z1:", z1.mean(), z1.min(), z1.max())
    print("z2:", z2.mean(), z2.min(), z2.max())
    q_loss, q_pred = agent.compute_loss_old(z1, z2, actions, rewards, dones)
    agent.update(q_loss)
    #q_loss, q_pred = agent.compute_loss(z1, z2, actions, rewards, dones)
    #agent.update(q_loss)




