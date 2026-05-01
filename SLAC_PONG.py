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

from SLAC_Agent import ModelDistributionNetwork



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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_model: bool = True
    """if toggled, the trained model will be saved to disk"""
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
    alpha_lr: float =3e-4
    """the learning rate of the alpha optimizer"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.2
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    gamma: float = 0.99
    """the discount factor"""
    learning_starts: int = 1000
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
    
    @torch.no_grad()
    def act(self, z):
        """
        Select an action based on the current state z.
        :param z: latent representation (concatenation of z1 and z2)
        :return: selected action
        """
        q_values = self.q_net(z).float()
        alpha = self.log_alpha.exp().float()
        logits = q_values / alpha
        pi = torch.softmax(logits, dim=1, dtype=torch.float32)

        # 3. Clamp / renormalise
        pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
        pi = pi + 1e-8                             # avoid zero-sum
        pi = pi / pi.sum(dim=1, keepdim=True)
        action = torch.multinomial(pi, num_samples=1)
        return action
    
    def compute_loss(self, z1, z2, actions, rewards, dones):
        z_t     = torch.cat([z1[:, :-1], z2[:, :-1]], dim=-1)      # (B,S-1, d1+d2)
        z_tp1   = torch.cat([z1[:, 1:],  z2[:, 1:]],  dim=-1)      # (B,S-1, d1+d2)
        a_t     = actions[:, :-1]                                  # (B,S-1)
        #r_t     = rewards[:, :-1]                                  # (B,S-1)
        r_t     = rewards[:, 1:]  
        #done_t  = dones[:, :-1].float()                            # (B,S-1)
        done_t  = dones[:, 1:].float()  

        # Flatten transition dimension for network calls --------------------------
        z_t_f   = z_t.reshape(-1, z_t.shape[-1])                 # (B*(S-1), d)
        z_tp1_f = z_tp1.reshape(-1, z_tp1.shape[-1])
        a_t_f   = a_t.flatten()     

        # -------------------------------------------------------------------------
        # 1. TD target   y_t = r + γ α log Σ_a' exp(Q_target(z_{t+1},a')/α)
        # -------------------------------------------------------------------------
        with torch.no_grad():
            q_tp1_all = self.q_target_net(z_tp1_f)                # (B*(S-1), A)
            alpha     = self.log_alpha.exp()

            soft_max  = torch.logsumexp(q_tp1_all / alpha, dim=-1, keepdim=False)
            y = r_t.flatten() + self.gamma * (1.0 - done_t.flatten()) * (alpha * soft_max)
        
        # -------------------------------------------------------------------------
        # 2. Q-loss     ½ (Q(z_t,a_t) − y)^2
        # -------------------------------------------------------------------------
        q_pred = self.q_net(z_t_f).gather(1, a_t_f.unsqueeze(-1)).squeeze(-1)
            #print("q_pred:", self.q_net(z_t_f))
        #print(z_t_f.isnan().any())
        #print("q_pred:", q_pred, "y:", y)
        if q_pred.isnan().any():
            print("q_pred contains NaN values, check your model outputs!")
            raise ValueError("NaN values in q_pred")
        q_loss = 0.5 * F.mse_loss(q_pred, y, reduction="mean")
        #print("q_loss:", q_loss.item())


        # -------------------------------------------------------------------------
        # 3. α-loss     L_α = α · ( − log π − target_entropy )
        # -------------------------------------------------------------------------
        with torch.no_grad():
            # policy derived from current Q-values
            q_all      = self.q_net(z_t_f)                       # (B*(S-1), A)
            log_pi_all = F.log_softmax(q_all / alpha, dim=-1)
            entropy    = -(log_pi_all.exp() * log_pi_all).sum(-1) # (B*(S-1),)

        alpha_loss = (self.log_alpha.exp() * (entropy - self.target_entropy)).mean()

        return q_loss, alpha_loss, q_pred

    def update(self, q_loss, alpha_loss):
        self.q_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_opt.step()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.log_alpha.data.clamp_(min=self.min_log_alpha)

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

        artifact = wandb.Artifact("source_code", type="code")
        artifact.add_dir(script_dir) 
        run.log_artifact(artifact)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    envs = SyncVectorEnv([make_env(args.env_id, args.seed + i, i,
              args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    ACTION_MAPPING = {0: 0, 1: 2, 2: 3}
    n_actions = len(ACTION_MAPPING.keys())
    action_space = Discrete(n_actions)
    #action_dim = int(np.prod(action_space.shape))

    obs,_ = envs.reset(seed=args.seed)

    Model = ModelDistributionNetwork(action_space, args)
    agent = Qagent(action_space.n, args)

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

    #start the game
    obs, _ = envs.reset(seed=args.seed)
    episode_first = np.ones(args.num_envs, dtype=bool)   # True right after reset
    
    ############################# THIS IS THE MODEL PRETRAINING ###############################
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
    for pretrain_step in tqdm(range(300_000)):
        batch = rb.sample(args.batch_size)
        images  = (batch["obs"].float() / 255.)
        #print(images.dtype, images.min().item(), images.max().item())
        actions = batch["action"]
        step_ty = batch["step_type"]
        #prev_weights = {name: param.clone().detach() for name, param in Model.named_parameters()}
        #print(images[0].shape)
        #recon, img_cond, img_u = Model.visual_diagnostics(images[0,:].unsqueeze(0), actions[0,:].unsqueeze(0), step_ty)
        #print(recon.shape)
        
        model_loss, output = Model.compute_loss(images, actions, step_ty, step=pretrain_step)
        Model.optimizer.zero_grad()
        model_loss.backward()
        Model.optimizer.step()

        if args.track:
            if (pretrain_step) % 10_000 == 0:
                sequence = rb.sample(1)
                image = sequence["obs"][0][0].unsqueeze(0)  # Get the first image in the sequence
                rollout_images = Model.model_rollout(image, H=args.sequence_len-1)  # Generate a sequence of images
                log_rollout_grid(rollout_images, step=0, caption="Model Rollout")

                images  = (sequence["obs"].float() / 255.)
                actions = sequence["action"]
                step_ty = sequence["step_type"]

                recon = Model.visual_diagnostics(images, actions, step_ty)
                log_rollout_grid(recon, step=0, caption="Recon")
                log_rollout_grid(images[0], step=0, caption="Sequence")
                preds_imgs, fd_pred, fd_true, mask = Model.build_motion_mask(images, actions, step_ty)
                log_rollout_grid(fd_pred.squeeze(0), step=0, caption="Frame Difference Prediction")
                log_rollout_grid(fd_true.squeeze(0), step=0, caption="Frame Difference Ground Truth")
                log_rollout_grid(mask.squeeze(0), step=0, caption="Motion Mask")
                log_rollout_grid(preds_imgs.squeeze(0), step=0, caption="Predicted Images") 
            if (pretrain_step) % 500 == 0:
                """model_grad_norm = Model.get_grad_norm()
                decoder_grad_norm = Model.get_grad_norm("decoder")
                encoder_grad_norm = Model.get_grad_norm("encoder")
                latent2_first_prior_norm = Model.get_grad_norm("latent2_first_prior")
                latent1_prior_norm = Model.get_grad_norm("latent1_prior")
                latent2_prior_norm = Model.get_grad_norm("latent2_prior")
                latent1_first_posterior_norm = Model.get_grad_norm("latent1_first_posterior")
                latent1_posterior_norm = Model.get_grad_norm("latent1_posterior")
                writer.add_scalar("grads/latent2_first_prior_norm", latent2_first_prior_norm, pretrain_step)
                writer.add_scalar("grads/latent1_prior_norm", latent1_prior_norm, pretrain_step)
                writer.add_scalar("grads/latent2_prior_norm", latent2_prior_norm, pretrain_step)
                writer.add_scalar("grads/latent1_first_posterior_norm", latent1_first_posterior_norm, pretrain_step)
                writer.add_scalar("grads/latent1_posterior_norm", latent1_posterior_norm, pretrain_step)
                writer.add_scalar("grads/encoder_grad_norm", encoder_grad_norm, pretrain_step)

                writer.add_scalar("grads/decoder_grad_norm", decoder_grad_norm, pretrain_step)
                writer.add_scalar("losses/pretraining_model_loss", model_loss.item(), pretrain_step)
                writer.add_scalar("losses/log_px", output["log_px"].item(), pretrain_step)"""
                writer.add_scalar("losses/kl_z1", output["kl_z1"].item(), pretrain_step)
                #writer.add_scalar("losses/kl_z2", output["kl_z2"].item(), pretrain_step)
                #writer.add_scalar("losses/pred_loss", output["pred_loss"].item(), pretrain_step)
                #writer.add_scalar("losses/sigma_min", output["sigma_min"].item(), pretrain_step)
                #writer.add_scalar("losses/sigma_median", output["sigma_median"].item(), pretrain_step)
                #writer.add_scalar("losses/sigma_max", output["sigma_max"].item(), pretrain_step)
                """writer.add_scalar("losses/sigma_reg", output["sigma_reg"].item(), pretrain_step)
                writer.add_scalar("grads/model_grad_norm", model_grad_norm, pretrain_step)"""
                writer.add_scalar("losses/pretraining_model_loss", model_loss.item(), pretrain_step)
                writer.add_scalar("losses/mse", output["mse"].item(), pretrain_step)
                writer.add_scalar("losses/log_px", output["log_px"].item(), pretrain_step)
                #writer.add_scalar("losses/fd_loss", fd_loss.item(), pretrain_step)

                """writer.add_scalar("weights/latent2_prior_weight_norm", Model.latent2_prior.output_layer.weight.norm().item(), pretrain_step)
                writer.add_scalar("weights/latent2_first_prior_weight_norm", Model.latent2_first_prior.output_layer.weight.norm().item(), pretrain_step)
                writer.add_scalar("weights/latent1_prior_weight_norm", Model.latent1_prior.output_layer.weight.norm().item(), pretrain_step)
                writer.add_scalar("weights/latent1_first_posterior_weight_norm", Model.latent1_first_posterior.output_layer.weight.norm().item(), pretrain_step)
                writer.add_scalar("weights/latent1_posterior_weight_norm", Model.latent1_posterior.output_layer.weight.norm().item(), pretrain_step)
                writer.add_scalar("weights/decoder_weight_norm", Model.decoder.deconv5.weight.norm().item(), pretrain_step)
                writer.add_scalar("weights/encoder_weight_norm", Model.encoder.conv5.weight.norm().item(), pretrain_step)

                step = (args.m_learning_rate * latent1_prior_norm) / Model.latent1_prior.output_layer.weight.norm().item()
                ratio = latent1_prior_norm / Model.latent1_prior.output_layer.weight.norm().item()
                writer.add_scalar("charts/latent1_prior_step", step, pretrain_step)
                writer.add_scalar("charts/latent1_prior_ratio", ratio, pretrain_step)

                step = (args.m_learning_rate * latent1_first_posterior_norm) / Model.latent1_first_posterior.output_layer.weight.norm().item()
                ratio = latent1_first_posterior_norm / Model.latent1_first_posterior.output_layer.weight.norm().item()
                writer.add_scalar("charts/latent1_first_posterior_step", step, pretrain_step)
                writer.add_scalar("charts/latent1_first_posterior_ratio", ratio, pretrain_step)

                step   = (args.m_learning_rate * latent1_posterior_norm) / Model.latent1_posterior.output_layer.weight.norm().item()
                ratio  = latent1_posterior_norm / Model.latent1_posterior.output_layer.weight.norm().item()    
                writer.add_scalar("charts/latent1_posterior_step", step, pretrain_step)
                writer.add_scalar("charts/latent1_posterior_ratio", ratio, pretrain_step)

                step = (args.m_learning_rate * latent2_first_prior_norm) / Model.latent2_first_prior.output_layer.weight.norm().item()
                ratio = latent2_first_prior_norm / Model.latent2_first_prior.output_layer.weight.norm().item()
                writer.add_scalar("charts/latent2_first_prior_step", step, pretrain_step)
                writer.add_scalar("charts/latent2_first_prior_ratio", ratio, pretrain_step)

                step = (args.m_learning_rate * latent2_prior_norm) / Model.latent2_prior.output_layer.weight.norm().item()
                ratio = latent2_prior_norm / Model.latent2_prior.output_layer.weight.norm().item()
                writer.add_scalar("charts/latent2_prior_step", step, pretrain_step)
                writer.add_scalar("charts/latent2_prior_ratio", ratio, pretrain_step)

                step = (args.m_learning_rate * encoder_grad_norm) / Model.encoder.conv5.weight.norm().item()
                ratio = encoder_grad_norm / Model.encoder.conv5.weight.norm().item()
                writer.add_scalar("charts/encoder_step", step, pretrain_step)
                writer.add_scalar("charts/encoder_ratio", ratio, pretrain_step)

                step = (args.m_learning_rate * decoder_grad_norm) / Model.decoder.deconv5.weight.norm().item()
                ratio = decoder_grad_norm / Model.decoder.deconv5.weight.norm().item()
                writer.add_scalar("charts/decoder_step", step, pretrain_step)
                writer.add_scalar("charts/decoder_ratio", ratio, pretrain_step)"""
                

                
            """norms  = []
            for name, param in Model.named_parameters():
                norms.append(param.norm().item())
                writer.add_scalar("charts/model_param_norm", np.mean(norms), pretrain_step)"""
        
    if args.save_model:
        path = f"checkpoints\\{run_name}\\model_pretrained.pth"
        save_world_model_ckpt(Model, pretrain_step+1, path)
        log_checkpoint_to_wandb(path, pretrain_step+1, run, aliases=("pretrain", "latest"))

    ######################## END OF MODEL PRETRAINING ###############################
    print("Model pretraining completed.")

    bla

    start_time = time.time()

    for global_step in range(args.total_timesteps):
        agent.epsilon = agent.linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < agent.epsilon:      
            actions = np.array([action_space.sample() for _ in range(envs.num_envs)])
        else:
            #with tic("model_act"):
            flat_obs = Model.encoder(torch.FloatTensor(obs/255).unsqueeze(1).to(device))
            dist_z_1 = Model.latent1_first_posterior(flat_obs)
            z_1 = dist_z_1.rsample()
            dist_z_2 = Model.latent2_first_posterior(z_1)
            z_2 = dist_z_2.rsample()
            z = torch.cat((z_1, z_2), dim=1)
            actions = agent.act(z).cpu().numpy().flatten() 
            
        real_actions = np.array([ACTION_MAPPING[a.item()] for a in actions])

        #execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(real_actions)
        done = terminations | truncations 
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        step_type = np.where(
                done,               2,
                np.where(episode_first, 0, 1)
            ).astype(np.int64)      # shape (n_envs,)

        #with tic("store_buffer"):
        for k in range(args.num_envs):
            rb.add(obs[k][None],
            actions[k],
            rewards[k],
            done[k],
            step_type[k])      

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        episode_first = done.copy() 
        

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            #running_done = 0
            #running_n    = 0

            """for _ in range(100):                     # 100 batches
                batch = rb.sample(32)
                dones = batch["done"]                # (32, 8)
                running_done += dones[:, 1:].sum().item()   # count successor terminals
                running_n    += dones[:, 1:].numel()        # 32 × 7 = 224 each

                done_ratio = running_done / running_n
            print("average done ratio over 100 batches:", done_ratio)"""
            
            #bla
            #with tic("sample"):
            data = rb.sample(args.batch_size)
            images  = data["obs"].to(dtype=torch.float32).div_(255.)
            actions = data["action"]
            step_ty = data["step_type"]
            rewards = data["reward"]
            dones   = data["done"]

            #with tic("model_forward"):
            #with torch.amp.autocast("cuda"):
            model_loss, _ = Model.compute_loss(images,actions,step_ty)
            #with tic("model_update"):
            Model.update(model_loss)
                #with tic("sample_posterior"):
            (z1, z2), _ = Model.sample_posterior(images, actions[:, :-1], step_ty)
                #with tic("q_forward"):
            #with torch.amp.autocast("cuda"):
            q_loss, alpha_loss, q_pred = agent.compute_loss(z1, z2, actions, rewards, dones)
            #with tic("agent_update"):
            agent.update(q_loss, alpha_loss)
            
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

            if global_step % 100 == 0:
                writer.add_scalar("losses/model_loss", model_loss.item(), global_step)
                writer.add_scalar("losses/q_loss", q_loss.item(), global_step)
                writer.add_scalar("losses/q_values", q_pred.mean().item(), global_step)
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/epsilon", agent.epsilon, global_step)

            if global_step % 5000 == 0:
                model_grad_norm = Model.get_grad_norm()
                agent_grad_norm = agent.get_grad_norm()
                writer.add_scalar("charts/grad_norm", agent_grad_norm, global_step)
                writer.add_scalar("charts/model_grad_norm", model_grad_norm, global_step)
            
             # update target network
            if global_step % args.target_network_frequency == 0:
                # -------------------------------------------------------------------------
                # 4. Target-net update (polyak)
                # -------------------------------------------------------------------------
                """with torch.no_grad():
                    for p, p_targ in zip(agent.q_net.parameters(), agent.q_target_net.parameters()):
                        p_targ.data.mul_(1.0 - args.tau).add_(args.tau * p.data)"""
                agent.update_target_model()
            
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



