import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal, Independent, Bernoulli, kl_divergence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import functools
import matplotlib.pyplot as plt

# helper to print tensor stats
def stats(t, tag):
    print(f"{tag}: min={t.min():.3e}  max={t.max():.3e}  mean={t.mean():.3e}  NaNs={torch.isnan(t).sum()}")

class StepType:
    FIRST = 0
    MID = 1
    LAST = 2

class MultivariateNormalDiag(nn.Module):
    """
    A neural network module for a multivariate normal distribution with diagonal covariance.
    Input: Observation
    Output: Learnt MultivariateNormal distribution with diagonal covariance matrix.
    """
    def __init__(self, input_dim, hidden_dim, latent_size):
        super().__init__()
        self.latent_size = latent_size
        #print("input_dim:", input_dim, "hidden_dim:", hidden_dim, "latent_size:", latent_size)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 2 * latent_size)
    
    def forward(self, *inputs):
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]

        x = F.leaky_relu(self.fc1(x))
        #stats(x, "after fc1")             # <--- add this
        x = F.leaky_relu(self.fc2(x))
        #stats(x, "after fc2")             # <--- and this
        out = self.output_layer(x)

        loc = out[..., :self.latent_size]
        scale_diag = F.softplus(out[..., self.latent_size:]) + 1e-5  # ensure positivity

        base  = Normal(loc, scale_diag)
        return Independent(base, 1)               # diagonal Gaussian
    
class ConstantMultivariateNormalDiag(nn.Module):
    """
        Constant diagonal Gaussian broadcast to the batch shape of *any* dummy
    input tensor.

        Returns Independent(Normal(loc, scale), 1), where
          • loc   ≡ 0
          • scale ≡ σ  (same on every coordinate unless you pass a vector)
    """

    def __init__(self, latent_size: int, scale: float | torch.Tensor = 1.0):
        super().__init__()
        self.latent_size = latent_size

        self.register_buffer("loc_const",
                             torch.zeros(latent_size))
        if isinstance(scale, torch.Tensor):
            assert scale.shape == (latent_size,)
            self.register_buffer("scale_const", scale.clone())
        else:                                # scalar σ
            self.register_buffer("scale_const",
                                 torch.ones(latent_size) * float(scale))

    def forward(self, *inputs):
        # infer batch_shape from first dummy arg (or ())
        batch_shape = inputs[0].shape if inputs else ()

        loc   = self.loc_const  .expand(*batch_shape, self.latent_size)
        scale = self.scale_const.expand_as(loc)

        base = Normal(loc, scale)            # (..., D)
        return Independent(base, 1)          # event_dim = 1  ⇒ mv Normal
    
class Encoder(nn.Module):
    """Encodes observations to the latent space."""
    def __init__(self, base_depth, feature_size):
        super().__init__()
        self.feature_size = feature_size
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=base_depth, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(base_depth, 2 * base_depth, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2 * base_depth, 4 * base_depth, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(4 * base_depth, 8 * base_depth, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(8 * base_depth, feature_size, kernel_size=4, stride=1, padding=0)  # VALID in TF = no padding

        self.activation = nn.LeakyReLU()

    def forward(self, image):
        """
        image: Tensor of shape (..., C, H, W), e.g., [B, T, C, H, W] or [B*T, C, H, W]
        Output: Tensor of shape (..., feature_size)
        """

        original_shape = image.shape[:-3]  # Save leading dims
        B = int(torch.prod(torch.tensor(original_shape)))  # Flatten batch

        x = image.view(B, *image.shape[-3:])  # reshape to (B*T, C, H, W)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))  # now shape (B*T, feature_size, 1, 1)

        x = x.view(*original_shape, self.feature_size)  # reshape to (..., feature_size)
        return x

class Decoder(nn.Module):
    """
    Probabilistic decoder p(x_t | z_t^1, z_t^2)
    with a learned per-pixel log-sigma.

    Output distribution:
        Normal(loc = μ, scale = σ)  where
            μ  ∈ ℝ^{C×H×W},
            σ  = softplus(logσ) + 1e-5   (positivity)
    """
    def __init__(self, base_depth: int, channels: int = 1):
        super().__init__()
        self.base_depth = base_depth
        self.act = nn.LeakyReLU()
        self.deconv1_initialized = False           

        # (latent_dim, 8*base_depth, 4*base, 2*base, base, 2*channels)
        # The final layer now outputs *twice* the channels: [μ ‖ logσ]
        self.deconv1 = nn.ConvTranspose2d(
            in_channels     = base_depth,
            out_channels    = 8 * base_depth,
            kernel_size     = 4, stride = 1, padding = 0)           # 1×1 → 4×4
        self.deconv2 = nn.ConvTranspose2d(8*base_depth, 4*base_depth,
                                          kernel_size = 3, stride = 2,
                                          padding = 1, output_padding = 1)      # 4→8
        self.deconv3 = nn.ConvTranspose2d(4*base_depth, 2*base_depth,
                                          kernel_size = 3, stride = 2,
                                          padding = 1, output_padding = 1)      # 8→16
        self.deconv4 = nn.ConvTranspose2d(2*base_depth, base_depth,
                                          kernel_size = 3, stride = 2,
                                          padding = 1, output_padding = 1)      # 16→32
        self.deconv5 = nn.ConvTranspose2d(base_depth, channels,
                                          kernel_size = 5, stride = 2,
                                          padding = 2, output_padding = 1)      # 32→64

        # small constant for numerical stability
        self.register_buffer("eps", torch.tensor(1e-5, dtype=torch.float32))

    # --------------------------------------------------------------------- #
    def forward(self, *inputs):
        # 1. concat latents -------------------------------------------------
        if len(inputs) > 1:
            z = torch.cat(inputs, dim=-1)     # (..., latent_dim_total)
        else:
            z = inputs[0]

        leading_shape = z.shape[:-1]          # e.g. (B, T)
        z = z.view(-1, z.shape[-1], 1, 1)     # (B*T, latent_dim, 1, 1)

        # -- lazy init ------------------------------------------------------
        if not self.deconv1_initialized:
            in_ch = z.shape[1]                       # latent_dim
            self.deconv1 = nn.ConvTranspose2d(in_ch,
                                              8*self.base_depth,
                                              4, 1, 0)
            self.deconv1.to(z.device)
            self.deconv1_initialized = True

        # 2. deconv tower ---------------------------------------------------
        x = self.act(self.deconv1(z))
        x = self.act(self.deconv2(x))
        x = self.act(self.deconv3(x))
        x = self.act(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))                  # (B*T, C, 64, 64)

        return x.view(*leading_shape, x.size(1), 64, 64)


class MLPEncoder(nn.Module):
    """Encodes tabular x_t ∈ R^D into a feature vector."""
    def __init__(self, in_dim: int, hidden: int, feature_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, feature_size)
        )
        self.feature_size = feature_size

    def forward(self, x):                     # x: (..., D) or (..., 1, D)
        # If there's a singleton right before the last dim, drop it
        if x.dim() >= 2 and x.shape[-2] == 1:
            x = x.squeeze(-2)

        batch = x.shape[:-1]                  # e.g., [128, 8]
        x = x.reshape(-1, x.shape[-1])        # use reshape (safer than view if not contiguous)
        y = self.net(x)
        return y.reshape(*batch, self.feature_size)  # -> [128, 8, 256]


class MLPDecoder(nn.Module):
    """Decodes latent → reconstruction in data space R^D (MSE loss)."""
    def __init__(self, latent_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, *inputs):  # inputs: (..., d1) and/or (..., d2)
        z = torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]
        orig = z.shape[:-1]                # e.g. (128, 8)
        z = z.reshape(-1, z.shape[-1])     # safer than view for non-contiguous
        y = self.net(z)                    # (N, out_dim)
        return y.reshape(*orig, 1, self.out_dim)  # -> (128, 8, 1, 4)
    
class ModelDistributionNetwork(nn.Module):
    def __init__(self, action_space, args, model_reward=False, model_discount=False,
                 decoder_stddev=0.05, reward_stddev=None):
        super().__init__()
        self.action_space = action_space
        self.action_dim   = action_space.n
        self.device       = args.device
        self.lr           = args.m_learning_rate
        self.latent1_size = args.latent1_size
        self.latent2_size = args.latent2_size
        self.base_depth   = args.base_depth
        self.kl_analytic  = args.kl_analytic

        # ----- modality selector -----
        self.obs_kind = getattr(args, "obs_kind", "image")   # "image" or "tabular"
        if self.obs_kind == "image":
            # image encoder/decoder
            self.encoder_output_size = 8 * self.base_depth
            self.encoder = Encoder(self.base_depth, self.encoder_output_size).to(self.device)
            self.decoder = Decoder(self.base_depth).to(self.device)
            self._decode_is_image = True
        else:
            # tabular encoder/decoder
            self.tabular_dim = int(getattr(args, "tabular_dim"))
            self.encoder_output_size = 8 * self.base_depth
            self.encoder = MLPEncoder(self.tabular_dim, hidden=256, feature_size=self.encoder_output_size).to(self.device)
            # latent concat dimension is set at first forward (same lazy pattern as image decoder)
            self.decoder_mlp_hidden = 256
            self.decoder = MLPDecoder(self.latent1_size+self.latent2_size, self.tabular_dim, hidden=self.decoder_mlp_hidden).to(self.device)
            self._decode_is_image = False

        # ----- priors & posteriors (unchanged) -----
        self.latent1_first_prior = ConstantMultivariateNormalDiag(self.latent1_size, scale=1.0).to(self.device)
        self.latent2_first_prior = MultivariateNormalDiag(self.latent1_size, 8*self.base_depth, self.latent2_size).to(self.device)
        self.latent1_prior       = MultivariateNormalDiag(self.latent2_size + self.action_dim, 8*self.base_depth, self.latent1_size).to(self.device)
        self.latent2_prior       = MultivariateNormalDiag(self.latent1_size + self.latent2_size + self.action_dim, 8*self.base_depth, self.latent2_size).to(self.device)

        self.latent1_first_posterior = MultivariateNormalDiag(self.encoder_output_size, 8*self.base_depth, self.latent1_size).to(self.device)
        self.latent2_first_posterior = self.latent2_first_prior
        self.latent1_posterior       = MultivariateNormalDiag(self.encoder_output_size + self.latent2_size + self.action_dim,
                                                              8*self.base_depth, self.latent1_size).to(self.device)
        self.latent2_posterior       = self.latent2_prior

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    # Lazy creation of the tabular decoder once we see latents
    def _ensure_tabular_decoder(self, z1, z2):
        if self.decoder is None:
            latent_dim = z1.size(-1) + z2.size(-1)
            self.decoder = MLPDecoder(latent_dim, self.tabular_dim, hidden=self.decoder_mlp_hidden).to(self.device)
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.parameters(), 20.0)
        self.optimizer.step()
    
    def compute_loss(self, x, actions, step_types,
                 step=None, rewards=None, discounts=None,
                 latent_posterior_samples_and_dists=None):

        # x: image  -> (B, S, 1, H, W)
        # x: table  -> (B, S, D)

        # 1) Get (z1,z2) from posterior
        if latent_posterior_samples_and_dists is None:
            (z1_post, z2_post), (q_z1, q_z2) = self.sample_posterior(x, actions, step_types)
        else:
            (z1_post, z2_post), (q_z1, q_z2) = latent_posterior_samples_and_dists

        # 2) Reconstruction
        if self._decode_is_image:
            # x in [0,1], decoder returns (B,S,1,H,W)
            preds = self.decoder(z1_post, z2_post)
            recon_loss = ((x - preds) ** 2).mean()
        else:
            # tabular: decoder is MLP on concat(z1,z2) → (B,S,D)
            self._ensure_tabular_decoder(z1_post, z2_post)
            preds = self.decoder(z1_post, z2_post)
            recon_loss = F.mse_loss(preds, x, reduction='mean')

        return recon_loss
    
    def sample_posterior(self, images, actions, step_types, features=None):
        """
        Sample latent1 and latent2 from the approximate posterior.
        Uses conditional_distribution and returns both samples and their stacked distributions.
        """
        #print("images.shape:", images.shape)
        # The sequence has T+1 timesteps (images.shape[1]), but actions only span T transitions. So we truncate actions to match the correct length.
        sequence_length = step_types.shape[1] - 1

        actions = actions[:, :sequence_length]
        #print("actions.shape:", actions.shape)

        if features is None:
            features = self.encoder(images)  # shape: (B, T+1, feat_dim)
        
        #print("features.shape:", features.shape)

        # Swap batch and time axes to get shape (T+1, B, ...)
        features = features.transpose(0, 1)       # (T+1, B, feature_dim)
        actions_tb = actions.transpose(0,1)          # (T, B, action_dim)
        actions = self._one_hot(actions_tb)
        step_types = step_types.transpose(0, 1)   # (T+1, B)

        latent1_dists, latent1_samples = [], []
        latent2_dists, latent2_samples = [], []

        for t in range(sequence_length + 1):
            if t == 0:
                # Initial step: no previous latents
                latent1_dist = self.latent1_first_posterior(features[t])           # q(z1_0 | x0)
                latent1_sample = latent1_dist.rsample()

                latent2_dist = self.latent2_first_posterior(latent1_sample)        # q(z2_0 | z1_0)
                latent2_sample = latent2_dist.rsample()
 
            else:
                #latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1].unsqueeze(-1))  # q(z1_t | x_t, z2_{t-1}, a_{t-1})
                latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1])  
                # Use conditional_distribution to conditionally select the correct posterior. Sample z1_t.
                latent1_sample = latent1_dist.rsample()
                latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1]) #  q(z2_t | z1_t, z2_{t-1}, a_{t-1})
                latent2_sample = latent2_dist.rsample()

            latent1_dists.append(latent1_dist)
            latent1_samples.append(latent1_sample)
            latent2_dists.append(latent2_dist)
            latent2_samples.append(latent2_sample)

        # Re-stack samples into shape (B, T+1, D)
        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        # Stack distributions into StackedNormal objects
        latent1_dists = stack_distributions(latent1_dists)
        latent2_dists = stack_distributions(latent2_dists)

        return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)
    
    def get_prior(self, z1_post, z2_post, actions, step_types=None):
        
        sequence_length = step_types.shape[1] - 1
        actions = actions[:, :sequence_length]
        actions = self._one_hot(actions)

        # t = 0  ---------
        p_z1_first = self.latent1_first_prior(step_types[:, :1])          # (B,1,d1) pψ​(z01​)
        p_z2_first = self.latent2_first_prior(z1_post[:, :1])             # (B,1,d2) pψ​(z02​∣z01​)

        # t = 1 … T  -----
        p_z1_auto  = self.latent1_prior(z2_post[:, :sequence_length], actions) # For every t=0…T−1: pψ(zt+1∣zt2,at)
        p_z2_auto  = self.latent2_prior(z1_post[:, 1:], z2_post[:, :sequence_length], actions) # For every t=0…T−1: pψ(zt+12∣zt+1,zt2,at)

        #------------------------ p_z1 -------------------------
        loc_first   = p_z1_first.base_dist.loc          # (B, 1, d1)
        scale_first = p_z1_first.base_dist.scale        # (B, 1, d1)
        loc_auto    = p_z1_auto .base_dist.loc          # (B, T, d1)
        scale_auto  = p_z1_auto .base_dist.scale        # (B, T, d1)
        locs_z1     = torch.cat([loc_first,   loc_auto],   dim=1)   # (B, T+1, d1)
        scales_z1   = torch.cat([scale_first, scale_auto], dim=1)   # (B, T+1, d1)
        p_z1 = StackedNormal(locs_z1, scales_z1)  

        #------------------------ p_z2 -------------------------
        loc_first2   = p_z2_first.base_dist.loc
        scale_first2 = p_z2_first.base_dist.scale
        loc_auto2    = p_z2_auto .base_dist.loc
        scale_auto2  = p_z2_auto .base_dist.scale
        locs_z2   = torch.cat([loc_first2,   loc_auto2],   dim=1)
        scales_z2 = torch.cat([scale_first2, scale_auto2], dim=1)
        p_z2 = StackedNormal(locs_z2, scales_z2)

        return p_z1, p_z2, p_z1_auto, p_z2_auto
    
    @torch.no_grad()
    def model_rollout(
        self,                     # your trained ModelDistributionNetwork
        init_obs,                  # (1, 1, H, W) uint8 tensor  (single frame)
        H     = 7):                 # horizon (so total frames = H+1):
        """
        Returns: decoded_imgs: (H+1, 1, H, W) float32 in [0,1]
        """
        #self.eval()

        # ---- 0. put on device, normalise ----------------------------------
        obs_t = init_obs.to(self.device, dtype=torch.float32) / 255.0  # (1,1,H,W)

        decoded = []                       # list of reconstructions
        decoded.append(obs_t)              # original frame at t=0

        # ---- 1. encode initial obs → z1_0, z2_0  ---------------------------
        feat   = self.encoder(obs_t)                    # (1, feat)
        z1_0   = self.latent1_first_posterior(feat).rsample()
        z2_0   = self.latent2_first_posterior(z1_0).rsample()

        z1_t, z2_t = z1_0, z2_0

        # ---- 2. autoregressive rollout -------------------------------------
        for t in range(H):
            # » pick an action
            a_t = torch.tensor([self.action_space.sample()]).to(self.device)  # (1, action_dim)
            a_t = F.one_hot(a_t, num_classes=self.action_dim).to(torch.float32)  # (1, action_dim)

            # • p(z1_{t+1} | z2_t, a_t)
            z1_dist = self.latent1_prior(torch.cat([z2_t, a_t], dim=-1))
            z1_t1   = z1_dist.sample()

            # • p(z2_{t+1} | z1_{t+1}, z2_t, a_t)
            z2_dist = self.latent2_prior(torch.cat([z1_t1, z2_t, a_t], dim=-1))
            z2_t1   = z2_dist.sample()

            # • decode image  p(x_{t+1}|z1_{t+1},z2_{t+1})
            img = self.decoder(z1_t1, z2_t1).clamp(0,1)           # Independent Normal

            #x_mean   = img_dist.mean.clamp(0, 1)            # (1,1,H,W)
            #x_sample = img_dist.rsample()          # or .sample()
            #x_clamp  = x_sample.clamp(0, 1)        # (1,1,64,64)
            #decoded.append(x_mean)
            decoded.append(img)

            # · shift latents
            z1_t, z2_t = z1_t1, z2_t1

        # ---- 3. stack & save grid ------------------------------------------
        imgs = torch.cat(decoded, dim=0)  # (H+1,1,H,W)

        #save_image(imgs, "rollout.png", nrow=H+1)

        return imgs  # (H+1,1,H,W)
    
    @torch.no_grad()
    def one_step_prior_predict(self, x, actions, step_types=None, use_posterior_means=True, clamp01=True):
        device = self.device

        # Ensure batch dims
        if x.dim() == 4:   # (S,1,H,W)  image
            x = x.unsqueeze(0)
        if x.dim() == 2:   # (S,D)      tabular
            x = x.unsqueeze(0)

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if step_types is not None and step_types.dim() == 1:
            step_types = step_types.unsqueeze(0)

        x = x.to(device).float()
        if self._decode_is_image and x.max() > 1.0 + 1e-6:
            x = x / 255.0

        actions = actions.to(device)
        step_types = step_types.to(device) if step_types is not None else None

        (z1_post, z2_post), (q_z1, q_z2) = self.sample_posterior(x, actions, step_types)
        z1_t = q_z1.dists.base_dist.loc if use_posterior_means else z1_post
        z2_t = q_z2.dists.base_dist.loc if use_posterior_means else z2_post
        z1_t, z2_t = z1_t.detach(), z2_t.detach()

        _, _, p_z1_auto, p_z2_auto = self.get_prior(z1_t, z2_t, actions, step_types)
        mu1 = p_z1_auto.base_dist.loc  # (B,T,d1)
        mu2 = p_z2_auto.base_dist.loc  # (B,T,d2)

        if self._decode_is_image:
            x_next = self.decoder(mu1, mu2)  # (B,T,1,H,W)
            if clamp01: x_next = x_next.clamp(0, 1)
        else:
            self._ensure_tabular_decoder(mu1, mu2)
            x_next = self.decoder(mu1, mu2)  # (B,T,D)

        return x_next, {"z1_teacher": z1_t, "z2_teacher": z2_t, "z1_next_mu": mu1, "z2_next_mu": mu2}
    
    def _one_hot(self, actions_bt: torch.Tensor) -> torch.Tensor:
        # actions_bt: (B, T) or (T, B) int64
        if actions_bt.ndim == 0: actions_bt = actions_bt.unsqueeze(0)  
        return F.one_hot(actions_bt.to(torch.long), num_classes=self.action_dim).to(actions_bt.device).float()

def stack_distributions(dists):
    locs = torch.stack([d.base_dist.loc for d in dists], dim=1)
    scales = torch.stack([d.base_dist.scale for d in dists], dim=1)
    return StackedNormal(locs, scales)

class StackedNormal:
        """Utility to represent a sequence of Normal distributions as a single distribution."""
        def __init__(self, locs, scales):
            self.dists = Independent(Normal(locs, scales), 1)

        def log_prob(self, value):
            return self.dists.log_prob(value)

        def sample(self):
            return self.dists.rsample()

        @property
        def loc(self):
            return self.dists.base_dist.loc