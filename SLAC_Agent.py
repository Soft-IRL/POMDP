import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal, Independent, Bernoulli, kl_divergence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import functools

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
        self.deconv5 = nn.ConvTranspose2d(base_depth, 2*channels,
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
        x = self.deconv5(x)                  # (B*T, 2C, 64, 64)

        # 3. split into mean & log_sigma ------------------------------------
        C = x.shape[1] // 2
        mu, log_sigma = x[:, :C], x[:, C:]

        sigma = F.softplus(log_sigma) + self.eps     # positivity

        # 4. reshape back & wrap into distribution --------------------------
        mu     = mu.view(*leading_shape, C, 64, 64)
        sigma  = sigma.view_as(mu)

        dist = Independent(Normal(loc = mu, scale = sigma),
                           reinterpreted_batch_ndims = 3)   # H,W,C as one event
        return dist


class ModelDistributionNetwork(nn.Module):
    def __init__(self, action_space, args, model_reward=False, model_discount=False,
                 decoder_stddev=0.05, reward_stddev=None):
        
        super().__init__()
        self.base_depth = args.base_depth
        self.encoder_output_size = 8 * self.base_depth
        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.device = args.device
        self.lr = args.m_learning_rate
        self.epsilon = args.start_e
        self.latent1_size = args.latent1_size
        self.latent2_size = args.latent2_size
        self.model_reward = model_reward
        self.model_discount = model_discount
        self.decoder_stddev = decoder_stddev
        self.reward_stddev = reward_stddev
        self.kl_analytic = args.kl_analytic
        
        # p(z_1^1)
        self.latent1_first_prior = ConstantMultivariateNormalDiag(self.latent1_size, scale=1.0).to(self.device)
        # p(z_1^2 | z_1^1)
        self.latent2_first_prior = MultivariateNormalDiag(self.latent1_size, 8 * self.base_depth, self.latent2_size).to(self.device)
        # p(z_{t+1}^1 | z_t^2, a_t)
        self.latent1_prior = MultivariateNormalDiag(self.latent2_size + self.action_dim, 8 * self.base_depth, self.latent1_size).to(self.device)
        # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        self.latent2_prior = MultivariateNormalDiag(self.latent1_size + self.latent2_size + self.action_dim, 8 * self.base_depth, self.latent2_size).to(self.device)

         # q(z_1^1 | x_1)
        self.latent1_first_posterior = MultivariateNormalDiag(self.encoder_output_size, 8 * self.base_depth, self.latent1_size).to(self.device)
        # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
        self.latent2_first_posterior = self.latent2_first_prior
        # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
        self.latent1_posterior = MultivariateNormalDiag(self.encoder_output_size + self.latent2_size + self.action_dim, 8 * self.base_depth, self.latent1_size).to(self.device)

        # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        self.latent2_posterior = self.latent2_prior

        # compresses x_t into a vector
        self.encoder = Encoder(self.base_depth, 8* self.base_depth).to(self.device)
        # p(x_t | z_t^1, z_t^2)
        #self.decoder = Decoder(self.base_depth, scale=self.decoder_stddev).to(self.device)
        self.decoder = Decoder(self.base_depth).to(self.device)

        # ------------ optimizer ----------------------------------------------------
        # gather *all* parameters of sub-modules registered above
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.parameters(), 20.0)
        self.optimizer.step()

    def compute_loss(self, images, actions, step_types, step=None, rewards=None, discounts=None, latent_posterior_samples_and_dists=None):
        
        #This gets the number of transitions, which is one less than the number of steps.
        sequence_length = step_types.shape[1] - 1
        
        #If not provided, sample the latent variables and distributions from the encoder (inference model) conditioned on the current sequence.
        if latent_posterior_samples_and_dists is None:
            latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types) # q(z1_0 | x0)  , q(z2_0 | z1_0), q(z1_t | x_t, z2_{t-1}, a_{t-1}), q(z2_t | z1_t, z2_{t-1}, a_{t-1})
        
        #Latent variables and their corresponding distributions for both z1 and z2.
        (z1_post, z2_post), (q_z1, q_z2) = latent_posterior_samples_and_dists

        #print(z1_post.dtype, z1_post.min().item(), z1_post.max().item())
        #print(z2_post.dtype, z2_post.min().item(), z2_post.max().item())

        # ------------------------------------------------------------------ build PRIOR distributions (aligned)
        p_z1, p_z2, p_z1_auto, p_z2_auto = self.get_prior(z1_post, z2_post, actions, step_types) # For every t=0…T−1: pψ(zt+1∣zt2,at) and pψ(zt+12∣zt+1,zt2,at)

        #print(q_z1.dists.base_dist.loc.shape)
        #print(kl_divergence(q_z1.dists, p_z1.dists).shape)
    

        # ------------------------------------------------------------------ KL terms
        if self.kl_analytic:
            kl_z1 = kl_divergence(q_z1.dists, p_z1.dists).sum(-1)  
            #kl_z1 = kl_divergence(q_z1.dists, p_z1.dists)
            #kl_z2 = kl_divergence(q_z2.dists, p_z2.dists).sum(-1)
        else:
            # sample-based (still broadcasts)
            kl_z1 = q_z1.log_prob(z1_post) - p_z1.log_prob(z1_post)         # (B,T+1)
            kl_z2 = q_z2.log_prob(z2_post) - p_z2.log_prob(z2_post)
        
        # ------------------------------------------------------------------ recon term
        x_dist   = self.decoder(z1_post, z2_post)           # p(x|z)  Independent Normal
        #preds_imgs = x_dist.base_dist.loc
        preds_imgs = x_dist.rsample()                              # (B,T+1,C,H,W)
        #print(x_dist.base_dist.loc.dtype, x_dist.base_dist.loc.min().item(), x_dist.base_dist.loc.max().item())
        log_px   = x_dist.log_prob(images).sum(1)           # (B,)
        
        #with torch.no_grad():
        #mse = ((images - preds_imgs)**2).sum((1,2,3,4))
        mse = ((images - preds_imgs)**2).mean()

        """# KL warm-up or β-VAE
        if step is not None:
            beta = min(1.0, step / 50000)     # linearly annealed over e.g. 10k steps
            C_max   = 160.0                       # nats
            C       = C_max * min(1.0, step/25_000)
        else: 
            beta = 1.0                         # no annealing, just use β=1
            C = 0
        #elbo = log_px - beta * (kl_z1 - C).abs() - kl_z2
        #beta = 0.0001
        #kl_z1_term = torch.clamp(C - kl_z1, min=0).sum(-1) 
        elbo = log_px - kl_z1 - kl_z2
        #elbo = log_px - kl_z1_term - kl_z2

        # ------------------------------------------------Action Conditionning 
        # Posterior samples, skip t = 0:
        z1_next = z1_post[:, 1:]          # (B,T,·)
        z2_next = z2_post[:, 1:]          # (B,T,·)

        # Negative log-likelihoods
        nll_z1 = -p_z1_auto.log_prob(z1_next)  # (B,T)
        nll_z2 = -p_z2_auto.log_prob(z2_next)  # (B,T)

        pred_loss = (nll_z1 + nll_z2).mean()   # scalar
        λ = 10.0

        sigma = x_dist.base_dist.scale
        sigma_reg = 2e-3 * (sigma.log() ** 2).mean() 

        # ------------------------------------------------ loss
        loss = -elbo.mean() + λ * pred_loss + sigma_reg

        # q_z1.dists is Independent(Normal(...), 1)
        q_base = q_z1.dists.base_dist           # Normal, shape (B,T,D)
        p_base = p_z1.dists.base_dist           # Normal, same shape
        kl_z1_per_dim = kl_divergence(q_base, p_base).mean(dim=(0, 1))


        output = {"log_px": log_px.mean(),
                  "kl_z1": kl_z1.mean(),
                  "kl_z2": kl_z2.mean(),
                  "kl_z1_dim": kl_z1_per_dim, 
                  "kl_z1_manual": kl_divergence(q_base, p_base).sum(dim=1).sum(dim=1).mean(),
                  "pred_loss": pred_loss,
                  "sigma_min": x_dist.base_dist.scale.min(),
                  "sigma_median": x_dist.base_dist.scale.median(),
                  "sigma_max": x_dist.base_dist.scale.max(),
                  "sigma_reg": sigma_reg}"""

        output = {"log_px": log_px.mean(),
                  "mse": mse,
                  "sigma_min": x_dist.base_dist.scale.min(),
                  "sigma_median": x_dist.base_dist.scale.median(),
                  "sigma_max": x_dist.base_dist.scale.max(),
                  "kl_z1": kl_z1.mean()
                  }

        loss = mse + kl_z1.mean()

        #return loss, (z1_post, z2_post), output
        return loss, output


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
        actions = actions.transpose(0, 1)         # (T, B, action_dim)
        step_types = step_types.transpose(0, 1)   # (T+1, B)

        latent1_dists, latent1_samples = [], []
        latent2_dists, latent2_samples = [], []

        for t in range(sequence_length + 1):
            if t == 0:
                # Initial step: no previous latents
                #print(features.shape)
                #print(features[t].shape)
                #bla
                latent1_dist = self.latent1_first_posterior(features[t])           # q(z1_0 | x0)
                latent1_sample = latent1_dist.rsample()

                latent2_dist = self.latent2_first_posterior(latent1_sample)        # q(z2_0 | z1_0)
                latent2_sample = latent2_dist.rsample()
 
            else:
                latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1].unsqueeze(-1))  # q(z1_t | x_t, z2_{t-1}, a_{t-1})
                # Use conditional_distribution to conditionally select the correct posterior. Sample z1_t.
                latent1_sample = latent1_dist.rsample()
                latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1].unsqueeze(-1)) #  q(z2_t | z1_t, z2_{t-1}, a_{t-1})
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

        # t = 0  ---------
        p_z1_first = self.latent1_first_prior(step_types[:, :1])          # (B,1,d1) pψ​(z01​)
        p_z2_first = self.latent2_first_prior(z1_post[:, :1])             # (B,1,d2) pψ​(z02​∣z01​)

        # t = 1 … T  -----
        p_z1_auto  = self.latent1_prior(z2_post[:, :sequence_length], actions.unsqueeze(-1)) # For every t=0…T−1: pψ(zt+1∣zt2,at)
        p_z2_auto  = self.latent2_prior(z1_post[:, 1:], z2_post[:, :sequence_length], actions.unsqueeze(-1)) # For every t=0…T−1: pψ(zt+12∣zt+1,zt2,at)

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
    
    def get_grad_norm(self, module=None):
        total_norm = 0
        if module is None:
            params = self.parameters() 
        elif module == "latent1_first_prior":
            params = self.latent1_first_prior.parameters()
        elif module == "latent2_first_prior":
            params = self.latent2_first_prior.parameters()
        elif module == "latent1_prior":
            params = self.latent1_prior.parameters()
        elif module == "latent2_prior":
            params = self.latent2_prior.parameters()
        elif module == "latent1_first_posterior":
            params = self.latent1_first_posterior.parameters()
        elif module == "latent2_first_posterior":
            params = self.latent2_first_posterior.parameters()
        elif module == "latent1_posterior":
            params = self.latent1_posterior.parameters()
        elif module == "latent2_posterior":
            params = self.latent2_posterior.parameters()
        elif module == "encoder":
            params = self.encoder.parameters()
        elif module == "decoder":
            params = self.decoder.parameters()

        for p in params:
            if p.grad is not None:
                param_norm = p.grad.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def get_decoder_grad_norm(self):
        """
        Returns the gradient norm of the decoder parameters.
        """
        total_norm = 0
        for p in self.decoder.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def get_encoder_grad_norm(self):
        """
        Returns the gradient norm of the decoder parameters.
        """
        total_norm = 0
        for p in self.encoder.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
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
        #print(feat.shape)
        #bla
        z1_0   = self.latent1_first_posterior(feat).rsample()
        z2_0   = self.latent2_first_posterior(z1_0).rsample()

        z1_t, z2_t = z1_0, z2_0

        # ---- 2. autoregressive rollout -------------------------------------
        for t in range(H):
            a_t = torch.tensor(self.action_space.sample()).reshape(1,1).to(self.device)  # (1, action_dim)
            # • p(z1_{t+1} | z2_t, a_t)
            z1_dist = self.latent1_prior(torch.cat([z2_t, a_t], dim=-1))
            z1_t1   = z1_dist.sample()

            # • p(z2_{t+1} | z1_{t+1}, z2_t, a_t)
            z2_dist = self.latent2_prior(torch.cat([z1_t1, z2_t, a_t], dim=-1))
            z2_t1   = z2_dist.sample()

            # • decode image  p(x_{t+1}|z1_{t+1},z2_{t+1})
            img_dist = self.decoder(z1_t1, z2_t1)          # Independent Normal
            #x_mean   = img_dist.mean.clamp(0, 1)            # (1,1,H,W)
            x_sample = img_dist.rsample()          # or .sample()
            x_clamp  = x_sample.clamp(0, 1)        # (1,1,64,64)
            #decoded.append(x_mean)
            decoded.append(x_clamp)

            # · shift latents
            z1_t, z2_t = z1_t1, z2_t1

        # ---- 3. stack & save grid ------------------------------------------
        imgs = torch.cat(decoded, dim=0)   # (H+1,1,H,W)
        #save_image(imgs, "rollout.png", nrow=H+1)

        return imgs  # (H+1,1,H,W)
    
    @torch.no_grad()
    def visual_diagnostics(self, seq, actions, step_types, H=7):
        """
        seq: uint8 tensor (1, L, 1, 64, 64) – a short real sequence.
        """
        #device = self.device
        #seq = seq.to(device).float() / 255.0

        ###############################################################
        # 1) Posterior reconstructions (use encoder every step)
        ###############################################################
        #feats   = self.encoder(seq)                            # (1,L,feat)
        (_, _), (z1_dist, z2_dist)  = self.sample_posterior(seq, actions, step_types)    # (1,L,d1/2)
        z1 = z1_dist.loc
        z2 = z2_dist.loc        #print("z1.shape:", z1.shape, "z2.shape:", z2.shape)
        recon_dist   = self.decoder(z1, z2)         # (1,L,1,64,64)
        recon = recon_dist.rsample().clamp(0, 1)

        ###############################################################
        # 2) Conditional prior rollout (anchor on first frame only)
        ###############################################################
        """z1_cond, z2_cond = z1[:, :1], z2[:, :1]
        #print("z1_cond.shape:", z1_cond.shape, "z2_cond.shape:", z2_cond.shape)
        imgs_cond = [seq[:,0]]
        #print("imgs_cond[0].shape:", imgs_cond[0].shape) 

        for t in range(H):
            #a_t = torch.randint(self.action_dim, (1,1), device=device)
            #print(z2_cond.shape)
            #print(z2_cond[:,-1].shape)
            z1_cond_next = self.latent1_prior(torch.cat([z2_cond[:,-1], actions[0,t].reshape(1,1)], -1)).sample()
            z2_cond_next = self.latent2_prior(torch.cat([z1_cond_next, z2_cond[:,-1], actions[0,t].reshape(1,1)], -1)).sample()
            z1_cond   = torch.cat([z1_cond, z1_cond_next.unsqueeze(1)], dim=1)
            z2_cond   = torch.cat([z2_cond, z2_cond_next.unsqueeze(1)], dim=1)
            imgs_cond.append(self.decoder(z1_cond_next, z2_cond_next).mean.clamp(0,1))
        imgs_cond = torch.cat(imgs_cond, 1)   # (1,H+1,1,64,64)
        #print("imgs_cond.shape:", imgs_cond.shape)

        ###############################################################
        # 3) Un-conditional prior rollout
        ###############################################################
        z1_u = torch.randn_like(z1[:,:1])                      # (1,1,d1)
        #print("z1_u.shape:", z1_u.shape)
        z2_u = self.latent2_first_prior(z1_u).sample()        # (1,1,d2)
        #print("z2_u.shape:", z2_u.shape)
        imgs_u = self.decoder(z1_u, z2_u).mean.clamp(0,1)
        #print("imgs_u.shape:", imgs_u.shape)
        for t in range(H):
            #a_t  = torch.randint(self.action_dim, (1,1), device=device)
            z1_u = self.latent1_prior(torch.cat([z2_u, actions[0,t].reshape(1,1,1)], -1)).sample()
            #print("z1_u.shape:", z1_u.shape)
            z2_u = self.latent2_prior(torch.cat([z1_u, z2_u, actions[0,t].reshape(1,1,1)], -1)).sample()
            #print("z2_u.shape:", z2_u.shape)
            #print("decoder(z1_u, z2_u).mean.clamp(0,1).shape:", self.decoder(z1_u, z2_u).mean.clamp(0,1).shape)
            imgs_u = torch.cat([imgs_u, self.decoder(z1_u, z2_u).mean.clamp(0,1)], dim=1)  # (1,H+1,1,64,64)
            #print("imgs_u.shape:", imgs_u.shape)
            #imgs_u.append(self.decoder(z1_u, z2_u).mean.clamp(0,1))
        #print("imgs_u.shape:", imgs_u.shape)
        #imgs_u = torch.cat(imgs_u, 1)          # (1,H+1,1,64,64)

        return recon.squeeze(0), imgs_cond.squeeze(0).unsqueeze(1), imgs_u.squeeze(0)"""
        return recon.squeeze(0)

    def build_motion_mask(self, images, actions, step_types):
        
        latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types)
        (z1_post, z2_post), (q_z1, q_z2) = latent_posterior_samples_and_dists
        z1 = q_z1.loc
        z2 = q_z2.loc
        preds_imgs_dist   = self.decoder(z1_post, z2_post)
        preds_imgs = preds_imgs_dist.rsample().clamp(0, 1)  # (B, T+1, C, H, W) 

        # 1. differences between consecutive frames
        fd_pred = preds_imgs[:, 1:] - preds_imgs[:, :-1]       # (B, T, 1, 64, 64)
        fd_true = images[:, 1:] - images[:, :-1]
        motion   = fd_true.abs()
        # mask = 1 where motion > 1 gray-level, else 0
        mask = (motion > (0.001)).float()

        return preds_imgs, fd_pred, fd_true, mask


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