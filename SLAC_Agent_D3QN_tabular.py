# D3QNAgent_tabular.py
import copy
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


class D3QNAgent(nn.Module):
    """
    C51 + Double DQN head for SLAC latents (tabular or image world model — doesn't matter).
    Input:  z_cat = concat(z1, z2)  shape (B, latent1+latent2)
    Output: categorical distribution over N atoms for each discrete action.

    Keeps:
      - _logits(z)  -> (B, A, N)   (needed by your MI pipeline)
      - atoms buffer
      - compute_loss(z1,z2,actions,rewards,dones) with C51 projection
    """

    def __init__(self, n_actions: int, args):
        super().__init__()

        self.n_actions = int(n_actions)
        self.device = args.device

        # latent state size (tabular SLAC uses same z sizes)
        self.state_size = int(args.latent1_size + args.latent2_size)
        self.hidden_dims = getattr(args, "hidden_dims", (256, 256))

        # RL hyperparams
        self.lr = float(getattr(args, "q_learning_rate", 5e-4))
        self.gamma = float(getattr(args, "gamma", 0.99))
        self.epsilon = float(getattr(args, "start_e", 1.0))

        # C51 support
        self.n_atoms = int(getattr(args, "N_atoms", getattr(args, "n_atoms", 51)))

        # IMPORTANT: For LightDark, returns are not in [-1,1] necessarily.
        # If you don't set these in args, pick something sane.
        # For example, step_cost=-0.01, success_reward=1.0, horizon~200 -> min about -2, max about 1.
        self.Qmin = float(getattr(args, "Q_min", -2.0))
        self.Qmax = float(getattr(args, "Q_max",  1.0))
        assert self.Qmax > self.Qmin and self.n_atoms >= 2

        self.dz = (self.Qmax - self.Qmin) / (self.n_atoms - 1)

        atoms = torch.linspace(self.Qmin, self.Qmax, self.n_atoms, dtype=torch.float32)
        self.register_buffer("atoms", atoms)  # (N,)

        # Networks
        out_dim = self.n_actions * self.n_atoms
        self.q_net = self._build_mlp(self.state_size, out_dim, self.hidden_dims)
        self.q_target_net = copy.deepcopy(self.q_net)
        self.q_target_net.requires_grad_(False)
        self.q_target_net.eval()

        self.q_opt = optim.Adam(self.q_net.parameters(), lr=self.lr)

        # move module to device (buffers + nets)
        self.to(self.device)

    # -------------------------
    # Public API
    # -------------------------
    @torch.no_grad()
    def act(self, z: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
        """
        ε-greedy action selection w.r.t expected value of C51 distribution.
        Returns: (B, 1) long
        """
        if epsilon is None:
            epsilon = self.epsilon

        # accept (B,d) or (B,1,d)
        if z.dim() == 3 and z.shape[1] == 1:
            z = z.squeeze(1)

        z = z.to(self.device)
        z = F.layer_norm(z, z.shape[-1:])

        B = z.size(0)
        if np.random.rand() < float(epsilon):
            a = torch.randint(low=0, high=self.n_actions, size=(B, 1), device=self.device)
            return a

        logits = self._logits(z)                  # (B,A,N)
        probs = logits.softmax(dim=-1)            # (B,A,N)
        q = (probs * self.atoms).sum(dim=-1)      # (B,A)
        return q.argmax(dim=1, keepdim=True)      # (B,1)

    def update(self, loss: torch.Tensor):
        self.q_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_opt.step()

    def update_target_model(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / max(1, duration)
        return max(slope * t + start_e, end_e)

    # -------------------------
    # Core: C51 loss on SLAC sequences
    # -------------------------
    def compute_loss(self, z1, z2, actions, rewards, dones):
        """
        z1,z2: (B,S,d1), (B,S,d2)
        actions,rewards,dones: (B,S)
        Uses transitions t=0..S-2 with reward r_{t+1} and done_{t+1}.
        """
        B, S, d1 = z1.shape
        d = d1 + z2.size(-1)

        z_all = torch.cat([z1, z2], dim=-1)              # (B,S,d)
        z_all = F.layer_norm(z_all, z_all.shape[-1:])

        a_all = actions.long()
        r_all = rewards
        done_all = dones.float()

        z_t = z_all[:, :-1]                               # (B,S-1,d)
        z_tp1 = z_all[:, 1:]                              # (B,S-1,d)
        a_t = a_all[:, :-1]                               # (B,S-1)
        r_tp1 = r_all[:, 1:]                              # (B,S-1)
        d_tp1 = done_all[:, 1:]                           # (B,S-1)

        BT = B * (S - 1)
        z_t_f = z_t.reshape(BT, d)
        z_tp1_f = z_tp1.reshape(BT, d)
        a_t_f = a_t.reshape(BT)
        r_tp1_f = r_tp1.reshape(BT)
        d_tp1_f = d_tp1.reshape(BT)

        # Double DQN selection + target projection
        with torch.no_grad():
            # online selects a*
            logits_online = self._logits(z_tp1_f)                 # (BT,A,N)
            probs_online = logits_online.softmax(-1)              # (BT,A,N)
            q_online = (probs_online * self.atoms).sum(-1)        # (BT,A)
            a_star = q_online.argmax(dim=1)                       # (BT,)

            # target evaluates Z(s', a*)
            logits_target = self.q_target_net(z_tp1_f).view(BT, self.n_actions, self.n_atoms)
            p_target = logits_target.softmax(-1)                  # (BT,A,N)
            p_next_a = p_target[torch.arange(BT, device=self.device), a_star]  # (BT,N)

            target_proj = self._target_projection(p_next_a, r_tp1_f, d_tp1_f)  # (BT,N)
            target_mean = (target_proj * self.atoms).sum(-1)                   # (BT,)

        # Online logits for taken actions
        logits_t = self._logits(z_t_f)                                              # (BT,A,N)
        logits_taken = logits_t[torch.arange(BT, device=self.device), a_t_f]        # (BT,N)
        log_prob = F.log_softmax(logits_taken, dim=-1)                              # (BT,N)

        loss = -(target_proj * log_prob).sum(dim=-1).mean()

        with torch.no_grad():
            probs_taken = logits_taken.softmax(-1)
            q_taken_mean = (probs_taken * self.atoms).sum(-1)

        # safety
        assert a_t_f.min().item() >= 0 and a_t_f.max().item() < self.n_actions, \
            f"Bad action indices in batch: [{a_t_f.min().item()}, {a_t_f.max().item()}]"

        return loss, q_taken_mean, target_mean

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _build_mlp(in_dim, out_dim, hidden_dims):
        layers, last = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        return nn.Sequential(*layers)

    def _logits(self, z_flat: torch.Tensor) -> torch.Tensor:
        """
        Return logits shaped (B, A, N). Accepts (B,d) tensor.
        This is the function your MI pipeline should call.
        """
        z_flat = z_flat.to(self.device)
        if z_flat.dim() == 3 and z_flat.shape[1] == 1:
            z_flat = z_flat.squeeze(1)
        return self.q_net(z_flat).view(z_flat.size(0), self.n_actions, self.n_atoms)

    @torch.no_grad()
    def _target_projection(self, p_next_a: torch.Tensor, r: torch.Tensor, done: torch.Tensor):
        """
        Your projection, including the l==u correction.
        p_next_a: (BT,N)
        r, done: (BT,)
        """
        BT, N = p_next_a.size()
        atoms = self.atoms.unsqueeze(0)  # (1,N)

        Tz = r.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * self.gamma * atoms   # (BT,N)
        Tz = Tz.clamp(self.Qmin, self.Qmax)

        b = (Tz - self.Qmin) / self.dz  # (BT,N)
        l = b.floor()
        u = b.ceil()

        l_long = l.long().clamp(0, self.n_atoms - 1)
        u_long = u.long().clamp(0, self.n_atoms - 1)

        offset = (torch.arange(BT, device=self.device) * N).unsqueeze(1)  # (BT,1)
        l_idx = (l_long + offset).view(-1)
        u_idx = (u_long + offset).view(-1)

        m = torch.zeros(BT * N, device=self.device)

        lower_mass = (p_next_a * (u - b)).view(-1)
        upper_mass = (p_next_a * (b - l)).view(-1)
        m.index_add_(0, l_idx, lower_mass)
        m.index_add_(0, u_idx, upper_mass)

        # exact integer case
        eq = (u_long == l_long)
        if eq.any():
            eq_idx = (l_long + offset)[eq]
            m.index_add_(0, eq_idx.view(-1), p_next_a[eq].view(-1))

        m = m.view(BT, N)
        m_sum = m.sum(dim=1, keepdim=True).clamp_(min=1e-8)
        m /= m_sum
        return m
