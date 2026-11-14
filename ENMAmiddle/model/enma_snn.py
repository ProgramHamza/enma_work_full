from math import dist

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class ENMASNN(nn.Module):
    """
    POPSAN-style SNN actor.

    Architecture:
        obs (12) → FC(256) → LIF1 → FC(256) → LIF2
                → FC(256) → LIF3 → FC(256) → LIF4
                → mean firing rate over T=16 → FC(256→6) → tanh-scaled action

    LIF dynamics (per timestep t):
        I_t   = W * S_{t-1} + b
        U_t   = β * U_{t-1} + I_t - S_{t-1} * threshold
        S_t   = 1  if U_t >= threshold  else  0
    Surrogate gradient: fast sigmoid   dS/dU ≈ 1 / (π(1 + (αU)²))
    β = 0.9 (learnable), threshold = 1.0, T = 16 timesteps.
    Membrane potentials reset to zero at episode boundaries.
    """

    def __init__(self, obs_dim=12, act_dim=6, num_envs=1):
        super(ENMASNN, self).__init__()
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.num_envs = num_envs
        self.spike_ts = 16          # T=16 timesteps (POPSAN-style)
        self._hidden  = 256

        sg = surrogate.fast_sigmoid()

        # ── Spiking layers ────────────────────────────────────────────────────
        self.fc1    = nn.Linear(obs_dim,       self._hidden)
        self.lif1   = snn.Leaky(beta=0.9, learn_beta=True, spike_grad=sg)

        self.fc2    = nn.Linear(self._hidden,  self._hidden)
        self.lif2   = snn.Leaky(beta=0.9, learn_beta=True, spike_grad=sg)

        self.fc3    = nn.Linear(self._hidden,  self._hidden)
        self.lif3   = snn.Leaky(beta=0.9, learn_beta=True, spike_grad=sg)

        self.fc4    = nn.Linear(self._hidden,  self._hidden)
        self.lif4   = snn.Leaky(beta=0.9, learn_beta=True, spike_grad=sg)

        # ── Output projection (no spike — receives mean firing rate) ──────────
        self.fc_out = nn.Linear(self._hidden, act_dim)

        # ── Membrane states (initialised in reset()) ──────────────────────────
        self.mem1 = self.mem2 = self.mem3 = self.mem4 = None
        self.reset()

    # ── State management ──────────────────────────────────────────────────────

    def reset(self, env_ids=None):
        device = self.fc1.weight.device
        h      = self._hidden

        if env_ids is None:
            n = self.num_envs
            self.mem1 = torch.zeros(n, h,            device=device)
            self.mem2 = torch.zeros(n, h,            device=device)
            self.mem3 = torch.zeros(n, h,            device=device)
            self.mem4 = torch.zeros(n, h,            device=device)
        else:
            self.mem1[env_ids] = 0.0
            self.mem2[env_ids] = 0.0
            self.mem3[env_ids] = 0.0
            self.mem4[env_ids] = 0.0

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, obs):
        obs        = torch.nan_to_num(obs)
        batch_size = obs.shape[0]

        # Re-init membrane states if batch size or device changed
        if self.mem1 is None or self.mem1.shape[0] != batch_size or self.mem1.device != obs.device:
            self.num_envs = batch_size
            if self.mem1 is not None and self.mem1.device != obs.device:
                self.to(obs.device)
            self.reset()

        # Accumulate spikes from LIF4 over T timesteps
        spk4_sum = torch.zeros(batch_size, self._hidden, device=obs.device)

        for _ in range(self.spike_ts):
            spk1, self.mem1 = self.lif1(self.fc1(obs),  self.mem1)
            spk2, self.mem2 = self.lif2(self.fc2(spk1), self.mem2)
            spk3, self.mem3 = self.lif3(self.fc3(spk2), self.mem3)
            spk4, self.mem4 = self.lif4(self.fc4(spk3), self.mem4)
            spk4_sum += spk4

        # Mean firing rate ∈ [0, 1]^256  →  linear projection to 6D
        rate = spk4_sum / self.spike_ts     # (batch, 256)
        out  = self.fc_out(rate)            # (batch, 6)  — continuous, no spike

        # Per-dimension tanh scaling — action bounds unchanged
        delta_vx    = torch.tanh(out[:, 0:1]) * 0.3   # linear x:   ±0.3 m/s
        delta_vy    = torch.tanh(out[:, 1:2]) * 0.2   # linear y:   ±0.2 m/s
        delta_vz    = torch.tanh(out[:, 2:3]) * 0.1   # linear z:   ±0.1 m/s
        delta_roll  = torch.tanh(out[:, 3:4]) * 0.3   # roll rate:  ±0.3 rad/s
        delta_pitch = torch.tanh(out[:, 4:5]) * 0.3   # pitch rate: ±0.3 rad/s
        delta_yaw   = torch.tanh(out[:, 5:6]) * 0.5   # yaw rate:   ±0.5 rad/s

        return torch.cat(
            [delta_vx, delta_vy, delta_vz, delta_roll, delta_pitch, delta_yaw],
            dim=1
        )


class ENMACritic(nn.Module):
    """
    Deeper MLP critic — symmetric depth to the POPSAN actor.
    Architecture: obs (12) → 256 → 256 → 128 → 1  (ReLU activations)
    """

    def __init__(self, obs_dim=12, num_envs=1):
        super(ENMACritic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256,     256), nn.ReLU(),
            nn.Linear(256,     128), nn.ReLU(),
            nn.Linear(128,       1),
        )

    def forward(self, obs):
        return self.net(obs)

    def reset(self, env_ids=None):
        return None
