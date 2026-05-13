# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Stochastic Latent Actor-Critic (SLAC)** deep reinforcement learning implementation for **Partially Observable MDPs (POMDPs)**, specifically applied to a Light-Dark Navigation task. The algorithm combines a learned world model (VAE-style latent dynamics), distributional RL (C51/D3QN), and mutual information-based intrinsic motivation.

## Running the Project

No `requirements.txt` exists. Install dependencies manually:

```bash
pip install torch torchvision gymnasium tyro tqdm wandb imageio opencv-python matplotlib
```

**Main training script:**
```bash
python SLAC_light_dark_POMDP_C51_MI_new.py [args]
```

Key CLI arguments (all passed via `tyro.cli(Args)`):
- `--total-timesteps 500000` — total training steps
- `--pretrain-steps 100000` — world model pretraining steps before RL begins
- `--batch-size 128`, `--buffer-size 100000`, `--sequence-len 8`
- `--cuda true` / `--seed 42`
- `--track true` — enable W&B logging
- `--from-scratch true` — train world model from scratch (vs. load checkpoint)
- `--eval-every 10000`, `--video-every 20000`

There are no test suites or linters configured; this is research code.

## Architecture

Three components train jointly:

### 1. World Model — `SLAC_Agent_deterministic_tabular.py` (`ModelDistributionNetwork`)
Implements a Bayes filter with a **two-level latent hierarchy**:
- **`z1`** (32-dim, fast): temporal/transition information
- **`z2`** (256-dim, slow): persistent state identity

Sub-networks:
- **Encoder**: observation → feature embedding
- **Posteriors**: `(feature, z2_prev, action) → z1_t, z2_t` (inference at training time)
- **Priors**: `(z2_t, action) → predicted z1_{t+1}, z2_{t+1}` (transition dynamics)
- **Decoder**: `(z1, z2) → reconstructed obs`

Training objective: MSE reconstruction + KL with free-bits balancing (prevents posterior collapse).

### 2. RL Agent — `SLAC_Agent_D3QN_tabular.py` (`D3QNAgent`)
Double DQN with **C51** (categorical distributional RL):
- Input: concatenated `[z1, z2]` (~288-dim)
- Output: categorical return distribution over 51 atoms (support: `[-2.0, 1.0]`)
- Exploration: ε-greedy on expected Q-values
- Target network updated via Polyak averaging (`tau=0.005`)

### 3. Intrinsic Motivation — `compute_mi_bonus()` in main script
Computes mutual information `I(Z; R | a)` — how much latent state uncertainty affects the return distribution. Added to environment rewards with **automatic scaling** to maintain a target intrinsic/extrinsic ratio.

### Replay Buffer — `SequenceReplayBuffer.py`
Ring buffer storing fixed-length **sequences** (not individual transitions). All world model and RL updates operate on sequences of length `sequence_len`.

### Environment — `envs/Light_dark_POMDP_flags.py`
- **State**: 2D position + 2D velocity + 2D hidden drift (per-episode)
- **Obs**: noisy position + cue signal + goal info (~5D)
- **Actions**: discrete 25-action grid (5×5 accelerations ∈ `[-0.5, 0.5]² m/s²`)
- **Reward**: −0.01/step + 1.0 for reaching goal within velocity tolerance
- **Key challenge**: light vs. dark regions have different observation noise; hidden drift creates genuine partial observability

## Training Phases

1. **Bootstrap** (~10k steps): random action collection, no learning
2. **World model pretraining** (`--pretrain-steps`): reconstruction + KL only, no RL
3. **Joint learning**: world model + C51 agent train simultaneously

Key functions in the main script:
- `compute_loss()` (line ~605) — world model loss
- `compute_mi_bonus()` (line ~518) — intrinsic reward
- `evaluate_policy_deterministic()` (line ~806) — greedy evaluation tracking success rate, episode return, band visitation
- Main loop starts at line ~1705

## Logging & Checkpoints

- **W&B**: set `--track true`; `WANDB_DISABLE_SERVICE=true` is set for Windows stability
- **TensorBoard**: logs to `runs/`
- **Checkpoints**: saved to `checkpoints/LightDarkNavigation_POMDP/` (paths use Windows backslashes)
- **Videos**: saved to `videos/` when `--video-every` triggers

## Codebase Notes

- `SLAC_light_dark_POMDP_C51_MI_new.py` is the **active main script** (~2148 lines); other `SLAC_*.py`, `C51_*.py`, `DQN_*.py`, and `SLAC_PONG_*.py` files are older experiments
- `code_zhuojun_v2/` contains a collaborator's variant
- `envs/` has three environment variants; `Light_dark_POMDP_flags.py` is the current one
