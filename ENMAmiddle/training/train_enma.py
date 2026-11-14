import argparse
import os
import sys
from pathlib import Path
import fcntl
import contextlib
import torch.nn.functional as F
from tqdm.auto import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train ENMA SNN")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--no-affect", "--no_affect", action="store_true", dest="no_affect",
                    help="Disable affect scoring and use zero scores.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument("--caesar_ckpt", type=str, default=None,
                    help="Path to the Caesar (FastAffectCNN) weights. Defaults to $ENMA_CAESAR_CKPT, "
                         "then <repo>/checkpoints/caesar/best_model.pth.")
AppLauncher.add_app_launcher_args(parser)
args, unknown_args = parser.parse_known_args()

# ALWAYS enable cameras so Isaac RTX renderer activates them even in --headless mode.
# Without this flag AppLauncher never initialises the RTX sensor pipeline and every
# camera.data.output["rgb"] tensor stays permanently zero.
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg

# Import custom modules
REPO_ROOT = Path(__file__).resolve().parents[2]
# Location of the unitree_rl_lab sources; override with $UNITREE_RL_LAB_SRC if it lives elsewhere.
UNITREE_SRC = Path(os.environ.get("UNITREE_RL_LAB_SRC", str(REPO_ROOT / "unitree_rl_lab" / "source")))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(UNITREE_SRC) not in sys.path:
    sys.path.insert(0, str(UNITREE_SRC))
from ENMAmiddle.model.enma_policy import ENMAPolicy
from ENMAmiddle.rewards.swifttribe_enma_plugin import SwiftTribeENMAPlugin
from ENMAmiddle.training.enma_ppo_cfg import ENMAPPORunnerCfg
from ENMAmiddle.training.rollout_buffer import RolloutBuffer
from ENMAmiddle.training.gae import compute_gae

# Isaac Lab imports for env registration
from unitree_rl_lab.tasks.locomotion.robots.go2.enma_env_cfg import ENMAEnv, ENMAInteractionSceneCfg
import gymnasium as gym

# ── FastCNN model definition (mirrors fast_cnn/train.py exactly) ─────────────
# Caesar weights: --caesar_ckpt, then $ENMA_CAESAR_CKPT, then <repo>/checkpoints/caesar/best_model.pth
_FAST_CNN_CKPT = Path(
    args.caesar_ckpt
    or os.environ.get("ENMA_CAESAR_CKPT")
    or str(REPO_ROOT / "checkpoints" / "caesar" / "best_model.pth")
)

def _build_fast_affect_cnn():
    """Build FastAffectCNN using torchvision r3d_18 backbone.

    Input to forward():  (B, T, H, W)  float32, values in [-1, 1]
    Output of forward(): (B, 2)         [valence, arousal] in approx [-1, 1]

    The model internally unsqueezes the channel dim and repeats to 3 channels
    to feed r3d_18, identical to how it was trained in fast_cnn/train.py.
    """
    from torchvision.models.video import r3d_18, R3D_18_Weights
    import torch.nn as nn

    class FastAffectCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = r3d_18(weights=None)  # weights loaded from checkpoint below
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 2)

        def forward(self, x):
            # x: (B, T, H, W) float32 in [-1, 1]
            x = x.unsqueeze(1)           # (B, 1, T, H, W)
            x = x.repeat(1, 3, 1, 1, 1)  # (B, 3, T, H, W)
            return self.backbone(x)

    return FastAffectCNN()


def _load_fast_cnn(device: torch.device) -> torch.nn.Module | None:
    """Load FastAffectCNN checkpoint. Returns None if checkpoint is missing."""
    if not _FAST_CNN_CKPT.exists():
        print(f"[FastCNN] WARNING: checkpoint not found at {_FAST_CNN_CKPT}. "
              "Affect scoring will be disabled.", flush=True)
        return None

    print(f"[FastCNN] Loading checkpoint from {_FAST_CNN_CKPT} ...", flush=True)
    model = _build_fast_affect_cnn()
    state_dict = torch.load(_FAST_CNN_CKPT, map_location="cpu", weights_only=True)
    # Checkpoint is saved as raw state_dict (see fast_cnn/train.py line 173)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("[FastCNN] Checkpoint loaded successfully.", flush=True)
    return model


def _policy_obs(obs):
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def train():
    # Create environment
    env_cfg = parse_env_cfg(
        "Isaac-ENMA-Go2-v0",
        num_envs=args.num_envs,
    )
    if not args.no_affect:
        env_cfg.scene = ENMAInteractionSceneCfg(num_envs=args.num_envs, env_spacing=15.0)
        env_cfg.use_human_pov_camera = True
    else:
        env_cfg.use_human_pov_camera = False

    # render_mode="rgb_array" is required for Isaac to expose camera tensors
    render_mode = "rgb_array" if not args.no_affect else None
    env = ENMAEnv(cfg=env_cfg, render_mode=render_mode)
    buffer = RolloutBuffer()

    # SwiftTribeENMAPlugin is kept for its score_frames() fallback path only;
    # _affect_model inside it is NOT used — we load FastCNN separately below.
    affect_plugin = SwiftTribeENMAPlugin(no_affect=args.no_affect, device="cpu")

    # Policy setup
    device = env.device
    policy = ENMAPolicy(obs_dim=12, act_dim=6, num_envs=args.num_envs, device=device)

    # ── Load Checkpoint if resuming ─────────────────────────────────────────
    if hasattr(args, 'resume') and args.resume:
        print(f"[Resume] Loading checkpoint from {args.resume} ...", flush=True)
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint)
        print("[Resume] Checkpoint loaded successfully.", flush=True)

    # ── Load FastCNN (the actual video affect model) ───────────────────────────
    fast_cnn_model: torch.nn.Module | None = None
    if not args.no_affect:
        fast_cnn_model = _load_fast_cnn(device)
        if fast_cnn_model is not None:
            import torchvision.transforms.functional as _TF

    # PPO Config
    runner_cfg = ENMAPPORunnerCfg()
    num_ppo_epochs = runner_cfg.algorithm.num_learning_epochs
    clip_eps = runner_cfg.algorithm.clip_param
    value_coef = runner_cfg.algorithm.value_loss_coef
    entropy_coef = runner_cfg.algorithm.entropy_coef
    max_grad_norm = runner_cfg.algorithm.max_grad_norm

    # Training Loop
    optimizer = torch.optim.Adam(policy.parameters(), lr=runner_cfg.algorithm.learning_rate)

    obs, _ = env.reset()
    obs = _policy_obs(obs)
    prev_action = torch.zeros(args.num_envs, 6, device=device)

    log_dir = os.path.join("ENMAmiddle", "checkpoints", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, exist_ok=True)

    # ── Persistent per-env valence/arousal state (FIXED: lives outside the loop)
    prev_sw_v_envs = [0.0] * args.num_envs
    prev_sw_a_envs = [0.0] * args.num_envs

    progress = tqdm(
        range(runner_cfg.max_iterations),
        desc="Training",
        unit="iter",
        dynamic_ncols=True,
    )
    for it in progress:
        rollout_frames = []
        for step in range(runner_cfg.num_steps_per_env):
            policy_obs = _policy_obs(obs)
            with torch.no_grad():
                actions, log_probs, value = policy.act_and_evaluate(policy_obs)

            next_obs, reward, terminated, truncated, info = env.step(actions)
            
            # --- CUSTOM ENMA AVOIDANCE REWARD ---
            # Reward moving out of the way when the human is close, especially in static scenarios
            with torch.no_grad():
                if hasattr(env, 'env_scenarios') and hasattr(env, 'human_pos'):
                    human_pos = env.human_pos
                    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
                    dist = torch.norm(human_pos - robot_pos, dim=1)
                    
                    # scenarios: 0=approach, 1=stand, 2=orbit, 3=cross, 4=navigation
                    is_static = (env.env_scenarios != 4)
                    is_close = (dist < 1.8)
                    
                    # We want the robot to move (norm of actions) when human is close
                    motion = torch.norm(actions[:, :2], dim=1)
                    
                    avoidance_reward = torch.zeros_like(reward)
                    # Large reward for explicit motion when static and human is present
                    avoidance_reward[is_static & is_close] = motion[is_static & is_close] * 5.0
                    
                    # Penalize being too close to the human (collision danger)
                    avoidance_reward[dist < 0.3] -= 8.0
                    
                    # Reward explicitly moving aside for 'cross' (3) and 'approach' (0)
                    avoidance_reward[(env.env_scenarios == 3) & is_close] += motion[(env.env_scenarios == 3) & is_close] * 5.0
                    
                    reward += avoidance_reward
            # ------------------------------------
            if not args.no_affect and fast_cnn_model is not None:
                camera = env.scene["human_pov_camera"]
                # Capture at ~10 FPS (every 4 steps)
                if step % 4 == 0:
                    rendered = camera.data.output["rgb"].clone()
                    # Drop alpha channel if present
                    if rendered.shape[-1] == 4:
                        rendered = rendered[..., :3]

                    # Check if camera is actually producing data (not all-zero)
                    if step == 0 and it == 0:
                        nonzero_frac = (rendered != 0).float().mean().item()
                        print(f"[FastCNN] Camera sanity check at iter=0 step=0: "
                              f"non-zero pixel fraction = {nonzero_frac:.4f} "
                              f"(shape={tuple(rendered.shape)})", flush=True)
                        if nonzero_frac < 1e-4:
                            print("[FastCNN] WARNING: Camera output is all-zero! "
                                  "RTX sensors may not be fully initialised yet — "
                                  "this should resolve after a few steps.", flush=True)

                    # Convert to (B, T=1, H, W) grayscale float in [-1, 1]
                    # FastAffectCNN was trained on (B, T, H, W) with values in [-1,1]
                    rendered_perm = rendered.permute(0, 3, 1, 2).float() / 255.0  # (B,3,H,W)
                    rendered_gray = rendered_perm.mean(dim=1, keepdim=False)       # (B,H,W)
                    rendered_gray = rendered_gray * 2.0 - 1.0                      # [-1,1]
                    # Resize to training resolution 180×320
                    rendered_gray_4d = rendered_gray.unsqueeze(1)                  # (B,1,H,W)
                    resized = _TF.resize(rendered_gray_4d, [180, 320])             # (B,1,180,320)
                    resized = resized.squeeze(1)                                   # (B,180,320)
                    rollout_frames.append(resized.cpu())

            next_obs = _policy_obs(next_obs)
            done = terminated | truncated

            buffer.add(
                policy_obs,
                actions,
                log_probs,
                value,
                reward,
                done
            )

            obs = next_obs
            reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
            if len(reset_ids) > 0:
                policy.reset(reset_ids)

        # ── Affect scoring with FastCNN ────────────────────────────────────────
        affect_bonus_value = 0.0
        mean_valence = 0.0
        mean_arousal = 0.0

        if not args.no_affect and rollout_frames and fast_cnn_model is not None:
            _TEMPORAL_FRAMES = 16          # Must match CAESAR_TEMPORAL_FRAMES in fast_cnn/config.py
            # With num_steps_per_env=200 and capture every 4 steps → 50 frames total.
            # Use 3 subwindows so each gets ~16 frames (exactly one clip per env).
            _N_SUBWIN        = 3
            num_envs_local   = args.num_envs

            # rollout_frames: list of (N_envs, H, W) tensors, one per captured step
            # Stack → (T_captured, N_envs, H, W), then permute → (N_envs, T_captured, H, W)
            stacked_frames = torch.stack(rollout_frames, dim=0)   # (T, N, H, W)
            num_frames_captured = stacked_frames.shape[0]
            resized = stacked_frames.permute(1, 0, 2, 3)          # (N, T, H, W)

            _subwin_sim_steps = runner_cfg.num_steps_per_env // _N_SUBWIN
            _frames_per_subwin = max(1, num_frames_captured // _N_SUBWIN)

            print(f"[FastCNN|it={it:04d}] frames_captured={num_frames_captured} "
                  f"subwindows={_N_SUBWIN} frames_per_sw={_frames_per_subwin} "
                  f"temporal_frames={_TEMPORAL_FRAMES}", flush=True)

            all_subwin_bonuses = []
            sw_v_prev = list(prev_sw_v_envs)
            sw_a_prev = list(prev_sw_a_envs)

            for sw in range(_N_SUBWIN):
                frame_start = sw * _frames_per_subwin
                frame_end = (frame_start + _frames_per_subwin
                             if sw < _N_SUBWIN - 1 else num_frames_captured)
                sw_frames = resized[:, frame_start:frame_end, :, :]  # (N, sw_T, H, W)
                sw_T = sw_frames.shape[1]

                # Build non-overlapping 16-frame clips per env.
                # If sw_T < _TEMPORAL_FRAMES, pad by repeating the last frame so
                # we always produce at least one clip per env instead of zero.
                flat_clips = []
                clip_mapping = []
                for env_idx in range(num_envs_local):
                    env_frames = sw_frames[env_idx]  # (sw_T, H, W)
                    # Pad if needed
                    if sw_T < _TEMPORAL_FRAMES:
                        pad_count = _TEMPORAL_FRAMES - sw_T
                        last_frame = env_frames[-1:].expand(pad_count, -1, -1)
                        env_frames_padded = torch.cat([env_frames, last_frame], dim=0)
                    else:
                        env_frames_padded = env_frames
                    padded_T = env_frames_padded.shape[0]
                    ci = 0
                    while ci + _TEMPORAL_FRAMES <= padded_T:
                        flat_clips.append(env_frames_padded[ci:ci + _TEMPORAL_FRAMES])  # (16, H, W)
                        clip_mapping.append(env_idx)
                        ci += _TEMPORAL_FRAMES

                sw_v = [0.0] * num_envs_local
                sw_a = [0.0] * num_envs_local

                if flat_clips:
                    # FastAffectCNN expects (B, T, H, W) float32 in [-1, 1]
                    batch_size = 16
                    all_preds = []
                    with torch.no_grad():
                        for i in range(0, len(flat_clips), batch_size):
                            chunk = torch.stack(flat_clips[i:i + batch_size]).to(device)  # (b, 16, H, W)
                            preds = fast_cnn_model(chunk)                                  # (b, 2)
                            all_preds.append(preds.cpu())

                    stacked_preds = torch.cat(all_preds, dim=0)  # (total_clips, 2)

                    env_valence_sums  = [0.0] * num_envs_local
                    env_arousal_sums  = [0.0] * num_envs_local
                    env_clip_counts   = [0]   * num_envs_local

                    for idx, env_idx in enumerate(clip_mapping):
                        env_valence_sums[env_idx] += stacked_preds[idx, 0].item()
                        env_arousal_sums[env_idx] += stacked_preds[idx, 1].item()
                        env_clip_counts[env_idx]  += 1

                    for env_idx in range(num_envs_local):
                        if env_clip_counts[env_idx] > 0:
                            sw_v[env_idx] = env_valence_sums[env_idx] / env_clip_counts[env_idx]
                            sw_a[env_idx] = env_arousal_sums[env_idx] / env_clip_counts[env_idx]

                    # Per-clip diagnostics every 10 iters
                    if it % 10 == 0 and sw == 0:
                        v_arr = stacked_preds[:, 0]
                        a_arr = stacked_preds[:, 1]
                        print(f"[FastCNN|it={it:04d}|sw=0] clips={len(flat_clips)} "
                              f"V: mean={v_arr.mean():.4f} std={v_arr.std():.4f} "
                              f"  A: mean={a_arr.mean():.4f} std={a_arr.std():.4f}", flush=True)

                sim_t_start = sw * _subwin_sim_steps
                sim_t_end   = (sim_t_start + _subwin_sim_steps
                               if sw < _N_SUBWIN - 1 else runner_cfg.num_steps_per_env)
                sw_bonus_envs = torch.zeros(num_envs_local, device=device)

                for env_idx in range(num_envs_local):
                    curr_v = sw_v[env_idx]
                    curr_a = sw_a[env_idx]
                    prev_v = prev_sw_v_envs[env_idx] if sw == 0 else sw_v_prev[env_idx]
                    prev_a = prev_sw_a_envs[env_idx] if sw == 0 else sw_a_prev[env_idx]
                    absolute_component   = curr_v + curr_a
                    derivation_component = (curr_v - prev_v) + (curr_a - prev_a)
                    raw_bonus = 0.25 * (absolute_component + derivation_component)
                    sw_bonus_envs[env_idx] = raw_bonus * 1.5

                all_subwin_bonuses.append(sw_bonus_envs.mean().item())
                bonus_tensor = sw_bonus_envs.to(
                    device=buffer.rewards[sim_t_start].device,
                    dtype=buffer.rewards[sim_t_start].dtype,
                )
                for t in range(sim_t_start, sim_t_end):
                    buffer.rewards[t] = buffer.rewards[t] + bonus_tensor

                sw_v_prev = list(sw_v)
                sw_a_prev = list(sw_a)

            # Persist across iterations
            for env_idx in range(num_envs_local):
                prev_sw_v_envs[env_idx] = sw_v_prev[env_idx]
                prev_sw_a_envs[env_idx] = sw_a_prev[env_idx]

            affect_bonus_value = float(np.mean(all_subwin_bonuses))
            mean_valence = float(np.mean(sw_v_prev)) if sw_v_prev else 0.0
            mean_arousal = float(np.mean(sw_a_prev)) if sw_a_prev else 0.0

            # Detailed per-env V/A print every 5 iters
            if it % 5 == 0:
                v_vals = [f"{v:+.3f}" for v in sw_v_prev[:8]]
                a_vals = [f"{a:+.3f}" for a in sw_a_prev[:8]]
                print(f"[Affect|it={it:04d}] Per-env Valence (first 8): {v_vals}", flush=True)
                print(f"[Affect|it={it:04d}] Per-env Arousal (first 8): {a_vals}", flush=True)

        elif not args.no_affect and (not rollout_frames):
            print(f"[Affect|it={it:04d}] WARNING: no camera frames collected this iteration "
                  "(rollout_frames is empty — camera may not be ready yet).", flush=True)

        # ── Clear per-iteration frame buffer ──────────────────────────────────
        rollout_frames = []

        # ── Print affect metrics ──────────────────────────────────────────────
        print(f"\n[Iter {it:04d}/{runner_cfg.max_iterations:04d}] "
              f"Valence: {mean_valence:+.4f} | Arousal: {mean_arousal:+.4f} | "
              f"Bonus: {affect_bonus_value:.4f}", flush=True)

        # ── Diagnostic print ──────────────────────────────────────────────────
        print(f"--- DIAGNOSTICS ---", flush=True)
        print(f"obs: nan={torch.isnan(obs).any().item()}, inf={torch.isinf(obs).any().item()}, "
              f"min={obs.min().item():.3f}, max={obs.max().item():.3f}", flush=True)
        raw_critic = policy.critic(_policy_obs(obs))
        print(f"raw_critic: nan={torch.isnan(raw_critic).any().item()}, "
              f"inf={torch.isinf(raw_critic).any().item()}, "
              f"min={raw_critic.min().item():.3f}, max={raw_critic.max().item():.3f}", flush=True)
        buf_vals = torch.stack(buffer.values)
        buf_rews = torch.stack(buffer.rewards)
        print(f"buf_vals: nan={torch.isnan(buf_vals).any().item()}, "
              f"inf={torch.isinf(buf_vals).any().item()}, "
              f"min={buf_vals.min().item():.3f}, max={buf_vals.max().item():.3f}", flush=True)
        print(f"buf_rews: nan={torch.isnan(buf_rews).any().item()}, "
              f"inf={torch.isinf(buf_rews).any().item()}, "
              f"min={buf_rews.min().item():.3f}, max={buf_rews.max().item():.3f}", flush=True)

        # one extra value estimate for the final state, needed for GAE bootstrap
        with torch.no_grad():
            last_value = policy.critic(policy._move_obs(obs)).squeeze(-1).detach().cpu()

        # ── Compute advantages ─────────────────────────────────────────────────
        returns, advantages = compute_gae(buffer, last_value, gamma=0.99, lam=0.95)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO Update ─────────────────────────────────────────────────────────
        for epoch in range(num_ppo_epochs):
            for batch in buffer.minibatches(returns, advantages, device=device):
                policy.reset()
                new_log_prob, new_value, entropy = policy.evaluate(batch.obs, batch.action)

                ratio         = torch.exp(new_log_prob - batch.log_prob)
                clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                policy_loss   = -torch.min(ratio * batch.advantage,
                                           clipped_ratio * batch.advantage).mean()
                value_loss    = F.mse_loss(new_value, batch.return_)
                loss          = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

                print(f"policy_loss={policy_loss.item():.4f} "
                      f"value_loss={value_loss.item():.4f} "
                      f"entropy={entropy.mean().item():.4f} "
                      f"ratio_max={ratio.max().item():.4f} "
                      f"ratio_min={ratio.min().item():.4f} "
                      f"adv_mean={advantages.mean().item():.4f} "
                      f"adv_std={advantages.std().item():.4f}", flush=True)
                print(f"  new_value: min={new_value.min().item():.3f} "
                      f"max={new_value.max().item():.3f} "
                      f"mean={new_value.mean().item():.3f}", flush=True)
                print(f"  return_:   min={batch.return_.min().item():.3f} "
                      f"max={batch.return_.max().item():.3f} "
                      f"mean={batch.return_.mean().item():.3f}", flush=True)

                optimizer.zero_grad()
                loss.backward()
                for name, param in policy.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        raise RuntimeError(f"NaN gradient detected in {name} at iteration {it}")
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
                optimizer.step()

        buffer.clear()

        # Save checkpoint every 50 iterations and at the end
        if (it + 1) % 50 == 0 or (it + 1) == runner_cfg.max_iterations:
            ckpt_path = os.path.join(log_dir, f"model_{it+1}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model to {ckpt_path}", flush=True)

        progress.set_postfix(
            {
                "loss":   f"{loss.item():.3f}",
                "steps":  runner_cfg.num_steps_per_env,
                "envs":   args.num_envs,
                "affect": f"{affect_bonus_value:.3f}",
                "V":      f"{mean_valence:+.3f}",
                "A":      f"{mean_arousal:+.3f}",
            }
        )

    env.close()


if __name__ == "__main__":
    train()
