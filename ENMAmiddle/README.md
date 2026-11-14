# ENMAmiddle — the ENMA middleware policy

This folder is the core of the project. The folder name is the original Python package name, so the imports in the
scripts (`from ENMAmiddle.model.enma_policy import ENMAPolicy`) still resolve.

## Files

| File | Lines | What it does |
|---|---|---|
| [`model/enma_snn.py`](model/enma_snn.py) | 136 | `ENMASNN`: the spiking actor. `ENMACritic`: the value network. |
| [`model/enma_policy.py`](model/enma_policy.py) | 84 | `ENMAPolicy`: wraps actor + critic as a Gaussian policy for PPO. |
| [`training/train_enma.py`](training/train_enma.py) | 520 | Main script: PPO loop in Isaac Lab, Caesar affect bonus, checkpointing. |
| [`training/enma_ppo_cfg.py`](training/enma_ppo_cfg.py) | 31 | PPO settings (clip, epochs, learning rate, rollout length). |
| [`training/gae.py`](training/gae.py) | 53 | Generalised Advantage Estimation. |
| [`training/rollout_buffer.py`](training/rollout_buffer.py) | 90 | Rollout storage and minibatch generator (batch size 256). |
| [`training/play.py`](training/play.py) | 227 | Runs a trained checkpoint in simulation; can record the human's point-of-view video. |
| [`isaaclab_env/enma_env_cfg.py`](isaaclab_env/enma_env_cfg.py) | 863 | Isaac Lab environment: scene, human scenarios, observation, command fusion, low-level policy call. |
| [`isaaclab_env/register_envs_template.py`](isaaclab_env/register_envs_template.py) | 39 | Reconstructed gymnasium registration for the two task ids (template; verify before use). |
| [`rewards/swifttribe_enma_plugin.py`](rewards/swifttribe_enma_plugin.py) | 391 | Wrapper around the TRIBE v2 + SwiFT teacher (imported by the training script; the online reward uses Caesar). |
| [`deploy/enma_inference_node.py`](deploy/enma_inference_node.py) | 204 | ROS 2 node that runs the policy on the Unitree Go2 and publishes `/cmd_vel`. |

## The policy network (`model/enma_snn.py`)

```
obs (12) -> FC(256) -> LIF1 -> FC(256) -> LIF2 -> FC(256) -> LIF3 -> FC(256) -> LIF4
         -> mean firing rate over T = 16 timesteps (256 values in [0, 1])
         -> FC(256 -> 6), non-spiking readout (signed values)
         -> tanh x per-dimension bound
```

- LIF neurons are `snntorch.Leaky` with membrane decay `beta = 0.9` (learnable), threshold 1.0 and a fast-sigmoid
  surrogate gradient. Membrane potentials are reset at episode boundaries (`reset()`).
- Output bounds: `0.3, 0.2, 0.1, 0.3, 0.3, 0.5` for the six correction channels.
- The critic is a plain MLP `12 -> 256 -> 256 -> 128 -> 1` with ReLU.
- `ENMAPolicy` treats the actor output as the mean of a Gaussian with a learned, state-independent `log_std`
  (initialised to -0.5). Training samples from it; deployment uses the mean.

## Observation (12 values, `isaaclab_env/enma_env_cfg.py`)

| Term | Size | Meaning |
|---|---|---|
| `nav_cmd` | 3 | Nominal command from the navigation stack (vx, vy, yaw) |
| `robot_vel` | 3 | Robot base velocity (vx, vy, yaw rate) |
| `human_rel_pos` | 2 | Human position relative to the robot (x, y) |
| `human_rel_vel` | 2 | Human velocity relative to the robot (x, y) |
| `task_state` | 1 | 1.0 normally, 2.0 when the human is closer than 1.5 m |
| `human_dist` | 1 | Distance to the human |

## How the correction is applied (`_step_impl` in `enma_env_cfg.py`)

```python
final_cmd[:, :3]  = nav_cmd + action[:, :3]        # robot moving: nominal command + correction
final_cmd[:, :3]  = action[:, :3]                  # robot idle: correction only
final_cmd[:, 3:6] = action[:, 3:6] * action_scale  # action_scale = (1, 0.3, 0.3)
```

`final_cmd` is written into the command slots of the low-level observation and the pre-trained walking policy
(`checkpoints/low_level_locomotion/policy.pt`, a 48-input / 12-output MLP) produces the joint targets. With
`use_new_ll_policy = False` the walking policy receives the first three command channels; a walking policy that
accepts all six is supported through `use_new_ll_policy = True`.

## Reward (as implemented in `training/train_enma.py`)

1. **Locomotion reward** from the Isaac Lab reward manager (the `unitree_rl_lab` Go2 terms; velocity tracking,
   flat orientation and nominal joint position are set to weight 0 in `ENMAEnvCfg`). Clamped to [-50, 50].
2. **Avoidance term** added in the training loop: `+5 x |a_xy|` when the human is within 1.8 m in the
   non-navigation scenarios, an extra `+5 x |a_xy|` in the *cross* scenario, and `-8` within 0.3 m.
3. **Affect bonus** from Caesar. The human's first-person camera is captured every 4 steps, converted to
   180x320 grayscale, and the rollout is split into 3 subwindows of about one 16-frame clip each. For each
   subwindow and environment:

   ```python
   raw_bonus = 0.25 * ((V + A) + (dV + dA))   # level + change since the previous subwindow
   bonus     = raw_bonus * 1.5                # = 0.375 * [...]
   ```

   The bonus is added to every simulation step of that subwindow before GAE.

## PPO settings

From [`training/enma_ppo_cfg.py`](training/enma_ppo_cfg.py) and the training loop: 64 environments, 200 steps per
rollout, 5 epochs, minibatch 256, Adam with learning rate 3e-4, clip 0.2, value coefficient 1.0, entropy
coefficient 0.01, gamma 0.99, lambda 0.95, gradient-norm clipping at 0.5, advantage normalisation. A checkpoint
is saved every 50 iterations.

## Running

The scripts need Isaac Lab and `unitree_rl_lab`, with the environment file copied into
`unitree_rl_lab/.../robots/go2/` and registered as `Isaac-ENMA-Go2-v0` (template:
[`isaaclab_env/register_envs_template.py`](isaaclab_env/register_envs_template.py)). Paths to model files are set
with arguments or environment variables. See [`../docs/REPRODUCING.md`](../docs/REPRODUCING.md).

```bash
# from the repository root
python ENMAmiddle/training/train_enma.py --headless --num_envs 64 --caesar_ckpt checkpoints/caesar/best_model.pth
python ENMAmiddle/training/train_enma.py --headless --num_envs 64 --no_affect   # without the affect bonus
python ENMAmiddle/training/play.py --checkpoint <model.pt> --video              # roll out and record
```
