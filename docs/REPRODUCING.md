# Reproducing the experiments

The code was developed and run on a lab workstation cluster. The fixed cluster paths have been replaced by
command-line arguments and environment variables (see [`../CHANGES.md`](../CHANGES.md)), but the modified scripts
have not been re-run yet, and some model files are still missing. This page lists what is needed.

## 1. Software

- NVIDIA Isaac Sim with Isaac Lab (the original runs used a conda environment named `isaacsim51`).
- [`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab) (Go2 assets, locomotion `mdp` terms and
  the Go2 velocity task that `enma_env_cfg.py` extends).
- The Python packages in [`../requirements.txt`](../requirements.txt) (versions were not recorded).
- Linux (the training script imports `fcntl`).
- For the teacher pipeline: TRIBE v2 (`facebook/tribev2`) and the SwiFT model code.
- For the robot: ROS 2 with the `unitree_go` message package.

## 2. Setting up the environment

1. Clone this repository next to (or inside) an Isaac Lab checkout and install `unitree_rl_lab`.
2. Copy `ENMAmiddle/isaaclab_env/enma_env_cfg.py` to
   `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/enma_env_cfg.py`
   (the training script imports it from there).
3. Register the two task ids. The original registration code was not part of the package; a reconstructed
   template is in `ENMAmiddle/isaaclab_env/register_envs_template.py`. **Check the mapping in that file.**
4. Set the environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `ENMA_ROOT` | Root of this repository (needed because the environment file lives inside `unitree_rl_lab`) | two folders above `enma_env_cfg.py` |
| `ENMA_LL_POLICY` | Low-level walking policy | `$ENMA_ROOT/checkpoints/low_level_locomotion/policy.pt` |
| `ENMA_HUMAN_USD` | USD file of the human avatar | `$ENMA_ROOT/ENMAmiddle/assets/humans/male_adult_medical_01/male_adult_medical_01.usd` |
| `ENMA_CAESAR_CKPT` | Caesar weights | `checkpoints/caesar/best_model.pth` |
| `UNITREE_RL_LAB_SRC` | `unitree_rl_lab/source` directory | `<repo>/unitree_rl_lab/source` |
| `ENMA_TEACHER_ROOT` | Folder with `models/tribev2`, `models/swift`, `models/swiftold` | `teacher_models` |

## 3. Files that are still missing

| Needed by | File | Where to put it |
|---|---|---|
| training (affect reward) | Caesar weights `best_model.pth` | `checkpoints/caesar/` |
| evaluation | ENMA checkpoints `model_0.pt`, `model_5550.pt` (run 2026-06-29_10-35-15) | `checkpoints/enma/` |
| environment | Human avatar USD asset (`male_adult_medical_01`) | `ENMAmiddle/assets/humans/male_adult_medical_01/` |
| Caesar training | the 3D ResNet training script and dataset class | `caesar/` |
| deployment | `~/enma_deploy/model_4650.pt` (path fixed inside the node) | — |

If the Caesar weights are missing, `train_enma.py` prints a warning and continues without the affect bonus.

## 4. Commands (from the repository root)

```bash
# training (affect bonus on); checkpoints go to ENMAmiddle/checkpoints/<timestamp>/model_<iter>.pt
python ENMAmiddle/training/train_enma.py --headless --num_envs 64 --caesar_ckpt checkpoints/caesar/best_model.pth

# training without the affect bonus (ablation)
python ENMAmiddle/training/train_enma.py --headless --num_envs 64 --no_affect

# continue from a checkpoint
python ENMAmiddle/training/train_enma.py --headless --resume <path/to/model.pt>

# roll out a checkpoint and record the human's point of view
python ENMAmiddle/training/play.py --checkpoint <path/to/model.pt> --video

# evaluation of all six configurations (defaults shown)
python evaluation/run_evaluation.py --headless \
    --ll_policy checkpoints/low_level_locomotion/policy.pt \
    --enma_return_ckpt checkpoints/enma/model_0.pt \
    --enma_affect_ckpt checkpoints/enma/model_5550.pt \
    --episodes 30 --out_dir evaluation/results

# training curves from a log file
python evaluation/plot_metrics.py --ckpt-root ENMAmiddle/checkpoints --logs train_enma.log
```

## 5. Still to do for a one-command reproduction

1. Run the modified scripts once in Isaac Lab to confirm the path changes work.
2. Add the missing files listed in section 3.
3. Confirm the gymnasium registration.
4. Pin package versions.
