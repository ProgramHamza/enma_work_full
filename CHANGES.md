# Changes relative to the original code package

The files in this repository were copied from the original package (`ENMA_publish.zip`). Most are byte-identical
copies. The files below were edited so the code no longer depends on the lab cluster's directory layout. The
exact differences are saved as unified diffs in [`docs/patches/`](docs/patches/).

None of the edits changes the algorithm, the reward, the network or any numeric setting that was in effect during
training. The edits have been checked for syntax and undefined names only; they have **not** been run, because
Isaac Lab is not available on the machine where this repository was assembled.

| File | Change | Why |
|---|---|---|
| `ENMAmiddle/training/train_enma.py` | New `--caesar_ckpt` argument (fallback `$ENMA_CAESAR_CKPT`, then `checkpoints/caesar/best_model.pth`) instead of a fixed cluster path | Caesar weights could only be found on the cluster |
| | `unitree_rl_lab` location can be set with `$UNITREE_RL_LAB_SRC` | Same reason |
| | Removed three start-up debug prints and the one-off camera screenshot block that wrote to a cluster path | Debug leftovers |
| | Gradient clipping now reads `max_grad_norm` from the config instead of a literal `0.5` | Keeps config and loop consistent (value unchanged: 0.5) |
| `ENMAmiddle/training/enma_ppo_cfg.py` | `max_grad_norm` 1.0 -> 0.5; removed `schedule = "adaptive"` | The loop always clipped at 0.5 and never used a scheduler, so the config was misleading |
| `ENMAmiddle/isaaclab_env/enma_env_cfg.py` | Low-level policy and human avatar paths come from `$ENMA_LL_POLICY`, `$ENMA_HUMAN_USD` or `$ENMA_ROOT` (defaults inside this repository) instead of a list of cluster locations | Environment could only start on the cluster |
| `ENMAmiddle/rewards/swifttribe_enma_plugin.py` | Teacher model root from `$ENMA_TEACHER_ROOT` instead of a fixed cluster path | Same reason |
| `evaluation/run_evaluation.py` | Docstring corrected (six configurations, new file name); new arguments `--ll_policy`, `--enma_return_ckpt`, `--enma_affect_ckpt`, `--episodes`, `--out_dir`; paths resolved relative to this repository | Paths pointed at the cluster; the header described four controllers |
| | Removed a third checkpoint (`model_it0800.pt`) that was loaded and required to exist but never evaluated | It blocked running without that file |
| `evaluation/plot_metrics.py` | New arguments `--ckpt-root` and `--logs` instead of fixed cluster paths | Paths pointed at the cluster |
| `caesar/test_fastcnn.py` | Checkpoint path from the command line, `$ENMA_CAESAR_CKPT` or `checkpoints/caesar/best_model.pth` | Path pointed at the cluster |
| `teacher_pipeline/data/build_dataset.py`, `video_processor.py` | Default data directories are now relative (`data/...`); help text no longer names a cluster path | Paths pointed at the cluster |

New files: all `README.md` files, `docs/`, `requirements.txt`, `.gitignore`, `CHANGES.md`, and
`ENMAmiddle/isaaclab_env/register_envs_template.py` (a reconstructed template, see the note inside it).

Not changed: `caesar/snn_variant/*` (kept as an archive of the earlier SNN trainer),
`ENMAmiddle/deploy/enma_inference_node.py`, and the `/mnt` <-> `/srv` mount-fallback helpers in two files, which
are harmless outside the cluster.
