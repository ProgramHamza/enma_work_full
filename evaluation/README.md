# Evaluation

## `run_evaluation.py`

Compares six configurations in the same Isaac Lab environment (`Isaac-ENMA-Go2-v0`):

| Configuration | Planner command | ENMA correction |
|---|---|---|
| Nominal Baseline | environment's nominal command | none |
| A* Baseline | A* planner (grid 0.2 m) | none |
| ENMA (mean) + Nominal / + A* | as above | checkpoint `model_0.pt` (best total return) |
| ENMA (best) + Nominal / + A* | as above | checkpoint `model_5550.pt` (best affective reward) |

How it keeps the comparison fair:

- The random seed is fixed before **every** reset (`BASE_SEED = 1234`), so all configurations face the same episodes.
- One environment, one episode at a time, up to `MAX_STEPS = 15000` steps; a fall is detected when the base height
  drops below `FALL_H = 0.15` m; a collision is logged when the robot-human distance is below 0.2 m.
- Per episode it records waypoints reached, path length, time, velocity, falls, collisions, minimum human distance
  and maximum roll / pitch / yaw, and writes a raw CSV, a JSON summary, a comparison CSV and a Markdown table.

Usage (from the repository root; defaults point into `checkpoints/`):

```bash
python evaluation/run_evaluation.py --headless \
    --ll_policy checkpoints/low_level_locomotion/policy.pt \
    --enma_return_ckpt checkpoints/enma/model_0.pt \
    --enma_affect_ckpt checkpoints/enma/model_5550.pt \
    --episodes 30 --out_dir evaluation/results
```

The two ENMA checkpoints are not included in this repository yet (see `checkpoints/README.md`).

## `plot_metrics.py`

Plots training metrics (valence, arousal, affect bonus, losses) from a training log:
`python evaluation/plot_metrics.py --ckpt-root ENMAmiddle/checkpoints --logs train_enma.log`.

## `results_2026-07-09/`

Output of one run of `run_evaluation.py` (generated 2026-07-09, **30 episodes per configuration**, 406 waypoint
targets):

| File | Content |
|---|---|
| `real_eval_table_20260709_012100.md` | Summary table |
| `real_eval_comparison_20260709_012100.csv` | Same summary as CSV |
| `real_eval_raw_20260709_012100.csv` | One row per episode |
| `real_eval_summary_20260709_012100.json` | Full summary with per-episode details |

**Note.** The tables in the paper report a different, 15-episode run (206 waypoint targets) that also includes
Caesar's valence and arousal. The raw output of that run is not part of this repository, so the numbers here are
not expected to match the paper's tables. In this 30-episode run the simulated walking controller falls in most
episodes for every configuration (73-100 %), and no collisions were recorded.
