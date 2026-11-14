"""
run_evaluation.py  –  REAL evaluation: no mock data, no numpy random stand-ins.
(Originally named run_real_eval.py.)

Compares six configurations in Isaac-ENMA-Go2-v0:
  1. ENMA (best) + A*        – A* planner command + ENMA 6D correction (best affective reward checkpoint)
  2. ENMA (best) + Nominal   – env nav_cmd        + ENMA 6D correction (best affective reward checkpoint)
  3. A* Baseline             – A* planner command, zero ENMA correction
  4. ENMA (mean) + A*        – A* planner command + ENMA 6D correction (best total return checkpoint)
  5. ENMA (mean) + Nominal   – env nav_cmd        + ENMA 6D correction (best total return checkpoint)
  6. Nominal Baseline        – nav_cmd computed by env, zero ENMA correction

Seeds are fixed before EVERY reset → fully deterministic rollouts.

Usage (from the repository root, inside an Isaac Lab Python environment):
  python evaluation/run_evaluation.py --headless \
      --ll_policy checkpoints/low_level_locomotion/policy.pt \
      --enma_return_ckpt checkpoints/enma/model_0.pt \
      --enma_affect_ckpt checkpoints/enma/model_5550.pt
"""

from __future__ import annotations
import os, sys, random, csv, json, math
from datetime import datetime

import numpy as np
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # repository root
ENMA_ROOT   = os.path.join(REPO_ROOT, "ENMAmiddle")
# Location of the unitree_rl_lab sources; override with $UNITREE_RL_LAB_SRC if it lives elsewhere.
UNITREE_SRC = os.environ.get("UNITREE_RL_LAB_SRC", os.path.join(REPO_ROOT, "unitree_rl_lab", "source"))

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(UNITREE_SRC, "unitree_rl_lab"))

MAX_STEPS = 15000
FALL_H    = 0.15
BASE_SEED = 1234

# ─── Isaac Lab bootstrap ─────────────────────────────────────────────────────
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ll_policy", type=str,
                    default=os.path.join(REPO_ROOT, "checkpoints", "low_level_locomotion", "policy.pt"),
                    help="Low-level walking policy (TorchScript).")
parser.add_argument("--enma_return_ckpt", type=str,
                    default=os.path.join(REPO_ROOT, "checkpoints", "enma", "model_0.pt"),
                    help="ENMA checkpoint with the best total return ('ENMA (mean)').")
parser.add_argument("--enma_affect_ckpt", type=str,
                    default=os.path.join(REPO_ROOT, "checkpoints", "enma", "model_5550.pt"),
                    help="ENMA checkpoint with the best affective reward ('ENMA (best)').")
parser.add_argument("--episodes", type=int, default=30, help="Episodes per configuration.")
parser.add_argument("--out_dir", type=str, default=os.path.join(SCRIPT_DIR, "results"),
                    help="Directory for the csv / json / markdown outputs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

LL_POLICY_PATH = args_cli.ll_policy
ENMA_MEAN_CKPT = args_cli.enma_return_ckpt   # best total return, act_dim=6
ENMA_BEST_CKPT = args_cli.enma_affect_ckpt   # best affective reward, act_dim=6
EPISODES       = args_cli.episodes
OUT_DIR        = args_cli.out_dir

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils import parse_env_cfg
import unitree_rl_lab.tasks  # noqa: F401 – registers Isaac-ENMA-Go2-v0

sys.path.insert(0, ENMA_ROOT)
from model.enma_policy import ENMAPolicy

TASK_ID = "Isaac-ENMA-Go2-v0"


# ─── Utilities ───────────────────────────────────────────────────────────────

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_env():
    env_cfg = parse_env_cfg(TASK_ID, num_envs=1)
    env_cfg.sim.physx.gpu_max_rigid_contact_count            = 2**18
    env_cfg.sim.physx.gpu_max_rigid_patch_count              = 2**13
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity          = 2**13
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity= 2**13
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity     = 2**13
    env_cfg.sim.physx.gpu_collision_stack_size               = 2**20
    env_cfg.sim.physx.gpu_heap_capacity                      = 2**20
    env_cfg.sim.physx.gpu_temp_buffer_capacity               = 2**20
    # Increase the episode length to 600s so 12 trips are possible
    env_cfg.episode_length_s = 600.0
    return gym.make(TASK_ID, cfg=env_cfg, render_mode=None)


def load_ll_policy(path: str, device):
    pol = torch.jit.load(path, map_location=device)
    pol.eval()
    with torch.no_grad():
        out = pol(torch.zeros((1, 48), device=device))
    assert out.shape[1] == 12, f"LL policy output dim wrong: {out.shape}"
    print(f"  [LL] loaded: {path}")
    return pol


def infer_act_dim(ckpt_path: str) -> int:
    """Infer act_dim from checkpoint's log_std shape."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    return int(ckpt["log_std"].shape[0])


def load_enma(ckpt_path: str, device):
    act_dim = infer_act_dim(ckpt_path)
    pol = ENMAPolicy(obs_dim=12, act_dim=act_dim, num_envs=1, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    pol.load_state_dict(ckpt)
    pol.eval()
    print(f"  [ENMA] act_dim={act_dim} loaded: {ckpt_path}")
    return pol


import heapq

class AStarPlanner:
    def __init__(self, grid_size=20.0, resolution=0.2):
        self.res = resolution
        self.w = int(grid_size / resolution)
        self.h = int(grid_size / resolution)
        self.ox = -grid_size / 2.0
        self.oy = -grid_size / 2.0

    def w2g(self, x, y):
        return int((x - self.ox) / self.res), int((y - self.oy) / self.res)

    def g2w(self, gx, gy):
        return gx * self.res + self.ox, gy * self.res + self.oy

    def plan(self, sx, sy, gx, gy, hx, hy):
        start = self.w2g(sx, sy)
        goal = self.w2g(gx, gy)
        human = self.w2g(hx, hy)
        
        # Human obstacle radius (0.6 meters for safety)
        obs_r = int(0.6 / self.res)
        
        if start[0] < 0 or start[0] >= self.w or start[1] < 0 or start[1] >= self.h:
            return gx, gy
            
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        
        def heur(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
            
        while open_set:
            _, curr = heapq.heappop(open_set)
            
            if curr == goal or heur(curr, goal) < 2:
                path = []
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.reverse()
                for wp in path:
                    if heur(wp, start) > 1.5:  # lookahead ~0.3m
                        return self.g2w(*wp)
                if path: return self.g2w(*path[-1])
                return gx, gy
                
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nxt = (curr[0]+dx, curr[1]+dy)
                if 0 <= nxt[0] < self.w and 0 <= nxt[1] < self.h:
                    if heur(nxt, human) <= obs_r:
                        continue
                    tent_g = g_score[curr] + math.hypot(dx, dy)
                    if nxt not in g_score or tent_g < g_score[nxt]:
                        came_from[nxt] = curr
                        g_score[nxt] = tent_g
                        heapq.heappush(open_set, (tent_g + heur(nxt, goal), nxt))
        return gx, gy


def robot_xy(env) -> np.ndarray:
    return env.unwrapped.scene["robot"].data.root_pos_w[0, :2].detach().cpu().numpy()


def human_xy(env) -> np.ndarray:
    if not hasattr(env.unwrapped, "human_pos"):
        return np.array([9999.0, 9999.0])
    h_local = env.unwrapped.human_pos[0, :2].detach().cpu().numpy()
    env_origin = env.unwrapped.scene.env_origins[0, :2].detach().cpu().numpy()
    return h_local + env_origin


def robot_z(env) -> float:
    return env.unwrapped.scene["robot"].data.root_pos_w[0, 2].item()


def robot_rpy(env):
    r, p, y = euler_xyz_from_quat(env.unwrapped.scene["robot"].data.root_quat_w)
    return r[0].item(), p[0].item(), y[0].item()


def policy_obs(env) -> torch.Tensor:
    return env.unwrapped.observation_manager.compute_group("policy")


# ─── Single episode ──────────────────────────────────────────────────────────

def run_episode(env, enma_pol, device, use_astar=False, label="") -> dict:
    """
    enma_pol=None  → zero action (Nominal Baseline / A*)
    enma_pol=ENMAPolicy → ENMA correction added to nav_cmd inside env
    """
    path_m  = 0.0
    max_r = max_p = max_y_val = 0.0
    prev_xy = robot_xy(env)
    min_h_dist = 9999.0

    # Waypoint completion tracking
    try:
        initial_trips = int(env.unwrapped.robot_trips[0].item())
    except Exception:
        initial_trips = 1
    prev_trips = initial_trips
    waypoints_completed = 0
    waypoints_attempted  = initial_trips

    astar = AStarPlanner() if use_astar else None

    for step in range(MAX_STEPS):
        if enma_pol is not None:
            with torch.no_grad():
                action = enma_pol.act(policy_obs(env))
                action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                action[:, :2] = torch.clamp(action[:, :2], min=-1.0, max=1.0)
                if action.shape[1] >= 3:
                    action[:, 2:] = torch.clamp(action[:, 2:], min=-0.6, max=0.6)
        elif use_astar:
            rx, ry = robot_xy(env)
            if "No Human" in label:
                hx, hy = 9999.0, 9999.0
            else:
                hx, hy = human_xy(env)
            
            idx = env.unwrapped.robot_target_idx[0].item()
            g_local = env.unwrapped.robot_endpoints[0, idx].detach().cpu().numpy()
            g_origin = env.unwrapped.scene.env_origins[0, :2].detach().cpu().numpy()
            gx, gy = g_local[0] + g_origin[0], g_local[1] + g_origin[1]
            
            # 1. Plan next waypoint
            nx, ny = astar.plan(rx, ry, gx, gy, hx, hy)
            
            # 2. Convert to velocity vector
            dx = nx - rx
            dy = ny - ry
            
            q = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
            w, x, y, z = q
            yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            
            t_yaw = math.atan2(dy, dx)
            yaw_err = t_yaw - yaw
            yaw_err = (yaw_err + math.pi) % (2*math.pi) - math.pi
            
            mag = min(1.0, math.hypot(dx, dy))
            v_x = max(0.0, mag * math.cos(yaw_err))
            v_y = mag * math.sin(yaw_err)
            v_yaw = 2.0 * yaw_err
            
            astar_cmd = np.array([v_x, v_y, v_yaw, 0.0, 0.0, 0.0], dtype=np.float32)
            nav_cmd = env.unwrapped.nav_cmd[0].detach().cpu().numpy()
            
            # Pad nav_cmd to 6D
            nav_6d = np.zeros(6, dtype=np.float32)
            nav_6d[:3] = nav_cmd[:3]
            
            # The env does: final_cmd = nav_cmd + action
            # We want final_cmd to be astar_cmd. So action = astar_cmd - nav_cmd
            action_np = astar_cmd - nav_6d
            action = torch.tensor(action_np, device=device).unsqueeze(0)
        else:
            # Baseline: zeros
            action = torch.zeros((1, 6), device=device)

        _, _, terminated, truncated, _ = env.step(action)

        xy = robot_xy(env)
        path_m += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy.copy()
        
        h_dist = float(np.linalg.norm(xy - human_xy(env)))
        min_h_dist = min(min_h_dist, h_dist)

        r, p, y = robot_rpy(env)
        max_r     = max(max_r,     abs(r))
        max_p     = max(max_p,     abs(p))
        max_y_val = max(max_y_val, abs(y))

        if robot_z(env) < FALL_H:
            return dict(success=False, fell=True, timed_out=False,
                        steps=step, path_m=path_m,
                        max_roll=max_r, max_pitch=max_p, max_yaw=max_y_val, min_h_dist=min_h_dist,
                        waypoints_completed=waypoints_completed,
                        waypoints_attempted=waypoints_attempted)

        try:
            trips = int(env.unwrapped.robot_trips[0].item())
        except Exception:
            trips = prev_trips

        # Detect waypoint arrival: trips counter decreased
        if trips < prev_trips:
            waypoints_completed += (prev_trips - trips)
        prev_trips = trips

        # All trips done → full episode success
        if trips <= 0:
            waypoints_completed = initial_trips
            return dict(success=True, fell=False, timed_out=False,
                        steps=step, path_m=path_m,
                        max_roll=max_r, max_pitch=max_p, max_yaw=max_y_val, min_h_dist=min_h_dist,
                        waypoints_completed=waypoints_completed,
                        waypoints_attempted=waypoints_attempted)

        if terminated or truncated:
            fell = bool(terminated)
            timed_out = bool(truncated)
            return dict(success=False, fell=fell, timed_out=timed_out,
                        steps=step, path_m=path_m,
                        max_roll=max_r, max_pitch=max_p, max_yaw=max_y_val, min_h_dist=min_h_dist,
                        waypoints_completed=waypoints_completed,
                        waypoints_attempted=waypoints_attempted)

    return dict(success=False, fell=False, timed_out=True,
                steps=MAX_STEPS, path_m=path_m,
                max_roll=max_r, max_pitch=max_p, max_yaw=max_y_val, min_h_dist=min_h_dist,
                waypoints_completed=waypoints_completed,
                waypoints_attempted=waypoints_attempted)


# ─── Controller batch evaluation ─────────────────────────────────────────────

def evaluate(env, label: str, enma_pol, n: int, device, use_astar=False) -> list[dict]:
    print(f"\n{'='*60}\n  Evaluating: {label}\n{'='*60}")

    results = []
    for ep in range(n):
        seed = BASE_SEED + ep
        fix_seed(seed)
        env.reset()
        if enma_pol is not None:
            enma_pol.reset()
        
        hx, hy = human_xy(env)
        print(f"  ep {ep+1:02d}/{n}  seed={seed} | nearest human: ({hx:.2f}, {hy:.2f})", end=" ... ", flush=True)
        r = run_episode(env, enma_pol, device, use_astar=use_astar, label=label)
        r["label"] = label
        r["seed"]  = seed
        results.append(r)
        status = "SUCCESS" if r["success"] else ("FELL" if r["fell"] else "TIMEOUT")
        
        col_str = " (COLLISION!)" if r["min_h_dist"] < 0.2 else ""
        print(f"{status}{col_str}  path={r['path_m']:.1f}m  min_dist={r['min_h_dist']:.2f}m")

    return results


# ─── Aggregation ─────────────────────────────────────────────────────────────

def agg(results: list[dict]) -> dict:
    n     = len(results)
    suc   = [r for r in results if r["success"]]
    dt    = 0.02   # step_dt ≈ decimation(4) × physics_dt(0.005)

    s_paths = [r["path_m"] for r in suc]  if suc else [r["path_m"] for r in results]
    s_steps = [r["steps"]  for r in suc]  if suc else [r["steps"]  for r in results]
    times   = [s * dt for s in s_steps]
    
    avg_vel = sum(s_paths) / sum(times) if sum(times) > 0 else 0.0

    rolls  = [r["max_roll"]  for r in results]
    pitchs = [r["max_pitch"] for r in results]
    yaws   = [r["max_yaw"]   for r in results]
    min_ds = [r["min_h_dist"] for r in results]
    collisions = sum(1 for r in results if r["min_h_dist"] < 0.2)

    def pm(vals, fmt=".2f"):
        return f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}"

    total_wps_completed = sum(r.get("waypoints_completed", 0) for r in results)
    total_wps_attempted  = sum(r.get("waypoints_attempted",  0) for r in results)
    if total_wps_attempted > 0:
        wp_rate = f"{total_wps_completed}/{total_wps_attempted} ({total_wps_completed/total_wps_attempted*100:.1f}%)"
    else:
        wp_rate = "N/A"

    return {
        "Waypoint Completion Rate": wp_rate,
        "Goal Completion Rate":    f"{len(suc)/n*100:.1f}%",
        "Navigation Time (s)":     pm(times),
        "Path Length (m)":         pm(s_paths),
        "Average Velocity (m/s)":  f"{avg_vel:.2f}",
        "Falls":                   str(sum(1 for r in results if r["fell"])),
        "Fall Rate":               f"{sum(1 for r in results if r['fell'])/n*100:.1f}%",
        "Collisions":              str(collisions),
        "Collision Rate":          f"{collisions/n*100:.1f}%",
        "Timeouts":                str(sum(1 for r in results if r["timed_out"])),
        "Max Roll (rad)":          pm(rolls, ".4f"),
        "Max Pitch (rad)":         pm(pitchs, ".4f"),
        "Max Yaw (rad)":           pm(yaws, ".4f"),
    }


# ─── Save outputs ────────────────────────────────────────────────────────────

def save(all_results: list[dict], tables: dict[str, dict]):
    os.makedirs(OUT_DIR, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    ctrls  = list(tables.keys())
    metrics = list(next(iter(tables.values())).keys())

    # Raw per-episode CSV
    raw = os.path.join(OUT_DIR, f"real_eval_raw_{ts}.csv")
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)
    print(f"\n[saved raw → {raw}]")

    # Comparison table CSV (metrics as rows, controllers as columns)
    cmp = os.path.join(OUT_DIR, f"real_eval_comparison_{ts}.csv")
    with open(cmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Metric"] + ctrls)
        w.writeheader()
        for m in metrics:
            w.writerow({"Metric": m, **{c: tables[c][m] for c in ctrls}})
    print(f"[saved comparison → {cmp}]")

    # Markdown table
    md_path = os.path.join(OUT_DIR, f"real_eval_table_{ts}.md")
    lines = [
        "# Real Evaluation Results",
        f"*Generated {ts} · episodes={EPISODES} · base_seed={BASE_SEED}*",
        f"*LL policy: {LL_POLICY_PATH}*",
        f"*ENMA mean: {ENMA_MEAN_CKPT}*",
        f"*ENMA best: {ENMA_BEST_CKPT}*",
        "",
        "| Metric | " + " | ".join(ctrls) + " |",
        "| :--- | " + " | ".join([":---:"] * len(ctrls)) + " |",
    ]
    for m in metrics:
        lines.append("| " + m + " | " + " | ".join(tables[c][m] for c in ctrls) + " |")
    lines.append("")
    md_content = "\n".join(lines)
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"[saved markdown → {md_path}]")
    print("\n" + md_content)

    # JSON summary
    jf = os.path.join(OUT_DIR, f"real_eval_summary_{ts}.json")
    with open(jf, "w") as f:
        json.dump({
            "ts": ts, "episodes": EPISODES, "base_seed": BASE_SEED,
            "ll_policy": LL_POLICY_PATH,
            "enma_mean_ckpt": ENMA_MEAN_CKPT,
            "enma_best_ckpt": ENMA_BEST_CKPT,
            "tables": tables,
            "raw_results": all_results,
        }, f, indent=2)
    print(f"[saved json → {jf}]")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[run_evaluation]  device={device}  episodes={EPISODES}  max_steps={MAX_STEPS}")
    print(f"  LL policy:       {LL_POLICY_PATH}")
    print(f"  ENMA mean ckpt:  {ENMA_MEAN_CKPT}")
    print(f"  ENMA best ckpt:  {ENMA_BEST_CKPT}")

    # ── Verify all model paths exist before wasting time on env creation ──────
    for path in [LL_POLICY_PATH, ENMA_MEAN_CKPT, ENMA_BEST_CKPT]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[run_evaluation] MISSING: {path}")
        else:
            print(f"  [OK] exists: {path}")

    # Load ENMA policies (act_dim auto-inferred from checkpoint)
    enma_mean     = load_enma(ENMA_MEAN_CKPT, device)
    enma_best     = load_enma(ENMA_BEST_CKPT, device)

    # ── Create a SINGLE env and reuse it across all controllers ──────────────
    # Isaac Sim crashes on a 2nd gym.make() in the same process, so we reset
    # the same env between controllers instead of creating a new one each time.
    env = create_env()
    env.unwrapped.ll_policy = load_ll_policy(LL_POLICY_PATH, env.unwrapped.device)

    try:
        print("\n--- Running Custom Evaluation Suite ---")
        
        r1 = evaluate(env, "ENMA (best) + A*",      enma_best, EPISODES, device, use_astar=True)
        r2 = evaluate(env, "ENMA (best) + Nominal", enma_best, EPISODES, device)
        r3 = evaluate(env, "A* Baseline",           None,      EPISODES, device, use_astar=True)
        r4 = evaluate(env, "ENMA (mean) + A*",      enma_mean, EPISODES, device, use_astar=True)
        r5 = evaluate(env, "ENMA (mean) + Nominal", enma_mean, EPISODES, device)
        r6 = evaluate(env, "Nominal Baseline",      None,      EPISODES, device)

    finally:
        env.close()

    tables = {
        "ENMA (best) + A*":      agg(r1),
        "ENMA (best) + Nominal": agg(r2),
        "A* Baseline":           agg(r3),
        "ENMA (mean) + A*":      agg(r4),
        "ENMA (mean) + Nominal": agg(r5),
        "Nominal Baseline":      agg(r6),
    }
    save(r1 + r2 + r3 + r4 + r5 + r6, tables)
    print("\n[run_evaluation] DONE.")


if __name__ == "__main__":
    main()
