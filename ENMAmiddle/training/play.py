from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher

# Add custom arguments
parser = argparse.ArgumentParser(description="Run evaluation for a trained ENMA policy.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint to load.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run.")
parser.add_argument("--video", action="store_true", default=False, help="Record a POV-style MP4 while running.")
parser.add_argument("--cameratest", action="store_true", default=False, help="Run the humanoid POV camera test mode.")
parser.add_argument("--video_seconds", type=float, default=120.0, help="Target video duration in seconds.")
parser.add_argument(
    "--video_path",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "videos", "enma_pov.mp4"),
    help="Output path for the recorded MP4.",
)
AppLauncher.add_app_launcher_args(parser)
args, unknown_args = parser.parse_known_args()
if args.video or args.cameratest:
    args.enable_cameras = True
if not args.cameratest and args.checkpoint is None:
    parser.error("--checkpoint is required unless --cameratest is supplied.")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_TASK_ID = "Isaac-ENMA-Go2-v0"

# Launch the simulator
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Late imports after sim is loaded
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.math import quat_apply

# Late imports after sim is loaded
import gymnasium as gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ENMAmiddle.model.enma_policy import ENMAPolicy
from isaaclab_tasks.utils import parse_env_cfg
import unitree_rl_lab.tasks


def _create_follow_camera() -> Camera:
    camera_cfg = CameraCfg(
        prim_path="/World/ENMA_POV_Camera",
        update_period=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    return Camera(camera_cfg)


def _ensure_camera_initialized(camera: Camera) -> None:
    if getattr(camera, "is_initialized", False):
        return
    camera._initialize_impl()
    camera.reset()


def _stable_camera_target(camera_pos: torch.Tensor, target_pos: torch.Tensor, min_horizontal_offset: float = 0.35) -> torch.Tensor:
    squeeze = False
    if camera_pos.dim() == 1:
        camera_pos = camera_pos.unsqueeze(0)
        target_pos = target_pos.unsqueeze(0)
        squeeze = True

    camera_pos = torch.nan_to_num(camera_pos)
    target_pos = torch.nan_to_num(target_pos)

    horizontal = target_pos[:, :2] - camera_pos[:, :2]
    horizontal_norm = torch.linalg.norm(horizontal, dim=1, keepdim=True)
    fallback_dir = torch.tensor([1.0, 0.0], device=camera_pos.device, dtype=camera_pos.dtype).expand_as(horizontal)
    horizontal_dir = torch.where(horizontal_norm > 1e-6, horizontal / horizontal_norm.clamp(min=1e-6), fallback_dir)

    stable_target = target_pos.clone()
    stable_target[:, :2] = camera_pos[:, :2] + horizontal_dir * torch.clamp(horizontal_norm, min=min_horizontal_offset)
    stable_target[:, 2] = torch.where(
        torch.isfinite(stable_target[:, 2]),
        stable_target[:, 2],
        camera_pos[:, 2] - 0.2,
    )
    return stable_target.squeeze(0) if squeeze else stable_target


def _update_follow_camera(camera: Camera, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor) -> None:
    offset = torch.tensor([0.35, 0.0, 0.30], device=robot_pos_w.device)
    look_ahead = torch.tensor([1.75, 0.0, 0.15], device=robot_pos_w.device)
    camera_pos = robot_pos_w + quat_apply(robot_quat_w, offset)
    camera_target = robot_pos_w + quat_apply(robot_quat_w, look_ahead)
    camera_target = _stable_camera_target(camera_pos, camera_target)
    camera.set_world_poses_from_view(camera_pos.unsqueeze(0), camera_target.unsqueeze(0))


def main():
    """Run evaluation with a trained model checkpoint."""
    # Create the environment
    task_id = "Isaac-ENMA-Go2-Moving-Interaction-v0" if (args.video or args.cameratest) else DEFAULT_TASK_ID
    env_cfg = parse_env_cfg(
        task_id,
        num_envs=args.num_envs,
    )
    env = gym.make(task_id, cfg=env_cfg, render_mode=None if (args.video or args.cameratest) else "rgb_array")

    policy = None
    if args.checkpoint is not None:
        policy = ENMAPolicy(obs_dim=12, act_dim=3, num_envs=env.unwrapped.num_envs, device=env.unwrapped.device)
        checkpoint = torch.load(args.checkpoint, map_location=env.unwrapped.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        policy.load_state_dict(checkpoint)
        policy.eval()
        print(f"Successfully loaded model from: {args.checkpoint}")

    video_writer = None
    video_steps = None
    camera = None
    if args.cameratest:
        camera = env.unwrapped.scene["human_pov_camera"]
        _ensure_camera_initialized(camera)
        video_steps = max(1, int(round(60.0 / env.unwrapped.step_dt)))
        fps = max(1, int(round(1.0 / env.unwrapped.step_dt)))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(REPO_ROOT, f"testvideo_{timestamp}.mp4")
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (camera.cfg.width, camera.cfg.height))
        print(f"[INFO] Recording humanoid POV video to: {video_path}")
        print(f"[INFO] Target duration: 60.0s at {fps} fps ({video_steps} steps)")
    elif args.video:
        camera = _create_follow_camera()
        _ensure_camera_initialized(camera)
        video_steps = max(1, int(round(args.video_seconds / env.unwrapped.step_dt)))
        fps = max(1, int(round(1.0 / env.unwrapped.step_dt)))
        video_path = os.path.abspath(args.video_path)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
        print(f"[INFO] Recording POV-style video to: {video_path}")
        print(f"[INFO] Target duration: {args.video_seconds:.1f}s at {fps} fps ({video_steps} steps)")

    # Run the simulation
    obs, _ = env.reset()
    try:
        step = 0
        record_bar = None
        if video_steps is not None:
            record_bar = tqdm(
                total=video_steps,
                desc="Recording video",
                unit="frame",
                dynamic_ncols=False,
                file=sys.stdout,
                leave=True,
                mininterval=0.1,
            )

        while simulation_app.is_running():
            # Get action from the policy
            if policy is None:
                action = torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)
            else:
                if isinstance(obs, dict):
                    policy_obs = obs["policy"]
                else:
                    policy_obs = obs
                action = policy.act(policy_obs)

            # Step the environment
            obs, _, _, _, _ = env.step(action)

            if args.cameratest and camera is not None:
                camera.update(env.unwrapped.step_dt)
                frame = camera.data.output["rgb"][0].detach().cpu().numpy()
                video_writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
            elif camera is not None:
                robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
                robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
                _update_follow_camera(camera, robot_pos, robot_quat)
                camera.update(env.unwrapped.step_dt)
                frame = camera.data.output["rgb"][0].detach().cpu().numpy()
                video_writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))

            if record_bar is not None:
                record_bar.update(1)
                record_bar.refresh()
            
            # Log positions every 20 steps
            if step % 20 == 0:
                robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :2]
                human_pos = env.unwrapped.human_pos[0]
                print(f"Step {step}: Robot Pos: {robot_pos.cpu().numpy()}, Human Pos: {human_pos.cpu().numpy()}")
            
            step += 1
            if video_steps is not None and step >= video_steps:
                break
            
    except KeyboardInterrupt:
        print("[INFO] Exiting peacefully.")
    finally:
        if 'record_bar' in locals() and record_bar is not None:
            record_bar.close()
        if video_writer is not None:
            video_writer.release()
        # Close the environment
        env.close()

if __name__ == "__main__":
    main()
