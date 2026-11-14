#!/usr/bin/env python3
"""
enma_inference_node.py
----------------------
Subscribes to the Unitree Go2 remote controller topic (/wirelesscontroller),
runs the controller stick inputs through the trained ENMAPolicy (middle layer),
and publishes the 6D affectionate motion commands to /cmd_vel (geometry_msgs/Twist).

Topic wiring:
  SUB  /wirelesscontroller  [unitree_go/msg/WirelessController]
         lx = left stick X  (strafe left/right)
         ly = left stick Y  (forward/backward)  <- mapped to vx
         rx = right stick X (yaw)
         ry = right stick Y (unused / pitch)
         keys = button bitmask

  PUB  /cmd_vel             [geometry_msgs/Twist]
         linear.x  = vx     (forward velocity)
         linear.y  = vy     (lateral velocity)
         linear.z  = yaw    (static body yaw, no legs)
         angular.x = roll
         angular.y = pitch
         angular.z = vyaw   (turning yaw, with legs)

The ENMAPolicy obs vector (12-dim) is constructed as:
  [vx_cmd, vy_cmd, yawrate_cmd,   <- 3 from controller
   0,0,0,0,0,0,0,0,0]             <- 9 zeros (gravity/joint placeholders at deploy)

At inference time we use the policy mean (no sampling) for deterministic deployment.
"""

import sys
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Add ENMAPolicy to path — copied locally to ~/enma_deploy/model/
import sys, os
from pathlib import Path
_ENMA_MODEL_DIR = Path.home() / "enma_deploy"
if str(_ENMA_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_ENMA_MODEL_DIR))

import torch
from model.enma_policy import ENMAPolicy

try:
    from unitree_go.msg import WirelessController
    _HAS_WIRELESS = True
except ImportError:
    _HAS_WIRELESS = False

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = Path.home() / "enma_deploy" / "model_4650.pt"

# Velocity scaling: controller sticks output [-1, 1], scale to m/s / rad/s
VX_SCALE    = 0.5   # max forward speed  (m/s)
VY_SCALE    = 0.3   # max lateral speed  (m/s)
YAW_SCALE   = 0.8   # max yaw rate       (rad/s)

# Output clamping (safety)
OUT_LIN_MAX  = 1.0  # m/s
OUT_ANG_MAX  = 0.6  # rad or rad/s

PUBLISH_RATE = 20.0  # Hz
OBS_DIM      = 12
ACT_DIM      = 6
# ─────────────────────────────────────────────────────────────────────────────


class ENMAInferenceNode(Node):
    def __init__(self):
        super().__init__("enma_inference_node")

        # ── Load model ───────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading ENMAPolicy from {MODEL_PATH} on {self.device}")

        self.policy = ENMAPolicy(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            num_envs=1,
            device=self.device,
        )
        state_dict = torch.load(str(MODEL_PATH), map_location=self.device, weights_only=True)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        self.get_logger().info("ENMAPolicy loaded successfully.")

        # ── State ────────────────────────────────────────────────────────────
        self.vx_cmd   = 0.0
        self.vy_cmd   = 0.0
        self.yaw_cmd  = 0.0
        self.active   = True   # set False if L2 held (deadman switch via keys bitmask)

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        if _HAS_WIRELESS:
            self.ctrl_sub = self.create_subscription(
                WirelessController,
                "/wirelesscontroller",
                self.controller_callback,
                10,
            )
            self.get_logger().info("Subscribed to /wirelesscontroller (unitree_go/WirelessController)")
        else:
            self.get_logger().warn(
                "unitree_go not found — cannot subscribe to /wirelesscontroller. "
                "Will publish zero commands. Source aisl_ros2_ws first."
            )

        self.timer = self.create_timer(1.0 / PUBLISH_RATE, self.inference_step)
        self.get_logger().info(
            f"ENMA Inference Node running at {PUBLISH_RATE} Hz. "
            "Use left stick (ly=vx, lx=vy) and right stick (rx=yaw)."
        )

    # ── Controller callback ──────────────────────────────────────────────────
    def controller_callback(self, msg):
        """
        Unitree Go2 controller layout (same as DualShock / Xbox):
          ly (+1 = up  = forward)  -> vx
          lx (+1 = right)          -> vy  (negated: push right = strafe right)
          rx (+1 = right)          -> yaw rate (negated: push right = turn right)
          ry (+1 = up)             -> unused (pitch handled by policy)

        keys bitmask (uint16):
          bit 1  (0x0002) = L2  — used as deadman; hold to enable output
          bit 4  (0x0010) = R2  — emergency stop when held
        """
        R2_MASK = 0x0020
        L2_MASK = 0x0002

        # Emergency stop: R2 held → zero everything
        if msg.keys & R2_MASK:
            self.vx_cmd  = 0.0
            self.vy_cmd  = 0.0
            self.yaw_cmd = 0.0
            self.active  = False
            return

        # Deadman switch: must hold L2 to move
        self.active = bool(msg.keys & L2_MASK)

        if self.active:
            self.vx_cmd  =  msg.ly * VX_SCALE
            self.vy_cmd  = -msg.lx * VY_SCALE   # lx right => vy positive = strafe right
            self.yaw_cmd = -msg.rx * YAW_SCALE   # rx right => yaw negative = turn right
        else:
            self.vx_cmd  = 0.0
            self.vy_cmd  = 0.0
            self.yaw_cmd = 0.0

    # ── Inference step ───────────────────────────────────────────────────────
    def inference_step(self):
        # Build 12-dim observation: [vx_cmd, vy_cmd, yawrate_cmd, 0...0]
        obs_np = [self.vx_cmd, self.vy_cmd, self.yaw_cmd] + [0.0] * (OBS_DIM - 3)
        obs = torch.tensor([obs_np], dtype=torch.float32, device=self.device)  # (1, 12)

        # Run policy (deterministic: use mean, no sampling)
        with torch.no_grad():
            obs_clamped = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            obs_clamped = torch.clamp(obs_clamped, -100.0, 100.0)
            action = self.policy.actor(obs_clamped)   # (1, 6)
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            action = torch.clamp(action, -OUT_LIN_MAX, OUT_LIN_MAX)

        a = action[0].cpu().tolist()   # [vx, vy, vz, roll, pitch, yaw]

        msg = Twist()
        if self.active:
            msg.linear.x  = float(a[0])                              # vx
            msg.linear.y  = float(a[1])                              # vy
            msg.linear.z  = float(max(-OUT_ANG_MAX, min(OUT_ANG_MAX, a[2])))  # static yaw
            msg.angular.x = float(max(-OUT_ANG_MAX, min(OUT_ANG_MAX, a[3])))  # roll
            msg.angular.y = float(max(-OUT_ANG_MAX, min(OUT_ANG_MAX, a[4])))  # pitch
            msg.angular.z = float(max(-OUT_ANG_MAX, min(OUT_ANG_MAX, a[5])))  # turning yaw

        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ENMAInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ENMA inference node.")
        # Publish zero to stop robot
        stop = Twist()
        node.cmd_pub.publish(stop)
        time.sleep(0.3)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
