import math
import torch
import os
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import EventTermCfg, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
    COBBLESTONE_ROAD_CFG,
)

# ── File locations (override with environment variables) ─────────────────────
# ENMA_ROOT      : repository root. Set it when this file is copied into unitree_rl_lab.
# ENMA_HUMAN_USD : USD file of the human avatar.
# ENMA_LL_POLICY : exported low-level walking policy (TorchScript, 48 -> 12).
ENMA_ROOT = os.environ.get(
    "ENMA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)
HUMAN_USD_PATH = os.environ.get(
    "ENMA_HUMAN_USD",
    os.path.join(ENMA_ROOT, "ENMAmiddle", "assets", "humans", "male_adult_medical_01", "male_adult_medical_01.usd"),
)
LL_POLICY_PATH = os.environ.get(
    "ENMA_LL_POLICY", os.path.join(ENMA_ROOT, "checkpoints", "low_level_locomotion", "policy.pt")
)


def get_first_valid_path(candidates: list[str]) -> str | None:
    """Return the first existing path from a list of candidate paths."""
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

# ── MDP Functions for ENMA (12-dim Observation) ────────────────────────────────

def nav_commands_intent(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the simulated nav stack intent (3 dims: vx, vy, yaw)."""
    if not hasattr(env, "nav_cmd"):
        return torch.zeros((env.num_envs, 3), device=env.device)
    return env.nav_cmd

def robot_velocity_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Robot base velocity (3 dims: vx, vy, yaw) in base frame."""
    lin_vel = torch.clamp(env.scene["robot"].data.root_lin_vel_b[:, :2], min=-100.0, max=100.0)
    ang_vel = torch.clamp(env.scene["robot"].data.root_ang_vel_b[:, 2:3], min=-100.0, max=100.0)
    return torch.cat([lin_vel, ang_vel], dim=1)

def human_relative_pos_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative position to human (2 dims: x, y) in world frame relative to robot."""
    if not hasattr(env, "human_pos"):
        return torch.zeros((env.num_envs, 2), device=env.device)
    rel_pos = env.human_pos - env.scene["robot"].data.root_pos_w[:, :2]
    return torch.clamp(rel_pos, min=-100.0, max=100.0)

def human_relative_vel_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative velocity of human (2 dims: vx, vy) in world frame relative to robot."""
    if not hasattr(env, "human_vel"):
        return torch.zeros((env.num_envs, 2), device=env.device)
    rel_vel = env.human_vel - env.scene["robot"].data.root_lin_vel_w[:, :2]
    return torch.clamp(rel_vel, min=-100.0, max=100.0)

def get_task_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 (moving), 2.0 (interrupted)."""
    if not hasattr(env, "task_state"):
        return torch.ones((env.num_envs, 1), device=env.device)
    return env.task_state.unsqueeze(1)

def get_human_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Euclidean distance to human."""
    if not hasattr(env, "human_pos"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return torch.norm(env.human_pos - env.scene["robot"].data.root_pos_w[:, :2], dim=1, keepdim=True)

# ── Social Force Model (SFM) Implementation ───────────────────────────────────

def compute_sfm_forces(human_pos, human_goal, robot_pos, env_device):
    """
    Minimal Social Force Model (Helbing & Molnár 1995).
    F_total = F_attractive(goal) + F_repulsive(walls) + F_repulsive(robot)
    """
    # Attractive to goal
    k_a = 1.0
    desired_speed = 1.0
    dir_to_goal = human_goal - human_pos
    dist_to_goal = torch.norm(dir_to_goal, dim=1, keepdim=True)
    f_attr = k_a * (dir_to_goal / (dist_to_goal + 1e-6)) * desired_speed

    # Repulsive from robot
    k_r = 2.0
    sigma = 0.5
    rel_robot = human_pos - robot_pos
    dist_to_robot = torch.norm(rel_robot, dim=1, keepdim=True)
    f_rep_robot = k_r * torch.exp(-dist_to_robot / sigma) * (rel_robot / (dist_to_robot + 1e-6))

    # Repulsive from walls
    dist_left = human_pos[:, 1:2] + 5.0
    dist_right = 5.0 - human_pos[:, 1:2]
    f_rep_walls = k_r * (
        torch.exp(-dist_left / sigma) * torch.tensor([[0.0, 1.0]], device=env_device) +
        torch.exp(-dist_right / sigma) * torch.tensor([[0.0, -1.0]], device=env_device)
    )

    return f_attr + f_rep_robot + f_rep_walls


def _zero_command_cfg() -> mdp.UniformLevelVelocityCommandCfg:
    return mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )


def _identity_quat(device: torch.device, num: int) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num, 1)


def _heading_quat(yaw: torch.Tensor) -> torch.Tensor:
    return quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)


def _unit_xy(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.clamp(torch.norm(vec, dim=1, keepdim=True), min=1e-6)


def _stable_camera_target(camera_pos: torch.Tensor, target_pos: torch.Tensor, min_horizontal_offset: float = 0.35) -> torch.Tensor:
    """Keep a look-at target from collapsing into a nearly vertical view direction.

    Isaac's look-at helper becomes numerically fragile when the view vector is aligned with the stage up axis.
    This helper preserves the intended target while enforcing a small horizontal component.
    """
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
    return stable_target

# ── Configuration Classes ─────────────────────────────────────────────────────

@configclass
class ENMASceneCfg(RobotSceneCfg):
    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.4, 0.6, 1.0),
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
    )
    human: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Human",
        spawn=UsdFileCfg(
            usd_path=HUMAN_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    human_pov_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Human/HeadCamera",
        update_period=0.1,  # 10 FPS
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.18, 0.0, 0.58), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


@configclass
class ENMATrainingSceneCfg(RobotSceneCfg):
    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.4, 0.6, 1.0),
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
    )
    human: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Human",
        spawn=UsdFileCfg(
            usd_path=HUMAN_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    human_pov_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Human/HeadCamera",
        update_period=0.1,  # 10 FPS
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.18, 0.0, 0.58), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


@configclass
class ENMAInteractionSceneCfg(ENMASceneCfg):
    wall_left: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallLeft",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 0.1, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(5.0, 5.0, 1.0))
    )
    wall_right: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallRight",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 0.1, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(5.0, -5.0, 1.0))
    )

# ── Observation Groups ────────────────────────────────────────────────────────

@configclass
class ENMAObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        """12-dimensional observation vector for ENMA."""
        nav_cmd = ObsTerm(func=nav_commands_intent)
        robot_vel = ObsTerm(func=robot_velocity_base)
        human_rel_pos = ObsTerm(func=human_relative_pos_base)
        human_rel_vel = ObsTerm(func=human_relative_vel_base)
        task_state = ObsTerm(func=get_task_state)
        human_dist = ObsTerm(func=get_human_dist)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    # Main policy group (12 dims)
    policy: PolicyCfg = PolicyCfg()
    
    # Internal group for low-level policy (48 dims)
    @configclass
    class LowLevelCfg(ObservationsCfg.PolicyCfg):
        # Match the exported low-level walking policy layout:
        # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
        # + joint_pos_rel(12) + joint_vel_rel(12) + last_action(12) = 48
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    low_level: LowLevelCfg = LowLevelCfg()

@configclass
class ENMAEnvCfg(RobotEnvCfg):
    scene: ENMATrainingSceneCfg = ENMATrainingSceneCfg(num_envs=4096, env_spacing=15.0)
    observations: ENMAObservationsCfg = ENMAObservationsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(10.0, 5.0, 5.0), lookat=(5.0, 0.0, 0.0))
    robot_stationary: bool = False
    human_motion_mode: str = "navigation"
    use_human_pov_camera: bool = True
    fake_controller: bool = False
    use_new_ll_policy: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        # Disable tracking rewards for the middle layer since it doesn't track base_velocity
        self.rewards.track_lin_vel_xy.weight = 0.0
        self.rewards.track_ang_vel_z.weight = 0.0
        # Relax standing still penalty so it can stand
        self.rewards.joint_pos.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0

@configclass
class ENMATestCfg(ENMAEnvCfg):
    scene: ENMAInteractionSceneCfg = ENMAInteractionSceneCfg(num_envs=4096, env_spacing=15.0)
    observations: ENMAObservationsCfg = ENMAObservationsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(10.0, 5.0, 5.0), lookat=(5.0, 0.0, 0.0))
    robot_stationary: bool = False
    human_motion_mode: str = "navigation"
    use_human_pov_camera: bool = True



@configclass
class StationaryCommandsCfg(CommandsCfg):
    base_velocity = _zero_command_cfg()


@configclass
class StationaryEventCfg(EventCfg):
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = None


@configclass
class ENMAInteractionEnvCfg(ENMAEnvCfg):
    scene: ENMAInteractionSceneCfg = ENMAInteractionSceneCfg(num_envs=4096, env_spacing=15.0)
    commands: StationaryCommandsCfg = StationaryCommandsCfg()
    events: StationaryEventCfg = StationaryEventCfg()
    robot_stationary: bool = True
    human_motion_mode: str = "approach"
    use_human_pov_camera: bool = True


@configclass
class ENMAInteractionApproachEnvCfg(ENMAInteractionEnvCfg):
    human_motion_mode: str = "approach"


@configclass
class ENMAInteractionStandEnvCfg(ENMAInteractionEnvCfg):
    human_motion_mode: str = "stand"


@configclass
class ENMAInteractionOrbitEnvCfg(ENMAInteractionEnvCfg):
    human_motion_mode: str = "orbit"


@configclass
class ENMAInteractionCrossEnvCfg(ENMAInteractionEnvCfg):
    human_motion_mode: str = "cross"

# ── Environment Class ─────────────────────────────────────────────────────────

class ENMAEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ENMAEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_stationary = bool(getattr(cfg, "robot_stationary", False))
        self.human_motion_mode = getattr(cfg, "human_motion_mode", "navigation")
        self.use_human_pov_camera = bool(getattr(cfg, "use_human_pov_camera", False))

        self.nav_cmd = torch.zeros((self.num_envs, 3), device=self.device)
        self.final_cmd = torch.zeros((self.num_envs, 6), device=self.device)
        self.human_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.human_vel = torch.zeros((self.num_envs, 2), device=self.device)
        self.human_heading = torch.zeros(self.num_envs, device=self.device)
        self.human_quat = _identity_quat(self.device, self.num_envs)
        self.task_state = torch.ones(self.num_envs, device=self.device)
        
        self.robot_endpoints = torch.zeros((self.num_envs, 2, 2), device=self.device)
        self.human_endpoints = torch.zeros((self.num_envs, 2, 2), device=self.device)
        self.robot_target_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.human_target_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.robot_trips = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.human_trips = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self.human_mode = torch.zeros(self.num_envs, device=self.device)
        self.human_base_speed = torch.zeros(self.num_envs, device=self.device)
        self.pause_timer = torch.zeros(self.num_envs, device=self.device)
        self.burst_timer = torch.zeros(self.num_envs, device=self.device)
        self.human_orbit_angle = torch.zeros(self.num_envs, device=self.device)
        self.human_orbit_direction = torch.ones(self.num_envs, device=self.device)
        self.human_target_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.camera_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.camera_target_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.camera_head_offset = torch.tensor([0.18, 0.0, 0.58], device=self.device)
        self.camera_look_offset = torch.tensor([0.0, 0.0, 0.25], device=self.device)
        self.camera_smoothing = 0.2
        self.camera_height_offset = 0.58
        
        ll_candidates = [LL_POLICY_PATH]
        ll_path = get_first_valid_path(ll_candidates)
        self.ll_policy = None if self.robot_stationary or ll_path is None else torch.jit.load(ll_path, map_location=self.device)
        if self.ll_policy: self.ll_policy.eval()

        self.use_new_ll_policy = bool(getattr(cfg, "use_new_ll_policy", False))
        self.fake_controller_enabled = bool(getattr(cfg, "fake_controller", False))
        if self.fake_controller_enabled:
            from isaaclab.controllers import FakeSportController
            self.fake_sport_controller = FakeSportController(self.num_envs, self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids, :2]
        robot_pos_l = self.scene["robot"].data.root_pos_w[env_ids, :2] - env_origins
        
        self.nav_cmd[env_ids] = 0.0
        self.robot_target_idx[env_ids] = 0
        self.robot_trips[env_ids] = 0

        if self.robot_stationary:
            self.robot_endpoints[env_ids] = 0.0
            self.human_trips[env_ids] = 0
            self.human_target_idx[env_ids] = 0
            self.human_base_speed[env_ids] = 0.0

        # Randomize human motion mode per environment if base mode is 'mixed' or we want to train all cases
        scenarios = torch.randint(0, 5, (n,), device=self.device)
        
        # Scenario 0: approach
        approach_mask = (scenarios == 0)
        if approach_mask.any():
            m_n = approach_mask.sum()
            idx = env_ids[approach_mask]
            directions = _unit_xy(torch.rand((m_n, 2), device=self.device) * 2.0 - 1.0)
            start_dist = 2.8 + torch.rand(m_n, device=self.device) * 0.7
            goal_dist = 1.0 + torch.rand(m_n, device=self.device) * 0.25
            self.human_endpoints[idx, 0] = robot_pos_l[approach_mask] + directions * start_dist.unsqueeze(1)
            self.human_endpoints[idx, 1] = robot_pos_l[approach_mask] + directions * goal_dist.unsqueeze(1)
            self.human_pos[idx] = self.human_endpoints[idx, 0]
            self.human_target_idx[idx] = 1
            self.human_trips[idx] = 1
            self.human_base_speed[idx] = 0.55 + torch.rand(m_n, device=self.device) * 0.2
            self.human_heading[idx] = torch.atan2(directions[:, 1], directions[:, 0])

        # Scenario 1: stand
        stand_mask = (scenarios == 1)
        if stand_mask.any():
            m_n = stand_mask.sum()
            idx = env_ids[stand_mask]
            offsets = _unit_xy(torch.rand((m_n, 2), device=self.device) * 2.0 - 1.0)
            stand_dist = 1.1 + torch.rand(m_n, device=self.device) * 0.2
            self.human_pos[idx] = robot_pos_l[stand_mask] + offsets * stand_dist.unsqueeze(1)
            self.human_target_idx[idx] = 0
            self.human_trips[idx] = 0
            self.human_base_speed[idx] = 0.0
            look_dir = robot_pos_l[stand_mask] - self.human_pos[idx]
            self.human_heading[idx] = torch.atan2(look_dir[:, 1], look_dir[:, 0])
            self.human_endpoints[idx, 0] = self.human_pos[idx]
            self.human_endpoints[idx, 1] = self.human_pos[idx]

        # Scenario 2: orbit
        orbit_mask = (scenarios == 2)
        if orbit_mask.any():
            m_n = orbit_mask.sum()
            idx = env_ids[orbit_mask]
            self.human_orbit_angle[idx] = torch.rand(m_n, device=self.device) * (2.0 * math.pi)
            self.human_orbit_direction[idx] = torch.where(
                torch.rand(m_n, device=self.device) < 0.5, -torch.ones(m_n, device=self.device), torch.ones(m_n, device=self.device)
            )
            self.human_base_speed[idx] = 0.65 + torch.rand(m_n, device=self.device) * 0.2
            orbit_radius = 1.35 + torch.rand(m_n, device=self.device) * 0.35
            orbit_pos = torch.stack(
                [
                    torch.cos(self.human_orbit_angle[idx]) * orbit_radius,
                    torch.sin(self.human_orbit_angle[idx]) * orbit_radius,
                ],
                dim=1,
            )
            self.human_pos[idx] = robot_pos_l[orbit_mask] + orbit_pos
            tangent = torch.stack(
                [
                    -torch.sin(self.human_orbit_angle[idx]) * self.human_orbit_direction[idx],
                    torch.cos(self.human_orbit_angle[idx]) * self.human_orbit_direction[idx],
                ],
                dim=1,
            )
            self.human_heading[idx] = torch.atan2(tangent[:, 1], tangent[:, 0])

        # Scenario 3: cross
        cross_mask = (scenarios == 3)
        if cross_mask.any():
            m_n = cross_mask.sum()
            idx = env_ids[cross_mask]
            side = torch.where(torch.rand(m_n, device=self.device) < 0.5, -torch.ones(m_n, device=self.device), torch.ones(m_n, device=self.device))
            cross_y = (torch.rand(m_n, device=self.device) - 0.5) * 1.0
            start_x = side * (3.0 + torch.rand(m_n, device=self.device) * 0.8)
            goal_x = -start_x
            self.human_endpoints[idx, 0] = robot_pos_l[cross_mask] + torch.stack([start_x, cross_y], dim=1)
            self.human_endpoints[idx, 1] = robot_pos_l[cross_mask] + torch.stack([goal_x, cross_y], dim=1)
            self.human_pos[idx] = self.human_endpoints[idx, 0]
            self.human_target_idx[idx] = 1
            self.human_trips[idx] = 2
            self.human_base_speed[idx] = 0.75 + torch.rand(m_n, device=self.device) * 0.15
            move_dir = self.human_endpoints[idx, 1] - self.human_pos[idx]
            self.human_heading[idx] = torch.atan2(move_dir[:, 1], move_dir[:, 0])

        # Scenario 4: navigation
        nav_mask = (scenarios == 4)
        if nav_mask.any():
            m_n = nav_mask.sum()
            idx = env_ids[nav_mask]
            self.robot_endpoints[idx, 0, 0] = 1.0 + torch.rand(m_n, device=self.device) * 0.5
            self.robot_endpoints[idx, 0, 1] = (torch.rand(m_n, device=self.device) - 0.5) * 2.0
            self.robot_endpoints[idx, 1, 0] = 9.0 - torch.rand(m_n, device=self.device) * 0.5
            self.robot_endpoints[idx, 1, 1] = (torch.rand(m_n, device=self.device) - 0.5) * 2.0
            
            self.human_endpoints[idx, 0, 0] = 5.0 + (torch.rand(m_n, device=self.device) - 0.5) * 2.0
            self.human_endpoints[idx, 0, 1] = 1.0 + torch.rand(m_n, device=self.device) * 0.5
            self.human_endpoints[idx, 1, 0] = 5.0 + (torch.rand(m_n, device=self.device) - 0.5) * 2.0
            self.human_endpoints[idx, 1, 1] = 9.0 - torch.rand(m_n, device=self.device) * 0.5
            
            self.robot_target_idx[idx] = torch.randint(0, 2, (m_n,), device=self.device)
            self.human_target_idx[idx] = torch.randint(0, 2, (m_n,), device=self.device)
            self.robot_trips[idx] = torch.randint(12, 17, (m_n,), device=self.device)
            self.human_trips[idx] = torch.randint(12, 17, (m_n,), device=self.device)
            self.human_base_speed[idx] = 0.8 + torch.rand(m_n, device=self.device) * 0.5
            self.human_pos[idx] = self.human_endpoints[idx, 1 - self.human_target_idx[idx]]
            self.human_heading[idx] = torch.zeros(m_n, device=self.device)
            
        # Store scenario in a new buffer for update loop
        if not hasattr(self, 'env_scenarios'):
            self.env_scenarios = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.env_scenarios[env_ids] = scenarios


        # Note: Scenarios already randomized above

    def _update_human_motion(self, env_ids, robot_pos_l, dt):
        if not hasattr(self, 'env_scenarios'):
            self.env_scenarios = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Determine movement logic based on per-env scenario
        nav_v = torch.zeros((self.num_envs, 2), device=self.device)
        target_yaw = torch.zeros(self.num_envs, device=self.device)
        f_total = torch.zeros((self.num_envs, 2), device=self.device)
        
        # Navigation scenario (4)
        nav_mask = (self.env_scenarios == 4)
        if nav_mask.any():
            r_target = torch.gather(self.robot_endpoints[nav_mask], 1, self.robot_target_idx[nav_mask].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            r_diff = r_target - robot_pos_l[nav_mask]
            r_dist = torch.norm(r_diff, dim=1, keepdim=True)
            reached = (r_dist < 0.3).squeeze(1)
            # update target for those that reached
            reached_full_idx = nav_mask.nonzero(as_tuple=True)[0][reached]
            self.robot_target_idx[reached_full_idx] = 1 - self.robot_target_idx[reached_full_idx]
            self.robot_trips[reached_full_idx] -= 1

            nav_v[nav_mask] = (r_diff / (r_dist + 1e-6)) * torch.clamp(r_dist, max=1.0)
            target_yaw[nav_mask] = torch.atan2(r_diff[:, 1], r_diff[:, 0])
            
            h_target = torch.gather(self.human_endpoints[nav_mask], 1, self.human_target_idx[nav_mask].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            h_reached = (torch.norm(h_target - self.human_pos[nav_mask], dim=1) < 0.3)
            h_reached_full_idx = nav_mask.nonzero(as_tuple=True)[0][h_reached]
            self.human_target_idx[h_reached_full_idx] = 1 - self.human_target_idx[h_reached_full_idx]
            self.human_trips[h_reached_full_idx] -= 1

            sfm_f = compute_sfm_forces(self.human_pos[nav_mask], h_target, robot_pos_l[nav_mask], self.device)
            direct_f = (h_target - self.human_pos[nav_mask]) / (torch.norm(h_target - self.human_pos[nav_mask], dim=1, keepdim=True) + 1e-6)
            f_total[nav_mask] = torch.where(self.human_mode[nav_mask].view(-1, 1) == 0.0, sfm_f, direct_f)
            self.human_target_pos[nav_mask] = h_target

        # Other scenarios: static robot, dynamic human
        static_mask = ~nav_mask
        if static_mask.any():
            self.nav_cmd[static_mask] = 0.0
            
            # approach (0)
            app_mask = (self.env_scenarios == 0)
            if app_mask.any():
                h_target = self.human_endpoints[app_mask, 1]
                f_total[app_mask] = (h_target - self.human_pos[app_mask]) / (torch.norm(h_target - self.human_pos[app_mask], dim=1, keepdim=True) + 1e-6)
                self.human_target_pos[app_mask] = h_target
                
            # stand (1)
            stand_mask = (self.env_scenarios == 1)
            if stand_mask.any():
                f_total[stand_mask] = 0.0
                
            # orbit (2)
            orb_mask = (self.env_scenarios == 2)
            if orb_mask.any():
                orbit_radius = torch.norm(self.human_pos[orb_mask] - robot_pos_l[orb_mask], dim=1, keepdim=True).clamp(min=1.0)
                self.human_orbit_angle[orb_mask] += self.human_orbit_direction[orb_mask] * 0.55 * dt
                h_target = robot_pos_l[orb_mask] + torch.stack([
                    torch.cos(self.human_orbit_angle[orb_mask]) * orbit_radius.squeeze(1),
                    torch.sin(self.human_orbit_angle[orb_mask]) * orbit_radius.squeeze(1)
                ], dim=1)
                f_total[orb_mask] = (h_target - self.human_pos[orb_mask]) / (torch.norm(h_target - self.human_pos[orb_mask], dim=1, keepdim=True) + 1e-6)
                self.human_target_pos[orb_mask] = h_target
                
            # cross (3)
            cross_mask = (self.env_scenarios == 3)
            if cross_mask.any():
                h_target = self.human_endpoints[cross_mask, self.human_target_idx[cross_mask]]
                reached = torch.norm(h_target - self.human_pos[cross_mask], dim=1) < 0.25
                if reached.any():
                    reached_full_idx = cross_mask.nonzero(as_tuple=True)[0][reached]
                    self.human_target_idx[reached_full_idx] = 1 - self.human_target_idx[reached_full_idx]
                    h_target = self.human_endpoints[cross_mask, self.human_target_idx[cross_mask]]
                f_total[cross_mask] = (h_target - self.human_pos[cross_mask]) / (torch.norm(h_target - self.human_pos[cross_mask], dim=1, keepdim=True) + 1e-6)
                self.human_target_pos[cross_mask] = h_target

        _, _, curr_yaw = euler_xyz_from_quat(self.scene["robot"].data.root_quat_w)
        yaw_err = (target_yaw - curr_yaw + math.pi) % (2 * math.pi) - math.pi
        
        target_nav_cmd = torch.zeros_like(self.nav_cmd)
        
        if nav_mask.any():
            # r_diff is already computed above inside the nav_mask block, but we need it here
            r_target = torch.gather(self.robot_endpoints[nav_mask], 1, self.robot_target_idx[nav_mask].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            r_diff = r_target - self.scene["robot"].data.root_pos_w[nav_mask, :2] + self.scene.env_origins[nav_mask, :2]
            r_dist = torch.norm(r_diff, dim=1)
            v_mag = torch.clamp(r_dist, max=1.0)
            
            # Use polar coordinates relative to current heading
            base_vx = v_mag * torch.cos(yaw_err[nav_mask])
            base_vy = v_mag * torch.sin(yaw_err[nav_mask])
            
            # Prevent fast backwards walking to avoid falls. If goal is behind, it should primarily turn in place.
            base_vx = torch.clamp(base_vx, min=-0.2) 
            
            target_nav_cmd[nav_mask, 0] = base_vx
            target_nav_cmd[nav_mask, 1] = base_vy
            target_nav_cmd[nav_mask, 2] = 2.0 * yaw_err[nav_mask]

        # Smooth the nav_cmd with EMA to prevent instantaneous velocity spikes
        if not hasattr(self, "smoothed_nav_cmd"):
            self.smoothed_nav_cmd = target_nav_cmd.clone()
        else:
            ALPHA_NAV = 0.1
            self.smoothed_nav_cmd = ALPHA_NAV * target_nav_cmd + (1.0 - ALPHA_NAV) * self.smoothed_nav_cmd

        self.nav_cmd = self.smoothed_nav_cmd.clone()
        
        return f_total

    def _update_human_camera(self, robot_pos_w, env_origins, reset: bool = False):
        if not self.use_human_pov_camera or "human_pov_camera" not in self.scene.keys():
            return

        camera = self.scene["human_pov_camera"]
        if not getattr(camera, "is_initialized", False):
            return
        h_pos_3d_w = torch.zeros((self.num_envs, 3), device=self.device)
        h_pos_3d_w[:, :2] = self.human_pos + env_origins
        h_pos_3d_w[:, 2] = 1.05 + self.camera_height_offset
        camera_target = torch.zeros((self.num_envs, 3), device=self.device)
        camera_target[:, :2] = robot_pos_w[:, :2]
        camera_target[:, 2] = robot_pos_w[:, 2] + 0.2
        camera_pos = h_pos_3d_w + quat_apply(self.human_quat, self.camera_head_offset.expand(self.num_envs, -1))
        camera_target = _stable_camera_target(camera_pos, camera_target)

        if reset:
            self.camera_pos_w = camera_pos
            self.camera_target_w = camera_target
        else:
            self.camera_pos_w = self.camera_pos_w * (1.0 - self.camera_smoothing) + camera_pos * self.camera_smoothing
            self.camera_target_w = self.camera_target_w * (1.0 - self.camera_smoothing) + camera_target * self.camera_smoothing
            self.camera_pos_w = torch.nan_to_num(self.camera_pos_w)
            self.camera_target_w = torch.nan_to_num(self.camera_target_w)
        camera.set_world_poses_from_view(self.camera_pos_w, self.camera_target_w)

    def _step_impl(self, action):
        robot_pos_w = self.scene["robot"].data.root_pos_w[:, :2]
        env_origins = self.scene.env_origins[:, :2]
        robot_pos_l = robot_pos_w - env_origins

        dt = self.physics_dt * self.cfg.decimation
        self.pause_timer = torch.clamp(self.pause_timer - dt, min=0.0)
        self.burst_timer = torch.clamp(self.burst_timer - dt, min=0.0)
        
        pause_trigger = (torch.rand(self.num_envs, device=self.device) < 0.01) & (self.pause_timer == 0.0)
        self.pause_timer[pause_trigger] = 1.0 + torch.rand(pause_trigger.sum(), device=self.device) * 2.0
        burst_trigger = (torch.rand(self.num_envs, device=self.device) < 0.005) & (self.burst_timer == 0.0)
        self.burst_timer[burst_trigger] = 1.0 + torch.rand(burst_trigger.sum(), device=self.device) * 1.0
        
        curr_h_speed = self.human_base_speed.clone()
        curr_h_speed[self.pause_timer > 0.0] = 0.0
        curr_h_speed[self.burst_timer > 0.0] *= 1.5

        f_total = self._update_human_motion(torch.arange(self.num_envs, device=self.device), robot_pos_l, dt)
        self.human_vel = torch.nan_to_num(f_total * curr_h_speed.view(-1, 1))
        self.human_pos = torch.nan_to_num(self.human_pos + self.human_vel * dt)
        moving = torch.norm(self.human_vel, dim=1) > 1e-4
        if moving.any():
            self.human_heading[moving] = torch.atan2(self.human_vel[moving, 1], self.human_vel[moving, 0])
        if self.human_motion_mode == "stand":
            look_dir = robot_pos_l - self.human_pos
            self.human_heading = torch.atan2(look_dir[:, 1], look_dir[:, 0])
        self.human_heading = torch.nan_to_num(self.human_heading)
        self.human_quat = torch.nan_to_num(_heading_quat(self.human_heading))

        self.final_cmd = torch.zeros((self.num_envs, 6), device=self.device)
        if self.fake_controller_enabled:
            n_act = min(action.shape[1], 6)
            self.final_cmd[:, :n_act] = action[:, :n_act].clone()
        else:
            action_scale = torch.ones_like(action)
            if action.shape[1] >= 6:
                action_scale[:, 4] = 0.3
                action_scale[:, 5] = 0.3
            if not self.robot_stationary:
                self.final_cmd[:, :3] = self.nav_cmd + action[:, :3]
            else:
                self.final_cmd[:, :3] = action[:, :3]
            if action.shape[1] >= 6:
                self.final_cmd[:, 3:6] = action[:, 3:6] * action_scale[:, 3:6]

        h_pos_3d_w = torch.zeros((self.num_envs, 3), device=self.device)
        h_pos_3d_w[:, :2] = self.human_pos + env_origins
        h_pos_3d_w[:, 2] = 1.05
        try:
            self.scene["human"].write_root_pose_to_sim(torch.cat([h_pos_3d_w, self.human_quat], dim=1))
        except Exception:
            pass

        self._update_human_camera(self.scene["robot"].data.root_pos_w, env_origins)

        self.task_state[:] = 1.0
        self.task_state[torch.norm(robot_pos_l - self.human_pos, dim=1) < 1.5] = 2.0
        
    def step(self, action: torch.Tensor):
        if self.fake_controller_enabled:
             ll_action = self.fake_sport_controller.compute(action, self)
             self.action_manager.process_action(ll_action.to(self.device))
        elif self.ll_policy is not None:
             ll_obs = self.observation_manager.compute_group("low_level")
             ll_obs = torch.nan_to_num(ll_obs, nan=0.0, posinf=0.0, neginf=0.0)
             ll_obs = torch.clamp(ll_obs, min=-100.0, max=100.0)
             
             # Overwrite command in low_level group (indices 6:12) with final 6D command
             ll_obs[:, 6:12] = torch.clamp(self.final_cmd[:, :6], min=-10.0, max=10.0)

             if self.use_new_ll_policy:
                 ll_obs_adapted = ll_obs[:, :48]
             else:
                 lin_vel = ll_obs[:, 48:51]
                 ang_vel = ll_obs[:, 0:3]
                 gravity = ll_obs[:, 3:6]
                 cmd_3d = ll_obs[:, 6:9]
                 joint_pos = ll_obs[:, 12:24]
                 joint_vel = ll_obs[:, 24:36]
                 last_act = ll_obs[:, 36:48]
                 ll_obs_adapted = torch.cat([lin_vel, ang_vel, gravity, cmd_3d, joint_pos, joint_vel, last_act], dim=1)

             ll_obs_adapted = torch.nan_to_num(ll_obs_adapted, nan=0.0, posinf=0.0, neginf=0.0)
             ll_obs_adapted = torch.clamp(ll_obs_adapted, min=-100.0, max=100.0)
             with torch.no_grad():
                  ll_action = self.ll_policy(ll_obs_adapted)
             ll_action = torch.nan_to_num(ll_action, nan=0.0, posinf=0.0, neginf=0.0)
             ll_action = torch.clamp(ll_action, min=-100.0, max=100.0)
             self.action_manager.process_action(ll_action.to(self.device))
        else:
             self.action_manager.process_action(torch.zeros(self.num_envs, 12, device=self.device))

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        
        # Hard simulator state check to reset envs with broken/NaN/Inf physics
        robot_data = self.scene["robot"].data
        invalid_mask = (
            torch.isnan(robot_data.root_pos_w).any(dim=-1) |
            torch.isinf(robot_data.root_pos_w).any(dim=-1) |
            torch.isnan(robot_data.joint_pos).any(dim=-1) |
            torch.isinf(robot_data.joint_pos).any(dim=-1) |
            torch.isnan(robot_data.root_lin_vel_w).any(dim=-1) |
            torch.isinf(robot_data.root_lin_vel_w).any(dim=-1)
        )
        if invalid_mask.any():
            invalid_ids = invalid_mask.nonzero(as_tuple=False).squeeze(-1)
            self.reset_buf[invalid_ids] = True

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # Ensure rewards remain finite
        self.reward_buf = torch.nan_to_num(self.reward_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self.reward_buf = torch.clamp(self.reward_buf, min=-50.0, max=50.0)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        
        # Add hard numerical safety checks and debug instrumentation
        for key in self.obs_buf:
            obs_tensor = self.obs_buf[key]
            if torch.isinf(obs_tensor).any() or torch.isnan(obs_tensor).any():
                print(f"INVALID OBS DETECTED in group {key}")
                print(torch.where(~torch.isfinite(obs_tensor)))
            self.obs_buf[key] = torch.nan_to_num(obs_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            self.obs_buf[key] = torch.clamp(self.obs_buf[key], min=-100.0, max=100.0)

        return self.obs_buf, self.reward_buf, self.reset_buf, self.reset_buf, self.extras
