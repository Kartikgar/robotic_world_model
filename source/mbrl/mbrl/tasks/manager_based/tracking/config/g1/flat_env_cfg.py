from __future__ import annotations

import os

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import mbrl.tasks.manager_based.tracking.mdp as mdp
from mbrl.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from mbrl.tasks.manager_based.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from mbrl.tasks.manager_based.tracking.tracking_env_cfg import (
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    TrackingEnvCfg,
)

JOINT_TRACKING_THRESHOLD_PRETRAIN = 1.0e6


def _resolve_default_motion_file() -> str:
    candidates = [
        os.environ.get("MBRL_G1_MOTION_FILE", ""),
        "/home/kartikgarg/whole_body_tracking/artifacts/walk_lvl1_g1:v0/motion.npz",
        "/home/kartikgarg/whole_body_tracking/ASAP_npz/walk_level1.npz",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    # Keep a deterministic fallback path so users see what to override.
    return "/home/kartikgarg/whole_body_tracking/artifacts/walk_lvl1_g1:v0/motion.npz"


@configclass
class MbrlPolicyObsCfg(ObsGroup):
    """Policy observations used for model-based pretrain/finetune."""

    command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class MbrlCriticObsCfg(ObsGroup):
    command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):
    """Adds system observation groups required by model-based training."""

    @configclass
    class SystemStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=mdp.joint_effort)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemActionCfg(ObsGroup):
        pred_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemContactCfg(ObsGroup):
        foot_contact = ObsTerm(
            func=mdp.body_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemTerminationCfg(ObsGroup):
        joint_tracking = ObsTerm(
            func=mdp.bad_joint_tracking_obs,
            params={"command_name": "motion", "threshold": JOINT_TRACKING_THRESHOLD_PRETRAIN},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: MbrlPolicyObsCfg = MbrlPolicyObsCfg()
    critic: MbrlCriticObsCfg = MbrlCriticObsCfg()
    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
    system_contact: SystemContactCfg = SystemContactCfg()
    system_termination: SystemTerminationCfg = SystemTerminationCfg()


@configclass
class RewardsCfg_MBRL(RewardsCfg):
    """Reward terms mirrored in the imagination wrapper for MBPO finetune."""

    # Disable pose-heavy terms that require full body-pose reconstruction.
    motion_global_anchor_pos = None
    motion_global_anchor_ori = None
    motion_body_pos = None
    motion_body_ori = None
    motion_body_pos_global = None
    motion_body_ori_global = None
    motion_body_lin_vel = None
    motion_body_ang_vel = None
    joint_limit = None
    undesired_contacts = None

    motion_joint_pos = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.5},
    )
    motion_joint_vel = RewTerm(
        func=mdp.motion_joint_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 2.0},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    penalty_minimal_action_norm = RewTerm(func=mdp.penalty_minimal_action_norm, weight=-0.1)


@configclass
class RewardsCfg_PRETRAIN_WHOLEBODY9(RewardsCfg):
    """Whole-body-style tracking rewards for pretraining (9 terms)."""

    # Keep the same 9 core terms used in whole-body style tracking.
    # 8 motion-tracking terms + action-rate penalty.
    mmotion_body_pos_global = None
    motion_body_ori_global = None


@configclass
class TerminationsCfg_MBRL(TerminationsCfg):
    # anchor_pos = None
    # anchor_ori = None
    # ee_body_pos = None
    joint_tracking = DoneTerm(
        func=mdp.bad_joint_tracking,
        params={"command_name": "motion", "threshold": JOINT_TRACKING_THRESHOLD_PRETRAIN},
    )


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    """Reference G1 tracking task (closest to whole-body-tracking baseline)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.motion_file = _resolve_default_motion_file()
        self.commands.motion.debug_vis_goal_relative_to_robot = False
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        self.episode_length_s = 10.0


@configclass
class G1FlatEnvCfg_INIT(G1FlatEnvCfg):
    pass


@configclass
class G1FlatEnvCfg_PRETRAIN(G1FlatEnvCfg):
    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()
    rewards: RewardsCfg_PRETRAIN_WHOLEBODY9 = RewardsCfg_PRETRAIN_WHOLEBODY9()
    terminations: TerminationsCfg_MBRL = TerminationsCfg_MBRL()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.sample_trajectories = True
        self.commands.motion.equal_trajectory_sampling = True


@configclass
class G1FlatEnvCfg_FINETUNE(G1FlatEnvCfg_PRETRAIN):
    rewards: RewardsCfg_MBRL = RewardsCfg_MBRL()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class G1FlatEnvCfg_VISUALIZE(G1FlatEnvCfg_PRETRAIN):
    rewards: RewardsCfg_MBRL = RewardsCfg_MBRL()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
