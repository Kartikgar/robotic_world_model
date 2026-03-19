# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from mbrl.mbrl.envs import ManagerBasedMBRLEnv


class G1ManagerBasedMBRLEnv(ManagerBasedMBRLEnv):
    """Model-based imagination environment for G1 tracking finetuning."""

    def _init_additional_attributes(self):
        robot = self.scene["robot"]
        self.default_joint_pos = robot.data.default_joint_pos[0]
        self.default_joint_vel = robot.data.default_joint_vel[0]
        self.num_joints = int(self.default_joint_pos.shape[0])
        self.motion_term = self.command_manager.get_term("motion")
        self.motion = self.motion_term.motion

        # Imagination command buffers (initialized in _init_imagination_command).
        self.motion_trajectory_ids = None
        self.motion_time_steps = None
        self.motion_joint_pos = None
        self.motion_joint_vel = None

    def _init_additional_imagination_attributes(self):
        # No extra temporal attributes needed beyond command/state history.
        pass

    def _reset_imagination_idx(self, env_ids):
        super()._reset_imagination_idx(env_ids)
        self._reset_imagination_command(env_ids)

    def _reset_additional_imagination_attributes(self, env_ids):
        # No extra temporal attributes needed beyond command/state history.
        pass

    def _sample_trajectory_ids(self, num_samples: int) -> torch.Tensor:
        if self.motion.num_trajectories <= 1:
            return torch.zeros(num_samples, dtype=torch.long, device=self.device)
        return torch.randint(0, self.motion.num_trajectories, (num_samples,), device=self.device)

    def _update_motion_buffers(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_imagination_envs, device=self.device)
        if env_ids.numel() == 0:
            return

        traj_ids = self.motion_trajectory_ids[env_ids]
        time_steps = self.motion_time_steps[env_ids]
        joint_pos = self.motion.get_joint_pos(traj_ids, time_steps)
        joint_vel = self.motion.get_joint_vel(traj_ids, time_steps)
        if joint_pos.shape[-1] != self.num_joints:
            raise RuntimeError(
                f"Motion joint dim ({joint_pos.shape[-1]}) does not match robot joint dim ({self.num_joints})."
            )
        self.motion_joint_pos[env_ids] = joint_pos
        self.motion_joint_vel[env_ids] = joint_vel

    def _sample_time_steps_for_trajectories(self, trajectory_ids: torch.Tensor) -> torch.Tensor:
        if trajectory_ids.numel() == 0:
            return torch.zeros_like(trajectory_ids)
        lengths = torch.clamp(self.motion.trajectory_time_step_total[trajectory_ids], min=1)
        return (torch.rand(trajectory_ids.shape, device=self.device) * lengths.float()).long()

    def _init_imagination_command(self):
        self.motion_trajectory_ids = self._sample_trajectory_ids(self.num_imagination_envs)
        self.motion_time_steps = self._sample_time_steps_for_trajectories(self.motion_trajectory_ids)
        self.motion_joint_pos = torch.zeros(self.num_imagination_envs, self.num_joints, device=self.device)
        self.motion_joint_vel = torch.zeros(self.num_imagination_envs, self.num_joints, device=self.device)
        self._update_motion_buffers()

    def _reset_imagination_command(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_trajectory_ids[env_ids] = self._sample_trajectory_ids(len(env_ids))
        self.motion_time_steps[env_ids] = self._sample_time_steps_for_trajectories(self.motion_trajectory_ids[env_ids])
        self._update_motion_buffers(env_ids)

    def _advance_motion_command(self):
        self.motion_time_steps += 1
        env_lengths = torch.clamp(self.motion.trajectory_time_step_total[self.motion_trajectory_ids], min=1)
        overflow_ids = (self.motion_time_steps >= env_lengths).nonzero(as_tuple=False).squeeze(-1)
        if overflow_ids.numel() > 0:
            self.motion_time_steps[overflow_ids] = 0
            self.motion_trajectory_ids[overflow_ids] = self._sample_trajectory_ids(len(overflow_ids))
        self._update_motion_buffers()

    def get_imagination_observation(self, state_history, action_history, observation_noise=True):
        denorm_state = self.imagination_state_normalizer.inverse(state_history[:, -1])
        denorm_action = self.imagination_action_normalizer.inverse(action_history[:, -1])

        base_lin_vel = denorm_state[:, 0:3]
        base_ang_vel = denorm_state[:, 3:6]
        joint_pos = denorm_state[:, 9 : 9 + self.num_joints]
        joint_vel = denorm_state[:, 9 + self.num_joints : 9 + 2 * self.num_joints]
        self.obs_last_action = denorm_action

        if observation_noise:
            base_lin_vel = base_lin_vel + 2 * (torch.rand_like(base_lin_vel) - 0.5) * 0.1
            base_ang_vel = base_ang_vel + 2 * (torch.rand_like(base_ang_vel) - 0.5) * 0.2
            joint_pos = joint_pos + 2 * (torch.rand_like(joint_pos) - 0.5) * 0.01
            joint_vel = joint_vel + 2 * (torch.rand_like(joint_vel) - 0.5) * 1.5

        obs = torch.cat(
            [
                self.motion_joint_pos,
                self.motion_joint_vel,
                base_lin_vel,
                base_ang_vel,
                joint_pos,
                joint_vel,
                self.obs_last_action,
            ],
            dim=1,
        )
        obs = TensorDict({"policy": obs}, batch_size=[self.num_imagination_envs], device=self.device)
        self.last_obs = obs
        return obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        return {
            "base_lin_vel": imagination_states_denormalized[:, 0:3],
            "base_ang_vel": imagination_states_denormalized[:, 3:6],
            "projected_gravity": imagination_states_denormalized[:, 6:9],
            "joint_pos": imagination_states_denormalized[:, 9 : 9 + self.num_joints],
            "joint_vel": imagination_states_denormalized[:, 9 + self.num_joints : 9 + 2 * self.num_joints],
            "joint_torque": imagination_states_denormalized[:, 9 + 2 * self.num_joints : 9 + 3 * self.num_joints],
        }

    def _parse_extensions(self, extensions):
        if extensions is None:
            return None
        return {}

    def _parse_contacts(self, contacts):
        # System contact target is [left_ankle_contact, right_ankle_contact].
        foot_contact = torch.sigmoid(contacts[:, 0:2]).round() if contacts is not None else None
        return {"foot_contact": foot_contact}

    def _parse_terminations(self, terminations):
        return torch.sigmoid(terminations).squeeze(-1).round().bool() if terminations is not None else None

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_vel = parsed_imagination_states["joint_vel"]

        joint_pos_error = torch.sum(torch.square(joint_pos - self.motion_joint_pos), dim=1)
        joint_vel_error = torch.sum(torch.square(joint_vel - self.motion_joint_vel), dim=1)

        motion_joint_pos = torch.exp(-joint_pos_error / (0.5**2))
        motion_joint_vel = torch.exp(-joint_vel_error / (2.0**2))
        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)
        penalty_minimal_action_norm = torch.exp(-torch.norm(rollout_action, dim=-1)) - 1.0

        self.imagination_reward_per_step = {
            "motion_joint_pos": motion_joint_pos,
            "motion_joint_vel": motion_joint_vel,
            "action_rate_l2": action_rate_l2,
            "penalty_minimal_action_norm": penalty_minimal_action_norm,
        }

        last_obs = torch.cat(
            [
                self.motion_joint_pos,
                self.motion_joint_vel,
                parsed_imagination_states["base_lin_vel"],
                parsed_imagination_states["base_ang_vel"],
                joint_pos,
                joint_vel,
                rollout_action,
            ],
            dim=1,
        )
        self.last_obs = TensorDict({"policy": last_obs}, batch_size=[self.num_imagination_envs], device=self.device)
        self._advance_motion_command()
