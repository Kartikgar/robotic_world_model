# G1 Motion-Tracking Integration (What Changed)

This document explains the G1 tracking integration that was added to this repository, with a focus on:
- what files changed,
- what new tasks were added,
- how model-based dynamics training was preserved,
- and how to run the new tasks.

## 1. Summary

We added a new **manager-based tracking task stack** for Unitree G1, inspired by `whole_body_tracking`.

The implementation includes:
- a new tracking module under `mbrl.tasks.manager_based.tracking`,
- G1 robot and assets under `mbrl.robots` and `mbrl.assets`,
- task registrations for init/pretrain/finetune/visualize,
- MBPO-compatible runner configs,
- and a G1 imagination wrapper for model-based finetuning.

## 2. New Task IDs

The following task IDs are now registered:

- `Tracking-Flat-G1-v0`
- `Tracking-Flat_g1-v0` (alias)
- `Template-Isaac-Tracking-Flat-G1-Init-v0`
- `Template-Isaac-Tracking-Flat-G1-Pretrain-v0`
- `Template-Isaac-Tracking-Flat-G1-Finetune-v0`
- `Template-Isaac-Tracking-Flat-G1-Visualize-v0`

Registration file:
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/__init__.py`

## 3. High-Level Design

### 3.1 Baseline Tracking Task

`Tracking-Flat-G1-v0` keeps a whole-body style tracking setup:
- motion command term (`MotionCommand`),
- anchor/body tracking terms,
- G1 body list and anchor configured in the G1 flat env config.

Main files:
- `source/mbrl/mbrl/tasks/manager_based/tracking/tracking_env_cfg.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/mdp/*`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/flat_env_cfg.py`

### 3.2 Model-Based Training Path (Preserved and Extended)

To preserve model-based dynamics training, G1 now has:
- `Pretrain` config with `system_state/system_action/system_contact/system_termination` groups,
- MBPO runner config (`class_name = "MBPOOnPolicyRunner"`),
- `Finetune` task using a custom G1 imagination env wrapper,
- `Visualize` task with paired real/imagination rollout support.

Main files:
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/envs/g1_manager_based_mbrl_env.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/envs/g1_manager_based_visualize_env.py`

## 4. Important Behavioral Choice

For MBPO pretrain/finetune, rewards/observations were simplified to be model-friendly:
- joint position tracking,
- joint velocity tracking,
- action-rate penalty,
- minimal action norm penalty.

This choice keeps imagination reward computation robust without requiring full pose reconstruction in latent rollout.

Where this is defined:
- `RewardsCfg_MBRL` and `ObservationsCfg_PRETRAIN` in
  `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/flat_env_cfg.py`

## 5. File-by-File Change Map

## 5.1 Package Wiring

- `source/mbrl/mbrl/tasks/manager_based/__init__.py`
  - now imports both locomotion and tracking modules.
- `source/mbrl/mbrl/tasks/manager_based/tracking/__init__.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/__init__.py`
  - imports `g1` to force task registration.

## 5.2 Tracking MDP Stack

Added/ported:
- `source/mbrl/mbrl/tasks/manager_based/tracking/mdp/__init__.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/mdp/commands.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/mdp/events.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/mdp/observations.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/mdp/rewards.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/mdp/terminations.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/mdp/delta_actions.py`

Additional helper terms added:
- `motion_joint_position_error_exp`
- `motion_joint_velocity_error_exp`
- `bad_joint_tracking`

## 5.3 G1 Config + Tasks

Added:
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/__init__.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/flat_env_cfg.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/agents/__init__.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/envs/__init__.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/envs/g1_manager_based_mbrl_env.py`
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/envs/g1_manager_based_visualize_env.py`

## 5.4 Robot + Assets

Added:
- `source/mbrl/mbrl/assets/__init__.py`
- `source/mbrl/mbrl/assets/unitree_description/...` (copied asset package)
- `source/mbrl/mbrl/robots/__init__.py`
- `source/mbrl/mbrl/robots/g1.py`

## 6. Motion File Resolution

Default motion file is resolved in:
- `source/mbrl/mbrl/tasks/manager_based/tracking/config/g1/flat_env_cfg.py`

Priority order:
1. `MBRL_G1_MOTION_FILE` env var (if set and exists)
2. `/home/kartikgarg/whole_body_tracking/artifacts/walk_lvl1_g1:v0/motion.npz`
3. `/home/kartikgarg/whole_body_tracking/ASAP_npz/walk_level1.npz`

## 7. How To Run

Examples:

```bash
# Baseline G1 tracking (on-policy PPO)
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Tracking-Flat_g1-v0 \
  --motion_file /absolute/path/to/motion.npz \
  --headless

# G1 dynamics pretraining (MBPO runner in pretrain mode)
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Template-Isaac-Tracking-Flat-G1-Pretrain-v0 \
  --motion_file /absolute/path/to/motion.npz \
  --headless

# G1 model-based finetune (imagined rollouts)
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Template-Isaac-Tracking-Flat-G1-Finetune-v0 \
  --headless \
  --motion_file /absolute/path/to/motion.npz \
  --system_dynamics_load_path <PATH_TO_MODEL_PT>
```

## 8. What Stayed Safe / Preserved

The existing ANYmal dynamics path remains untouched:
- no behavior changes in existing ANYmal task configs,
- no deletion/rewrite of ANYmal runners,
- only additive integration under the new tracking module.

## 9. Validation Performed

Completed:
- Python syntax compilation (`py_compile`) for all new tracking files.
- Python syntax compilation for key existing ANYmal config/training files.

Not completed in this shell:
- Runtime task-registry smoke test using `gymnasium` (missing in active Python environment).

## 10. Known Follow-Ups

Potential next improvements:
- add explicit runtime smoke test script for task instantiation,
- add README coverage for the new `--motion_file` CLI override across train/play/visualize scripts,
- add dedicated README section linking these task IDs.
