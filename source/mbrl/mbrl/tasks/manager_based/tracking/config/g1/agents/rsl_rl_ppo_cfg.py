from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from mbrl.rl.rsl_rl import (
    RslRlMbrlImaginationCfg,
    RslRlMbrlPpoAlgorithmCfg,
    RslRlNormalizerCfg,
    RslRlSystemDynamicsCfg,
)


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_tracking_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)


@configclass
class G1FlatPPOPretrainRunnerCfg(G1FlatPPORunnerCfg):
    class_name: str = "MBPOOnPolicyRunner"
    run_name = "pretrain"
    max_iterations = 5000
    save_interval = 100

    system_dynamics = RslRlSystemDynamicsCfg(
        ensemble_size=1,
        history_horizon=32,
        architecture_config={
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
            "state_mean_shape": [128],
            "state_logstd_shape": [128],
            "extension_shape": [128],
            "contact_shape": [128],
            "termination_shape": [128],
        },
        freeze_auxiliary=False,
    )
    imagination = RslRlMbrlImaginationCfg(
        num_envs=0,
        num_steps_per_env=0,
        max_episode_length=0,
        command_resample_interval_range=None,
        uncertainty_penalty_weight=-0.0,
        state_normalizer=RslRlNormalizerCfg(
            mean=[
                0.0,
                0.0,
                0.0,  # base_lin_vel
                0.0,
                0.0,
                0.0,  # base_ang_vel
                0.0,
                0.0,
                -1.0,  # projected_gravity
            ]
            + [0.0] * 29
            + [0.0] * 29
            + [0.0] * 29,
            std=[1.0] * 96,
        ),
        action_normalizer=RslRlNormalizerCfg(
            mean=[0.0] * 29,
            std=[1.0] * 29,
        ),
    )
    algorithm = RslRlMbrlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        policy_learning_rate=1.0e-3,
        system_dynamics_learning_rate=1.0e-3,
        system_dynamics_weight_decay=0.0,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        system_dynamics_forecast_horizon=8,
        system_dynamics_loss_weights={
            "state": 1.0,
            "sequence": 1.0,
            "bound": 1.0,
            "kl": 0.1,
            "extension": 1.0,
            "contact": 1.0,
            "termination": 1.0,
        },
        system_dynamics_num_mini_batches=20,
        system_dynamics_mini_batch_size=5000,
        system_dynamics_replay_buffer_size=1000,
        system_dynamics_num_eval_trajectories=64,
        system_dynamics_len_eval_trajectory=300,
        system_dynamics_eval_traj_noise_scale=[0.1, 0.2, 0.4, 0.5, 0.8],
    )
    load_system_dynamics = False
    system_dynamics_load_path = None
    system_dynamics_warmup_iterations = 0
    system_dynamics_num_visualizations = 4
    system_dynamics_state_idx_dict = {
        r"$v$": [0, 1, 2],
        r"$\omega$": [3, 4, 5],
        r"$g$": [6, 7, 8],
        r"$q$": list(range(9, 38)),
        r"$\dot{q}$": list(range(38, 67)),
        r"$\tau$": list(range(67, 96)),
    }
    pca_obs_buf_size = 10000


@configclass
class G1FlatPPOFinetuneRunnerCfg(G1FlatPPOPretrainRunnerCfg):
    resume = True
    load_system_dynamics = True
    run_name = "finetune"

    def __post_init__(self):
        super().__post_init__()
        self.imagination.num_envs = 4096
        self.imagination.num_steps_per_env = 24
        self.imagination.max_episode_length = 256
        self.imagination.command_resample_interval_range = None
        self.imagination.uncertainty_penalty_weight = -0.0


@configclass
class G1FlatPPOVisualizeRunnerCfg(G1FlatPPOPretrainRunnerCfg):
    resume = True
    load_system_dynamics = True
    run_name = "visualize"
