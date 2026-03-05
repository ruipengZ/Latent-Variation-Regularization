import gymnasium as gym


gym.register(
    id="Unitree-Go2-Velocity-Forward",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go2_forward_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.unitree_go2_forward_env_cfg:RobotPlayEnvCfg",
        # "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Sideway",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go2_sideway_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.unitree_go2_sideway_env_cfg:RobotPlayEnvCfg",
        # "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Unitree-Go2-Velocity-Backward",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go2_backward_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.unitree_go2_backward_env_cfg:RobotPlayEnvCfg",
        # "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)



gym.register(
    id="Isaac-Velocity-Flat-Forward-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_forward_env_cfg:UnitreeGo2FlatForwardEnvCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Backward-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_backward_env_cfg:UnitreeGo2FlatBackwardEnvCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Sideway-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_sideway_env_cfg:UnitreeGo2FlatSidewayEnvCfg",
    },
)

