import os
import gymnasium as gym

from isaaclab.app import AppLauncher
import custom_env
from tianshou.env.venvs_isaac import IsaacVectorEnv, IsaacVectorEnvNormObs



def make_isaac_env(task, seed, training_num, obs_norm, device='cuda', headless=True,
                   render_video=False, render_gif=False, video_kwargs=None):
    if headless:
        os.environ["HEADLESS"] = "1"
    if render_video or render_gif:
        os.environ["ENABLE_CAMERAS"] = "1"


    app_launcher = AppLauncher({'enable_cameras': render_video})
    simulation_app = app_launcher.app

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    args_dict = {
        "task": task,
        "num_train_envs": training_num,
        "device": device,


    }
    train_env_cfg = parse_env_cfg(
        args_dict['task'], device=args_dict['device'], num_envs=args_dict['num_train_envs']
    )

    env = gym.make(args_dict['task'], cfg=train_env_cfg, render_mode="rgb_array" if render_video or render_gif else None)
    if render_gif:
        print("[INFO] Recording gifs during training.")
        env = gym.wrappers.RenderCollection(env)

    elif render_video:
        video_dict = {
            "video_folder": os.path.join(video_kwargs['video_dir'], "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_kwargs['video_length'],
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_dict)

    train_envs = IsaacVectorEnv(args_dict['num_train_envs'], env)
    train_envs.seed(seed)

    if obs_norm:
        # obs norm wrapper
        train_envs = IsaacVectorEnvNormObs(train_envs)


    return env, train_envs, simulation_app