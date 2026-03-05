import os, sys
sys.path.insert(0, os.path.abspath("."))
from examples.imitation.isaac_env import make_isaac_env

import argparse
import pprint

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import UnboundedActor

from tianshou.policy.imitation.base import ImitationPolicy


from tianshou.data.isaac_render_collector import Collector, save_frames_as_gif

import wandb
wandb.init(mode='disabled')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8889)

    parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Forward-Unitree-Go2-v0")
    parser.add_argument("--resume-path", type=str, default=None)  ## ours

    parser.add_argument("--use-tanh", type=int, default=0)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--render-gif", type=int, default=0)

    parser.add_argument("--obs-norm", type=int, default=0)
    parser.add_argument("--obs-norm-path", type=str, default=None)


    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--update-per-epoch", type=int, default=10000)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128, 128])

    parser.add_argument("--test-num", type=int, default=100)

    parser.add_argument("--render-video", type=float, default=0.)



    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )

    parser.add_argument("--buffer-size", type=int, default=1000)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_known_args()[0]
    return args


def train_imitation(args=get_args()):
    env, test_envs, simulation_app = make_isaac_env(
        args.task, args.seed, args.test_num, obs_norm=args.obs_norm, headless=args.headless,
        render_gif=args.render_gif,
        # render_video=args.render_video > 0, video_kwargs={'video_dir': os.path.dirname(args.resume_path), 'video_length': 1000}
    )
    args.state_shape = env.observation_space['policy'].shape[1:]
    args.action_shape = env.action_space.shape[1:]
    args.max_action = env.action_space.high[0]

    print("Env:", args.task)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # model
    net = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=torch.nn.Tanh if args.use_tanh else torch.nn.ELU,
        device=args.device,
    )
    actor = UnboundedActor(
        net,
        args.action_shape,
        device=args.device,
    ).to(args.device)

    optim = torch.optim.Adam(actor.parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )


    # define policy
    policy = ImitationPolicy(
        actor,
        optim,
        action_space=env.action_space,
        lr_scheduler=lr_scheduler,
    ).to(args.device)

    # load a previous policy
    if args.resume_path:
        dict = torch.load(args.resume_path, map_location=args.device)
        print(dict.keys())
        policy.load_state_dict(dict["policy"])
        print("Loaded agent from: ", args.resume_path)

    if args.obs_norm == 1:
        ckpt = torch.load(args.obs_norm_path, map_location=args.device)
        test_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.update_obs_rms = False


    if args.render_gif:
        # collector
        test_collector = Collector(policy, test_envs)
        print("Setup test envs ...")
        policy.eval()

        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=1, render=0.001)
        pprint.pprint(result)
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')


        save_frames_as_gif(test_collector.render_frames, path=os.path.dirname(args.resume_path),
                           filename=f'traj-rew_{result["rews"].mean()}-len_{result["lens"].mean()}.gif')


    else:
        test_collector = Collector(policy, test_envs)
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=0)
        pprint.pprint(result)
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')


    env.close()
    simulation_app.close()


if __name__ == "__main__":
    train_imitation(get_args())
