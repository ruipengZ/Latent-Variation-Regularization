import datetime
import pprint
import os, sys
sys.path.insert(0, os.path.abspath("."))
from examples.imitation.isaac_env import make_isaac_env

import argparse
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.utils.logger.wandb import WandbLogger
from tianshou.policy.imitation.base import ImitationPolicy

from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import UnboundedActor


from tianshou.data.buffer.base import ReplayBuffer

from tianshou.data.isaac_collector import Collector
from tianshou.trainer import offline_trainer

import wandb
# wandb.init(mode='disabled') # uncomment if you want to mute wandb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log/imitation")
    parser.add_argument("--algo-name", type=str, default="imitation-vanilla")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Forward-Unitree-Go2-v0")
    parser.add_argument("--dataset-path", type=str, default='imitation_data/isaac_go2_forward/traj-1.hdf5')
    parser.add_argument("--oracle-buffer-length", type=int, default=None)

    parser.add_argument("--use-tanh", type=int, default=0)
    parser.add_argument("--headless", type=int, default=1)

    parser.add_argument("--obs-norm", type=int, default=0)
    parser.add_argument("--obs-norm-path", type=str, default=None)


    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=int, default=0)

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--update-per-epoch", type=int, default=1000)

    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128, 128])

    parser.add_argument("--test-num", type=int, default=100)

    parser.add_argument("--render", type=float, default=0.)



    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--resume-path", type=str, default=None)



    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="ImitationLVR")

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_known_args()[0]
    return args


def train_imitation(args=get_args()):
    env, test_envs, simulation_app = make_isaac_env(
        args.task, args.seed, args.test_num, obs_norm=args.obs_norm, headless=args.headless,
    )
    args.state_shape = env.observation_space['policy'].shape[1:]
    args.action_shape = env.action_space.shape[1:]
    args.max_action = env.action_space.high[0]

    print("Env:", args.task)
    print("Algorithm", args.algo_name)
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

    optim = torch.optim.AdamW(actor.parameters(), lr=args.lr)

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
        policy.load_state_dict(dict["policy"])
        print("Loaded agent from: ", args.resume_path)

    if args.obs_norm == 1:
        ckpt = torch.load(args.obs_norm_path, map_location=args.device)
        test_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.update_obs_rms = False

    # buffer
    buffer = ReplayBuffer.load_hdf5(args.dataset_path)

    if args.oracle_buffer_length:
        buffer.reset_length(args.oracle_buffer_length)

    print("Replay buffer size:", len(buffer), flush=True)
    test_collector = Collector(policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )

    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        ckpt_path = os.path.join(log_path, "policy.pth")
        save_dict = \
            {
                "policy": policy.state_dict(),
                "optim": optim.state_dict(),
            }
        if args.lr_decay:
            save_dict["lr_scheduler"] = lr_scheduler.state_dict()
        torch.save(save_dict, ckpt_path)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, f"checkpoint_env{env_step}_g{gradient_step}.pth")

        save_dict = \
            {
                "policy": policy.state_dict(),
                "optim": optim.state_dict(),
            }
        if args.lr_decay:
            save_dict["lr_scheduler"] = lr_scheduler.state_dict()
        torch.save(save_dict, ckpt_path)
        return ckpt_path

    def stop_fn(mean_rewards):
        return False

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        pprint.pprint(result)
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    if args.watch:
        watch()
        exit(0)

    result = offline_trainer(
        policy,
        buffer,
        test_collector,
        args.epoch,
        args.update_per_epoch,
        args.test_num,
        args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
    )

    pprint.pprint(result)
    watch()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    train_imitation(get_args())
