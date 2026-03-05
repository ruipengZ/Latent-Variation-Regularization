import numpy as np
import torch
from torch.distributions import Independent, Normal

from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import PPOPolicy, DeterministicPolicy
from tianshou.utils.net.common import Net, MLP


def load_ppo(state_shape, action_shape, max_action, hidden_sizes, activation_func,
             resume_path, device,
             action_scaling=False,
             action_bound_method="clip",
             action_space=None,
             unbounded=True):

    net_a = Net(
        state_shape,
        hidden_sizes=hidden_sizes,
        activation=activation_func,
        device=device,
    )
    oracle_actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        device=device,
        unbounded=unbounded,
    ).to(device)
    net_c = Net(
        state_shape,
        hidden_sizes=hidden_sizes,
        activation=activation_func,
        device=device,
    )
    critic = Critic(net_c, device=device).to(device)
    optim = torch.optim.Adam(critic.parameters(), lr=0.1)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        oracle_actor,
        critic,
        optim,
        dist,
        discount_factor=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0,
        reward_normalization=1,
        action_scaling=action_scaling,
        action_bound_method=action_bound_method,
        lr_scheduler=None,
        action_space=action_space,
        eps_clip=0.2,
        value_clip=0,
        dual_clip=None,
        advantage_normalization=0,
        recompute_advantage=1,
    )

    # load a previous policy

    state_dict = torch.load(resume_path, map_location=device)
    policy.actor.load_state_dict(state_dict["actor"])
    policy.critic.load_state_dict(state_dict["critic"])
    print("Loaded oracle from: ", resume_path)

    return policy

def load_rsl_ppo(state_shape, action_shape, action_space, hidden_sizes, activation_func,
             resume_path, device,
                 ):

    class RSL_Actor(torch.nn.Module):
        def __init__(
                self,
                state_shape,
                action_shape,
                hidden_sizes,
                activation_func,
                device="cuda:0",
        ) -> None:
            super().__init__()
            self.device = device
            input_dim = int(np.prod(state_shape))
            self.output_dim = int(np.prod(action_shape))
            self.actor = MLP(
                input_dim,  # type: ignore
                self.output_dim,
                hidden_sizes,
                activation=activation_func,
                device=self.device
            )

        def forward(
                self,
                obs,
                state=None,
                info = {},
        ):
            logits = self.actor(obs)
            return logits, None

    actor = RSL_Actor(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        activation_func=activation_func,
        device=device,
    ).to(device)

    policy = DeterministicPolicy(
        actor,
        action_space=action_space,
    )

    # load a previous policy

    state_dict = torch.load(resume_path, map_location=device)
    actor_weights = {f"actor.model{k[5:]}": v for k, v in state_dict["model_state_dict"].items() if k.startswith("actor.")}
    load_results = policy.actor.load_state_dict(actor_weights)
    if load_results.missing_keys:
        print("Warning: Some keys are missing when loading the actor model:", load_results.missing_keys)
    print("Loaded oracle from: ", resume_path)

    return policy


def load_rsl_ppo_partial_obs(state_shape, action_shape, action_space, hidden_sizes, activation_func,
             resume_path, device,
                 ):

    class RSL_Actor(torch.nn.Module):
        def __init__(
                self,
                state_shape,
                action_shape,
                hidden_sizes,
                activation_func,
                device="cuda:0",
        ) -> None:
            super().__init__()
            self.device = device
            input_dim = int(np.prod(state_shape))
            self.output_dim = int(np.prod(action_shape))
            self.actor = MLP(
                input_dim,  # type: ignore
                self.output_dim,
                hidden_sizes,
                activation=activation_func,
                device=self.device
            )

        def forward(
                self,
                obs,
                state=None,
                info = {},
        ):
            logits = self.actor(obs[:,3:])
            return logits, None

    actor = RSL_Actor(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        activation_func=activation_func,
        device=device,
    ).to(device)

    policy = DeterministicPolicy(
        actor,
        action_space=action_space,
    )

    # load a previous policy

    state_dict = torch.load(resume_path, map_location=device)
    actor_weights = {f"actor.model{k[5:]}": v for k, v in state_dict["model_state_dict"].items() if k.startswith("actor.")}
    load_results = policy.actor.load_state_dict(actor_weights)
    if load_results.missing_keys:
        print("Warning: Some keys are missing when loading the actor model:", load_results.missing_keys)
    print("Loaded oracle from: ", resume_path)

    return policy












