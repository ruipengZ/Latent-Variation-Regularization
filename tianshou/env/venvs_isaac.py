from typing import Any, List, Optional, Tuple, Union

import gym
import numpy as np
import torch

from tianshou.env.utils import ENV_TYPE
from tianshou.env.venvs import GYM_RESERVED_KEYS
from tianshou.utils import RunningMeanStd


class IsaacVectorEnv(object):

    def __init__(
        self,
        num_envs: int,
        env: ENV_TYPE,
        timeout: Optional[float] = None,
        fast_variant=False,
    ) -> None:
        self.env = env
        self.num_envs = num_envs # todo: change to actual parallel env number
        self.timeout = timeout
        assert (
            self.timeout is None or self.timeout > 0
        ), f"timeout is {timeout}, it should be positive if provided!"

        self.is_closed = False
        self.is_async = False
        self.fast_variant = fast_variant

    def _assert_is_not_closed(self) -> None:
        assert (
            not self.is_closed
        ), f"Methods of {self.__class__.__name__} cannot be called after close."

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.num_envs

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.

        Any class who inherits ``gym.Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return self.get_env_attr(key)
        else:
            return super().__getattribute__(key)


    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped


    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> List[Any]:
        # resolve indices
        if id is None:
            id = list(range(self.num_envs))
            num_indices = self.num_envs
        else:
            num_indices = len(id)
        # obtain attribute value
        attr_val = getattr(self.env, key)

        # return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[id].detach().cpu().numpy()


    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> None:
        raise NotImplementedError("Setting attributes is not supported.")


    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Union[dict, List[dict]]]:
        """Reset the state of some envs and return initial observations.

        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        obs_dict, _ = self.env.reset(seed=kwargs["seed"] if "seed" in kwargs else None)
        obs_dict = self._process_obs(obs_dict)
        if id is not None:
            obs_dict = obs_dict[id]
            infos = [{} for _ in range(len(id))]
        else:
            infos = [{} for _ in range(self.num_envs)]
        # print(obs_dict)
        return obs_dict, infos

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        obs = obs_dict["policy"]
        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            for key, value in obs.items():
                obs[key] = value.detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs


    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ):
        """Run one timestep of some environments' dynamics.

        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.

        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple consisting of either:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``terminated`` a numpy.ndarray, whether these episodes have been \
                terminated
            * ``truncated`` a numpy.ndarray, whether these episodes have been truncated
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        # print(action.shape)
        # if id is not None:
        #     print(id.shape,self.num_envs)
        # todo: remap action to original env ids, and fill with zeros
        if id is not None:
            action_full = np.zeros((self.num_envs,) + action.shape[1:], dtype=action.dtype)
            action_full[id] = action
        else:
            action_full = action

        # convert input to numpy array
        if not isinstance(action_full, torch.Tensor):
            action_full = np.asarray(action_full)
            action_full = torch.from_numpy(action_full).to(device=self.env.unwrapped.device, dtype=torch.float32)
        else:
            action_full = action_full.to(device=self.env.unwrapped.device, dtype=torch.float32)

        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(action_full)
        # compute reset ids
        dones = terminated | truncated

        # convert data types to numpy depending on backend
        # note: ManagerBasedRLEnv uses torch backend (by default).
        obs = self._process_obs(obs_dict)
        rewards = rew.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()

        reset_ids = dones.nonzero()[0]

        # convert extra information to list of dicts
        infos = self._process_extras(id, obs, terminated, truncated, extras, reset_ids)


        return obs[id], rewards[id], terminated[id], truncated[id], infos



    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> List[Optional[List[int]]]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        if seed is None:
            seed_list = [-1] * self.num_envs
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.num_envs)]
        else:
            seed_list = seed
        return [self.unwrapped.seed(seed) for seed in seed_list]

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        self._assert_is_not_closed()
        if method_name == "render":
            # gymnasium does not support changing render mode at runtime
            return self.env.render()
        else:
            # this isn't properly implemented but it is not necessary.
            # mostly done for completeness.
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def render(self, **kwargs: Any):
        """Render all of the environments."""
        self._assert_is_not_closed()
        return self.env.render()


    def close(self) -> None:
        """Close all of the environments.

        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        self.env.close()
        self.is_closed = True


    def _process_extras(
            self, indices, obs: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, extras: dict, reset_ids: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Convert miscellaneous information into dictionary for each sub-environment."""
        # faster version: only process env that terminated and add bootstrapping info
        if self.fast_variant:
            infos = [{} for _ in range(self.num_envs)]

            for idx in reset_ids:
                # fill-in episode monitoring info
                # infos[idx]["episode"] = {
                #     "r": self._ep_rew_buf[idx],
                #     "l": self._ep_len_buf[idx],
                # }

                # fill-in bootstrap information
                infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]

                # # add information about terminal observation separately
                # if isinstance(obs, dict):
                #     terminal_obs = {key: value[idx] for key, value in obs.items()}
                # else:
                #     terminal_obs = obs[idx]
                # infos[idx]["terminal_observation"] = terminal_obs

            return [infos[i] for i in indices]

        # create empty list of dictionaries to fill
        infos: list[dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.num_envs)]
        # fill-in information for each sub-environment
        # note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # fill-in episode monitoring info
            # if idx in reset_ids:
            #     infos[idx]["episode"] = dict()
            #     infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
            #     infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            # else:
            #     infos[idx]["episode"] = None
            # fill-in bootstrap information
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # fill-in information from extras
            # for key, value in extras.items():
            #     # 1. remap extra episodes information safely
            #     # 2. for others just store their values
            #     if key == "log":
            #         # only log this data for episodes that are terminated
            #         if infos[idx]["episode"] is not None:
            #             for sub_key, sub_value in value.items():
            #                 infos[idx]["episode"][sub_key] = sub_value
            #     else:
            #         infos[idx][key] = value[idx]
            # add information about terminal observation separately
            # if idx in reset_ids:
            #     # extract terminal observations
            #     if isinstance(obs, dict):
            #         terminal_obs = dict.fromkeys(obs.keys())
            #         for key, value in obs.items():
            #             terminal_obs[key] = value[idx]
            #     else:
            #         terminal_obs = obs[idx]
            #     # add info to dict
            #     infos[idx]["terminal_observation"] = terminal_obs
            # else:
            #     infos[idx]["terminal_observation"] = None
        # return list of dictionaries
        return [infos[i] for i in indices]


class IsaacVectorEnvWrapper(IsaacVectorEnv):
    def __init__(self, venv: IsaacVectorEnv) -> None:
        self.venv = venv
        self.is_async = venv.is_async

    def __len__(self) -> int:
        return len(self.venv)

    def __getattribute__(self, key: str) -> Any:
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return getattr(self.venv, key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> List[Any]:
        return self.venv.get_env_attr(key, id)

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> None:
        return self.venv.set_env_attr(key, value, id)

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Union[dict, List[dict]]]:
        return self.venv.reset(id, **kwargs)

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ):
        return self.venv.step(action, id)

    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> List[Optional[List[int]]]:
        return self.venv.seed(seed)

    def render(self, **kwargs: Any) -> List[Any]:
        return self.venv.render(**kwargs)

    def close(self) -> None:
        self.venv.close()



class IsaacVectorEnvNormObs(IsaacVectorEnvWrapper):
    """An observation normalization wrapper for vectorized environments.

    :param bool update_obs_rms: whether to update obs_rms. Default to True.
    """

    def __init__(
        self,
        venv: IsaacVectorEnv,
        update_obs_rms: bool = True,
    ) -> None:
        super().__init__(venv)
        # initialize observation running mean/std
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd()

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Union[dict, List[dict]]]:
        obs, info = self.venv.reset(id, **kwargs)

        if isinstance(obs, tuple):  # type: ignore
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        obs = self._norm_obs(obs)
        return obs, info

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ):
        step_results = self.venv.step(action, id)
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(step_results[0])
        return (self._norm_obs(step_results[0]), *step_results[1:])

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms:
            return self.obs_rms.norm(obs)  # type: ignore
        return obs

    def set_obs_rms(self, obs_rms: RunningMeanStd) -> None:
        """Set with given observation running mean/std."""
        self.obs_rms = obs_rms

    def get_obs_rms(self) -> RunningMeanStd:
        """Return observation running mean/std."""
        return self.obs_rms