from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import BaseLogger, LazyLogger


class OracleTrainer(BaseTrainer):

    def __init__(
        self,
        policy: BasePolicy,
        oracle_frequency: int,
        buffer: ReplayBuffer,
        test_collector: Optional[Collector],
        max_epoch: int,
        update_per_epoch: int,
        episode_per_test: int,
        batch_size: int,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            learning_type="dagger",
            policy=policy,
            buffer=buffer,
            test_collector=test_collector,
            max_epoch=max_epoch,
            update_per_epoch=update_per_epoch,
            step_per_epoch=update_per_epoch,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            **kwargs,
        )
        self.oracle_frequency = oracle_frequency
        self.one_time_augment = False

    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform one off-line policy update."""
        assert self.buffer
        if self.one_time_augment == False:
            # if self.gradient_step !=0 and self.gradient_step % self.oracle_frequency == 0:
            if self.gradient_step % self.oracle_frequency == 0:

                self.policy.augment_buffer(
                    self.buffer
                )
                self.one_time_augment =True
        self.gradient_step += 1
        losses = self.policy.update(self.batch_size, self.buffer)
        data.update({"gradient_step": str(self.gradient_step)})
        self.log_update_data(data, losses)


def oracle_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for offline_trainer run method.

    It is identical to ``OfflineTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OracleTrainer(*args, **kwargs).run()


oracle_trainer_iter = OracleTrainer
