from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch
from tianshou.policy import BasePolicy


class DeterministicPolicy(BasePolicy):
    """Implementation of vanilla deterministic learning.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param gym.Space action_space: env's action space.
    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.actor = model
        assert self.action_type in ["continuous", "discrete"], \
            "Please specify action_space."

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        if self.action_type == "discrete":
            act = logits.max(dim=1)[1]
        else:
            act = logits
        return Batch(logits=logits, act=act, state=hidden)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        pass
