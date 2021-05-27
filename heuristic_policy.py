from ray.rllib.policy.policy import Policy
import numpy as np
from ray.rllib.utils.typing import ModelWeights


class HeuristicPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()
        self.w = 1

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return np.array([np.minimum(obs_batch[0][2], 10000)]), state_batches, {}

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights: ModelWeights) -> None:
        self.w = weights
