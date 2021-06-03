from ray.rllib.policy.policy import Policy
import numpy as np
from ray.rllib.utils.typing import ModelWeights


class HeuristicPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations_size = 10
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
        ## inventory - leftover_demand
        return np.array([32 - (obs_batch[0][-5] - obs_batch[0][-1])]), state_batches, {}

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights: ModelWeights) -> None:
        self.w = weights
