import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import ModelWeights


class HeuristicPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_stock = args[2]['base_stock']
        # self.exploration = self._create_exploration()
        self.w = 1

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        if obs_batch[0][-5] > 0:
            level = obs_batch[0][-5]
        else:
            level = - obs_batch[0][-4]
        decision = self.base_stock - level - obs_batch[0][-1]
        return np.array([decision]), state_batches, {}

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights: ModelWeights) -> None:
        self.w = weights
