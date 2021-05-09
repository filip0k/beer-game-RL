import os
import sys
import numpy as np

import ray
from gym.spaces import Box
from multiagent_env.envs import MultiAgentBeerGame
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy, PPOTFPolicy
from ray.tune import register_env
from ray.tune.logger import pretty_print

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_env(args):
    return MultiAgentBeerGame(None)


env = create_env(None)
register_env("mabeer-game", create_env)

obs_space = Box(low=0, high=np.finfo(np.float32).max, shape=(28,), dtype=np.float32)
action_space = Box(low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32)
policies = {str(agent.name): (None, obs_space, action_space, {}) for agent in env.agents}

ray.init()
trainer = PPOTrainer(env="mabeer-game", config={
    "env_config": {},
    "num_workers": 0,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": (lambda agent_id: agent_id),
    },
})

result = trainer.train()
print(pretty_print(result))
