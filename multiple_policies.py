import os

import numpy as np
import ray
from gym.spaces import Box
from gym_env.envs.agent import Agent
from multiagent_env.envs import MultiAgentBeerGame
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
from ray.tune.logger import pretty_print

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

N_AGENTS = 4
OBSERVATIONS_TO_TRACK = 5


def create_env(config):
    return MultiAgentBeerGame(config)


env_config = {
    "n_agents": N_AGENTS,
    "n_iterations": 1000,
    "observations_to_track": OBSERVATIONS_TO_TRACK
}
env = create_env(env_config)
register_env("mabeer-game", create_env)

obs_space = Box(low=0, high=np.finfo(np.float32).max, shape=(OBSERVATIONS_TO_TRACK * Agent.N_OBSERVATIONS,),
                dtype=np.float32)
action_space = Box(low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32)
policies = {str(agent.name): (None, obs_space, action_space, {}) for agent in env.agents}

ray.init()
trainer = PPOTrainer(env="mabeer-game", config={
    "model": {"use_lstm": True},
    "num_sgd_iter": 100,
    "sgd_minibatch_size": 250,
    "num_workers": 12,
    "env_config": env_config,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": (lambda agent_id: agent_id),
    },
})

for i in range(10):
    result = trainer.train()
    print(pretty_print(result))
