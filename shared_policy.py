import os

import numpy as np
import ray
from gym.spaces import Box
from gym_env.envs.agent import Agent
from multiagent_env.envs import MultiAgentBeerGame
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
from ray.tune.logger import pretty_print

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

N_AGENTS = 4
OBSERVATIONS_TO_TRACK = 4
N_ITERATIONS = 1000


def create_env(config):
    return MultiAgentBeerGame(config)


obs_space = Box(low=0, high=np.finfo(np.float32).max, shape=(OBSERVATIONS_TO_TRACK * Agent.N_OBSERVATIONS,),
                dtype=np.float32)
action_space = Box(low=0, high=100000, shape=(1,), dtype=np.float32)

env_config = {
    "n_agents": N_AGENTS,
    "n_iterations": N_ITERATIONS,
    "observations_to_track": OBSERVATIONS_TO_TRACK,
    'accumulate_backlog_cost': False,
    'accumulate_stock_cost': False,
    'observation_space': obs_space,
    'action_space': action_space
}
env = create_env(env_config)
register_env("mabeer-game", create_env)

ray.init()
trainer = PGTrainer(env="mabeer-game", config={
    # "train_batch_size": 200,
    # "sgd_minibatch_size": 200,
    "num_workers": 0,
    # "clip_rewards": 10000.0,
    "env_config": env_config,
})

for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))
