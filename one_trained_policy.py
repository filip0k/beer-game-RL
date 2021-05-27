import os

import numpy as np
import ray
from gym.spaces import Box
from gym_env.envs.agent import Agent
from multiagent_env.envs import MultiAgentBeerGame
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
from ray.tune.logger import pretty_print
from heuristic_policy import HeuristicPolicy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

N_AGENTS = 4
OBSERVATIONS_TO_TRACK = 4
N_ITERATIONS = 5000


def create_env(config):
    return MultiAgentBeerGame(config)


env_config = {
    "n_agents": N_AGENTS,
    "n_iterations": N_ITERATIONS,
    "observations_to_track": OBSERVATIONS_TO_TRACK,
    'accumulate_backlog_cost': False,
    'accumulate_stock_cost': False
}
env = create_env(env_config)
register_env("mabeer-game", create_env)
obs_space = Box(low=0, high=np.finfo(np.float32).max, shape=(OBSERVATIONS_TO_TRACK * Agent.N_OBSERVATIONS,),
                dtype=np.float32)
action_space = Box(low=0, high=1000, shape=(1,), dtype=np.float32)
policies = {str(agent.name): (HeuristicPolicy, obs_space, action_space, {}) for agent in env.agents}
policies[str(env.agents[0].name)] = (None, obs_space, action_space, {})

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
trainer = PPOTrainer(env="mabeer-game", config={
    "num_workers": 0,
    "env_config": env_config,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": (lambda agent_id: agent_id),
        "policies_to_train": ['0']
    },
})

for i in range(1000):
    result = trainer.train()
    if i % 50 == 0:
        print(pretty_print(result))
        trainer.save_checkpoint('checkpoints2')

ray.shutdown()