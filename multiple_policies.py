import numpy as np
import os
import ray
import tempfile
from datetime import datetime
from gym.spaces import Box
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation.metrics import summarize_episodes, collect_episodes
from ray.tune import register_env
from ray.tune.logger import pretty_print, UnifiedLogger

from gym_env.envs.agent import Agent
from heuristic_policy import HeuristicPolicy
from multiagent_env.envs import MultiAgentBeerGame

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

N_AGENTS = 4
OBSERVATIONS_TO_TRACK = 10
N_ITERATIONS = 1000


def custom_log_creator(custom_path):
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}".format(timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def create_env(config):
    return MultiAgentBeerGame(config)


obs_space = Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max,
                shape=(OBSERVATIONS_TO_TRACK * Agent.N_OBSERVATIONS,),
                dtype=np.float32)
action_space = Box(low=-8, high=8, shape=(1,), dtype=np.float32)
heuristic_action_space = Box(low=0, high=32, shape=(1,), dtype=np.float32)

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

policies = {str(agent.name): (None, obs_space, heuristic_action_space, {}) for agent in env.agents}
policies[str(env.agents[1].name)] = (HeuristicPolicy, obs_space, heuristic_action_space, {"base_stock": 24})
policies[str(env.agents[3].name)] = (HeuristicPolicy, obs_space, heuristic_action_space, {"base_stock": 8})

hp24 = HeuristicPolicy(obs_space, heuristic_action_space, {"base_stock": 24})
hp8 = HeuristicPolicy(obs_space, heuristic_action_space, {"base_stock": 8})


def train():
    ray.init()
    trainer = init_trainer()
    trainer.get_policy('0').model.base_model.summary()
    #trainer.load_checkpoint('policy0_2_retrain/checkpoint-1')
    for i in range(1000):
        result = trainer.train()
        if i % 5 == 0:
            print(pretty_print(result))
            #trainer.save_checkpoint('test')
    ray.shutdown()


def test():
    ray.init()
    trainer = init_trainer()
    #trainer.load_checkpoint('policy0_2_retrain2/checkpoint-6')

    actions = {}
    for k in range(0, 10):
        env = create_env(env_config)
        for i in range(0, 10):
            obs = env.reset()
            while not env.done:
                for j, _ in enumerate(policies):
                    if j == 0:
                        actions['0'] = trainer.get_policy('0').compute_actions([obs['0']])[0]
                    elif j == 1:
                        actions['1'] = hp24.compute_actions(obs_batch=[obs['1']])[0]
                    elif j == 2:
                        actions['2'] = trainer.get_policy('2').compute_actions([obs['2']])[0]
                    else:
                        actions['3'] = hp8.compute_actions(obs_batch=[obs['3']])[0]
                obs, reward, done, info = env.step(actions)
                actions = {}
        print(env.r0 / 10000)
        print(env.r1 / 10000)
        print(env.r2 / 10000)
        print(env.r3 / 10000)


def eval_fn(trainer, eval_workers):
    actions = {}
    env = create_env(env_config)
    for i in range(0, 10):
        obs = env.reset()
        while not env.done:
            for j, _ in enumerate(policies):
                if j == 0:
                    actions['0'] = trainer.get_policy('0').compute_actions([obs['0']])[0]
                elif j == 1:
                    actions['1'] = hp24.compute_actions(obs_batch=[obs['1']])[0]
                elif j == 2:
                    actions['2'] = trainer.get_policy('2').compute_actions([obs['2']])[0]
                else:
                    actions['3'] = hp8.compute_actions(obs_batch=[obs['3']])[0]
            obs, reward, done, info = env.step(actions)
            actions = {}
    print(env.r0 / 10000)
    print(env.r1 / 10000)
    print(env.r2 / 10000)
    print(env.r3 / 10000)
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    return summarize_episodes(episodes)


def init_trainer():
    return PPOTrainer(env="mabeer-game", logger_creator=custom_log_creator(os.path.expanduser("./ray_results")),
                      config={
                          "num_workers": 0,
                          "env_config": env_config,
                          "model": {
                              "fcnet_hiddens": [180, 130, 61]
                          },
                          'lr': 0.001,
                          'lambda': 0.9,
                          'gamma': 0.9,
                          'sgd_minibatch_size': 64,
                          'clip_param': 1.0,
                          "entropy_coeff": 0.01,
                          "vf_loss_coeff": 5e-8,
                          'num_sgd_iter': 30, "evaluation_interval": 1,
                          "evaluation_config": {
                              "explore": False
                          }, "custom_eval_function": eval_fn,
                          "multiagent": {
                              "policies": policies,
                              "policy_mapping_fn": (lambda agent_id: agent_id),
                              "policies_to_train": ["0", "2"]
                          }
                      })


if __name__ == '__main__':
    train()
