import gym
from gym_env.envs import BeerGame
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

gym.make('beer-game-v0')
# tune.run(PPOTrainer, config={"env": BeerGame})
