import ray
from multiagent_env.envs import MultiAgentBeerGame
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env

ray.init()
trainer = PPOTrainer(env="mabeer-game", config={
    "env_config": {},
    "num_workers": 0
})
trainer.train()
