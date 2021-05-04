import ray
from multiagent_env.envs import MultiAgentBeerGame
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.ppo import PPOTrainer

ray.init()
trainer = PPOTrainer(env=MultiAgentBeerGame, config={
    "env_config": {},
    "num_workers": 0
})
trainer.train()
