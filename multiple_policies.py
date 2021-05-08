import ray
from multiagent_env.envs import MultiAgentBeerGame
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env


def create_env(args):
    return MultiAgentBeerGame(None)


env = create_env(None)

register_env("mabeer-game", create_env)
policies = {str(agent.name): (None, env.observation_space, env.action_space, {}) for agent in env.agents}

ray.init()
trainer = PPOTrainer(env="mabeer-game", config={
    "env_config": {},
    "num_workers": 0, "multiagent": {
        "policies": policies,
        "policy_mapping_fn": (lambda agent_id: agent_id),
    },
})
trainer.train()
