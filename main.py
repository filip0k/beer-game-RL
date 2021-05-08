import ray
from multiagent_env.envs import MultiAgentBeerGame
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env


def create_env(args):
    return MultiAgentBeerGame(None)


env = create_env(None)

register_env("mabeer-game", create_env)
policies = {str(agent.name): (None, env.observation_space, env.action_space, {}) for agent in env.agents}

analysis = tune.run("PPO",
                    stop={"training_iteration": 1},
                    checkpoint_freq=10,
                    reuse_actors=False,
                    checkpoint_at_end=True,
                    config={"num_sgd_iter": 1,
                            "env": "mabeer-game",
                            "env_config": {},
                            "num_workers": 0,
                            "multiagent": {
                                "policies": policies,
                                "policy_mapping_fn": (lambda agent_id: agent_id)
                            }
                            })
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean", mode='max'),
    metric="episode_reward_mean")
