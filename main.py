from gym.spaces import Box, Discrete
from multiagent_env.envs import MultiAgentBeerGame
from ray import tune
from ray.tune import register_env


def create_env(args):
    return MultiAgentBeerGame(None)


env = create_env(None)
register_env("mabeer-game", create_env)

obs_space = Box(low=0, high=float('inf'), shape=(28,))
action_space = Box(low=0, high=float('inf'), shape=(1,))
policies = {str(agent.name): (None, obs_space, Discrete(1), {}) for agent in env.agents}

analysis = tune.run("PPO",
                    stop={"training_iteration": 1},
                    reuse_actors=False,
                    checkpoint_at_end=True,
                    config={"env": MultiAgentBeerGame,
                            "env_config": {},
                            "num_workers": 0,
                            # "multiagent": {
                            #     "policies": policies,
                            #     "policy_mapping_fn": (lambda agent_id: agent_id)
                            # }
                            })

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean", mode='max'),
    metric="episode_reward_mean")
