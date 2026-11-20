from collections import namedtuple
import numpy as np


def convert(dictionary):
    return namedtuple("GenericDict", dictionary.keys())(**dictionary)


class MultiAgentEnv(object):
    def __init__(self, batch_size=None, **kwargs):
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        if getattr(args, "seed", None) is not None:
            self.seed = args.seed
            self.rs = np.random.RandomState(self.seed)

    def step(self, actions):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        raise NotImplementedError

    def get_obs_size(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        raise NotImplementedError

    def get_total_actions(self):
        raise NotImplementedError

    def get_stats(self):
        raise NotImplementedError

    def get_agg_stats(self, stats):
        return {}

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info
