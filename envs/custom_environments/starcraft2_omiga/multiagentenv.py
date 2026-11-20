from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class MultiAgentEnv(object):
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

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "obs_alone_shape": self.get_obs_alone_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info
