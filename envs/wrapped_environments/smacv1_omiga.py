from typing import Dict, List

import numpy as np
from dataclasses import dataclass
from envs.custom_environments.starcraft2_omiga.StarCraft2_Env import StarCraft2Env

from .base import BaseEnvironment, ResetReturn, StepReturn


@dataclass
class OmigaConf:
    map_name: str
    add_move_state: bool = False
    add_local_obs: bool = False
    add_distance_state: bool = False
    add_enemy_action_state: bool = False
    add_visible_state: bool = False
    add_xy_state: bool = False
    add_agent_id: bool = True
    use_state_agent: bool = True
    use_mustalive: bool = True
    add_center_xy: bool = True
    use_stacked_frames: bool = False
    stacked_frames: int = 1
    use_obs_instead_of_state: bool = False


class SMACv1OMIGA(BaseEnvironment):

    def __init__(
        self,
        map_name: str,
        seed: int = 0,
    ):
        self._environment = StarCraft2Env(
            args=OmigaConf(map_name=map_name),
            seed=seed,
        )
        self.agents = [f"agent_{n}" for n in range(self._environment.n_agents)]

        self.num_agents = len(self.agents)
        self.num_actions = self._environment.n_actions

    def reset(self) -> ResetReturn:
        self._environment.reset()
        self.done = False

        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.agents)}

        env_state = np.concatenate(
            [self._environment.get_state(agent_id=i) for i in range(len(self.agents))],
            axis=-1,
        ).astype("float32")

        info = {"legals": legals, "state": env_state}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        smac_actions = []
        for agent in self.agents:
            smac_actions.append(actions[agent])

        o, g, r, d, agent_infos, ava = self._environment.step(smac_actions)

        observations = {agent: o[idx] for idx, agent in enumerate(self.agents)}
        rewards = {
            agent: np.array(r[idx], "float32").squeeze(-1) for idx, agent in enumerate(self.agents)
        }
        terminals = {agent: np.array(d[idx]) for idx, agent in enumerate(self.agents)}
        truncations = {agent: np.array(False) for agent in self.agents}
        legals = {agent: np.array(ava[idx], "int32") for idx, agent in enumerate(self.agents)}
        agent_states = {agent: np.array(g[idx], "float32") for idx, agent in enumerate(self.agents)}
        env_state = np.concatenate([agent_states[agent] for agent in self.agents], axis=-1)
        info = {"legals": legals, "state": env_state}

        if agent_infos and len(agent_infos) > 0:
            first_info = agent_infos[0]
            if isinstance(first_info, dict) and first_info.get("won", False):
                info["battle_won"] = True

        return observations, rewards, terminals, truncations, info

    def _get_legal_actions(self) -> List[np.ndarray]:
        legal_actions = []
        for i, _ in enumerate(self.agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return legal_actions
