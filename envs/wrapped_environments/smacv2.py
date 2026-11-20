# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List

import numpy as np
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from envs.wrapped_environments.base import BaseEnvironment, ResetReturn, StepReturn

DISTRIBUTION_CONFIGS = {
    "terran_5_vs_5": {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    },
    "zerg_5_vs_5": {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    },
    "terran_10_vs_10": {
        "n_units": 10,
        "n_enemies": 10,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    },
}

MAP_NAMES = {
    "terran_5_vs_5": "10gen_terran",
    "zerg_5_vs_5": "10gen_zerg",
    "terran_10_vs_10": "10gen_terran",
}


class SMACv2(BaseEnvironment):

    def __init__(self, scenario: str, seed: int):
        self._environment = StarCraftCapabilityEnvWrapper(
            capability_config=DISTRIBUTION_CONFIGS[scenario],
            map_name=MAP_NAMES[scenario],
            debug=False,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
            seed=seed,
        )

        self.agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_agents = len(self.agents)
        self.num_actions = self._environment.n_actions

    def reset(self) -> ResetReturn:
        self._environment.reset()
        self._done = False

        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.agents)}

        env_state = self._environment.get_state().astype("float32")

        info = {"legals": legals, "state": env_state}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        smac_actions = []
        for agent in self.agents:
            smac_actions.append(actions[agent])

        reward, done, env_info = self._environment.step(smac_actions)

        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.agents)}

        env_state = self._environment.get_state().astype("float32")

        rewards = {agent: np.array(reward, "float32") for agent in self.agents}

        terminals = {agent: np.array(done) for agent in self.agents}
        truncations = {agent: np.array(False) for agent in self.agents}

        info = {"legals": legals, "state": env_state}
        if isinstance(env_info, dict) and env_info.get("battle_won", False):
            info["battle_won"] = True

        return observations, rewards, terminals, truncations, info

    def _get_legal_actions(self) -> List[np.ndarray]:
        legal_actions = []
        for i, _ in enumerate(self.agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return legal_actions

    def render(self, mode: str = "rgb_array"):
        if hasattr(self._environment, "render"):
            try:
                return self._environment.render(mode=mode)
            except TypeError:
                return self._environment.render()
        return None

    def save_replay(self, path: str):
        if hasattr(self._environment, "save_replay"):
            return self._environment.save_replay(path)
        return None

    def render(self, mode: str = "rgb_array"):
        if hasattr(self._environment, "render"):
            try:
                return self._environment.render(mode=mode)
            except TypeError:
                return self._environment.render()
        return None
