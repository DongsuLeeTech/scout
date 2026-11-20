from typing import Any, Dict, Tuple

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import Node, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from gymnasium.spaces import Box, Discrete

from envs.wrapped_environments.base import BaseEnvironment, Observations, ResetReturn, StepReturn

FLATLAND_MAP_CONFIGS = {
    "3_trains": {
        "num_trains": 3,
        "num_cities": 2,
        "width": 25,
        "height": 25,
        "max_episode_len": 80,
    },
    "5_trains": {
        "num_trains": 5,
        "num_cities": 2,
        "width": 25,
        "height": 25,
        "max_episode_len": 100,
    },
    "20_trains": {
        "num_trains": 20,
        "num_cities": 3,
        "width": 30,
        "height": 30,
        "max_episode_len": 100,
    },
    "30_trains": {
        "num_trains": 30,
        "num_cities": 3,
        "width": 35,
        "height": 30,
        "max_episode_len": 100,
    },
    "40_trains": {
        "num_trains": 40,
        "num_cities": 4,
        "width": 35,
        "height": 35,
        "max_episode_len": 100,
    },
    "50_trains": {
        "num_trains": 50,
        "num_cities": 4,
        "width": 35,
        "height": 35,
        "max_episode_len": 100,
    },
}


class Flatland(BaseEnvironment):
    def __init__(self, map_name: str = "5_trains"):
        map_config = FLATLAND_MAP_CONFIGS[map_name]

        self.num_actions = 5
        self.num_agents = map_config["num_trains"]
        self._num_cities = map_config["num_cities"]
        self._map_width = map_config["width"]
        self._map_height = map_config["height"]
        self._tree_depth = 2

        self.agents = [f"{i}" for i in range(self.num_agents)]

        self.rail_generator = sparse_rail_generator(max_num_cities=self._num_cities)

        self.obs_builder = TreeObsForRailEnv(
            max_depth=self._tree_depth,
            predictor=ShortestPathPredictorForRailEnv(max_depth=20),
        )

        self._environment = RailEnv(
            width=self._map_width,
            height=self._map_height,
            number_of_agents=self.num_agents,
            rail_generator=self.rail_generator,
            line_generator=sparse_line_generator(),
            obs_builder_object=self.obs_builder,
        )

        self._obs_dim = 11 * sum(4**i for i in range(self._tree_depth + 1)) + 7

        self.action_spaces = {agent: Discrete(self.num_actions) for agent in self.agents}
        self.observation_spaces = {
            agent: Box(-np.inf, np.inf, (self._obs_dim,)) for agent in self.agents
        }

        self.info_spec = {
            "state": np.zeros((11 * self.num_agents,), "float32"),
            "legals": {
                agent: np.zeros((self.num_actions,), "int64") for agent in self.agents
            },
        }


    def render(self) -> Any:
        return self._environment.render()

    def reset(self) -> ResetReturn:
        self._done = False

        observations, info = self._environment.reset()

        legal_actions = self._get_legal_actions()

        observations = self._convert_observations(observations, info)

        state = self._make_state_representation()

        info = {"state": state, "legals": legal_actions}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        actions = {int(agent): action.item() for agent, action in actions.items()}

        next_observations, all_rewards, all_dones, info = self._environment.step(actions)

        self._done = all(list(all_dones.values()))

        rewards = {
            agent: np.array(all_rewards[int(agent)], dtype="float32")
            for agent in self.agents
        }

        legal_actions = self._get_legal_actions()

        next_observations = self._convert_observations(next_observations, info)

        state = self._make_state_representation()

        info = {"state": state, "legals": legal_actions}

        if self._done:
            num_arrived = sum(self._environment.agents[int(agent)].state == 6 for agent in self.agents)
            info["arrived"] = num_arrived

        terminals = {agent: np.array(self._done) for agent in self.agents}
        truncations = {agent: np.array(False) for agent in self.agents}

        return next_observations, rewards, terminals, truncations, info

    def _get_legal_actions(self) -> Dict[str, np.ndarray]:
        legal_actions = {}
        for agent in self.agents:
            agent_id = int(agent)
            flatland_agent = self._environment.agents[agent_id]

            if not self._environment.action_required(
                flatland_agent
            ):
                legals = np.zeros(self.num_actions, "float32")
                legals[0] = 1
            else:
                legals = np.ones(5, "float32")

            legal_actions[agent] = legals

        return legal_actions

    def _make_state_representation(self) -> np.ndarray:
        state = []
        for i, _ in enumerate(self.agents):
            agent = self._environment.agents[i]
            state.append(np.array(agent.target, "float32"))

            pos = agent.position
            if pos is None:
                pos = (-1, -1)
            if agent.state == 7:
                pos = agent.target

            state.append(np.array(pos, "float32"))

            one_hot_state = np.zeros((7,), "float32")
            one_hot_state[agent.state] = 1
            state.append(one_hot_state)
        state = np.concatenate(state)
        return state  # type: ignore

    def _convert_observations(
        self,
        observations: Dict[int, np.ndarray],
        info: Dict[str, Dict[int, np.ndarray]],
    ) -> Observations:
        new_observations = {}
        for i, agent in enumerate(self.agents):
            agent_id = i
            norm_observation = normalize_observation(
                observations[agent_id],
                tree_depth=self._tree_depth,
            )
            state = info["state"][agent_id]
            one_hot_state = np.zeros((7,), "float32")
            one_hot_state[state] = 1
            obs = np.concatenate([one_hot_state, norm_observation], axis=-1)
            new_observations[agent] = obs
        return new_observations


def find_all_cells_where_agent_can_choose(env: RailEnv):  # type: ignore
    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(env.height):
        for w in range(env.width):
            pos = (h, w)

            is_switch = False
            for orientation in directions:
                possible_transitions = env.rail.get_transitions(*pos, orientation)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                for orientation in directions:
                    possible_transitions = env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    decision_cells = switches + switches_neighbors
    return tuple(map(set, (switches, switches_neighbors, decision_cells)))


def max_lt(seq: np.ndarray, val: Any) -> Any:  # type: ignore
    max_val = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max_val:
            max_val = seq[idx]  # type: ignore
        idx -= 1
    return max_val


def min_gt(seq: np.ndarray, val: Any) -> Any:  # type: ignore
    min_val = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min_val:
            min_val = seq[idx]  # type: ignore
        idx -= 1
    return min_val


def norm_obs_clip(  # type: ignore
    obs: np.ndarray,
    clip_min: int = -1,
    clip_max: int = 1,
    fixed_radius: int = 0,
    normalize_to_range: bool = False,
) -> np.ndarray:
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)  # type: ignore
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)  # type: ignore


def _split_node_into_feature_groups(  # type: ignore
    node: Node,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(  # type: ignore
    node: Node, current_tree_depth: int, max_tree_depth: int
) -> Tuple:
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return (
            [-np.inf] * num_remaining_nodes * 6,
            [-np.inf] * num_remaining_nodes,
            [-np.inf] * num_remaining_nodes * 4,
        )

    data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(
            node.childs[direction], current_tree_depth + 1, max_tree_depth
        )
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(  # type: ignore
    tree: Node, max_tree_depth: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(
            tree.childs[direction], 1, max_tree_depth
        )
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_observation(  # type: ignore
    observation: Node, tree_depth: int, observation_radius: int = 0
) -> np.ndarray:
    if observation is None:
        return np.zeros(  # type: ignore
            11 * sum(np.power(4, i) for i in range(tree_depth + 1)),
            dtype=np.float32,
        )
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.array(
        np.concatenate((np.concatenate((data, distance)), agent_data)),
        dtype=np.float32,
    )
    return normalized_obs  # type: ignore
