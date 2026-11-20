from typing import Dict, Tuple, List
from chex import Array

import jax
import jax.numpy as jnp
import numpy as np
from flashbax.vault import Vault
from vault_utils.subsample_smaller import stitch_vault_from_sampled_episodes_
from vault_utils.download_vault import check_directory_exists_and_not_empty
import pickle
from os.path import exists


def get_episode_returns_and_term_idxes(offline_data: Dict[str, Array], done_flags: tuple = ("terminals",)) -> Tuple[Array, Array]:
    rewards = offline_data["rewards"][0, :, 0]
    if len(done_flags)==1:
        terminal_flag = offline_data[done_flags[0]][0, :, ...].all(axis=-1)
    elif len(done_flags)==2:
        done_1 = offline_data[done_flags[0]][0, :, ...].all(axis=-1)
        done_2 = offline_data[done_flags[1]][0, :, ...].all(axis=-1)

        terminal_flag = jnp.logical_or(done_1,done_2)

    def scan_cumsum(
        return_so_far: float, prev_term_reward: Tuple[bool, float]
    ) -> Tuple[float, float]:
        term, reward = prev_term_reward
        return_so_far = return_so_far * (1 - term) + reward
        return return_so_far, return_so_far

    xs = (terminal_flag[:-1], rewards[1:])

    _, cumsums = jax.lax.scan(scan_cumsum, rewards[0], xs)

    term_idxes = np.argwhere(terminal_flag)

    return cumsums[term_idxes - 1], term_idxes


def sort_concat(returns: Array, eps_ends: Array) -> Array:
    episode_start_idxes = eps_ends[:-1] + 1
    episode_start_idxes = jnp.insert(episode_start_idxes, 0, 0).reshape(-1, 1)
    sorting_idxes = jnp.lexsort(jnp.array([returns[:, 0]]), axis=-1)

    return_start_end = jnp.concatenate(
        [returns, episode_start_idxes.reshape(-1, 1), eps_ends], axis=-1
    )

    sorted_return_start_end = return_start_end[sorting_idxes]
    return sorted_return_start_end


def get_idxes_of_similar_subsets(
    base_returns: List, comp_returns: List, tol: float = 0.1
) -> Tuple[List, List]:
    base_selected_idxes = []
    comp_selected_idxes = []

    comp_idx = 0

    for i, ret in enumerate(base_returns):
        ret_dealt_with = False
        while (
            comp_idx < len(comp_returns)
            and (comp_returns[comp_idx] <= (ret + tol))
            and not ret_dealt_with
        ):
            if np.abs((ret - comp_returns[comp_idx])) < tol:
                base_selected_idxes.append(i)
                comp_selected_idxes.append(comp_idx)
                ret_dealt_with = True
            comp_idx += 1

    return base_selected_idxes, comp_selected_idxes


def subsample_similar(
    first_vault_info: Dict[str, str],
    second_vault_info: Dict[str, str],
    new_rel_dir: str,
    new_vault_name: str,
    done_flags: tuple = ("terminals",),
) -> None:
    if check_directory_exists_and_not_empty(f"./{new_rel_dir}/{new_vault_name}"):
        print(
            f"Vault '{new_rel_dir}/{new_vault_name.removesuffix('.vlt')}' already exists. To subsample from scratch, please remove the current subsampled vault from its directory."  # noqa
        )
        return

    return_start_end_list = []
    experience_list = []
    for vault_info in [first_vault_info, second_vault_info]:
        rel_dir = vault_info["rel_dir"]
        vault_name = vault_info["vault_name"]
        vault_uid = vault_info["uid"]

        if not check_directory_exists_and_not_empty(f"./{rel_dir}/{vault_name}"):
            print(f"Vault './{rel_dir}/{vault_name}' does not exist and cannot be subsampled.")
            return

        vlt = Vault(rel_dir=rel_dir, vault_name=vault_name, vault_uid=vault_uid)

        all_data = vlt.read()
        offline_data = all_data.experience
        del all_data

        returns, episode_end_idxes = get_episode_returns_and_term_idxes(offline_data, done_flags=done_flags)

        if not exists(f"./{rel_dir}/{vault_name}/{vault_uid}/returns.pickle"):
            with open(f"{rel_dir}/{vault_name}/{vault_uid}/returns.pickle", "wb") as f:
                pickle.dump(returns, f)

        return_start_end_list.append(sort_concat(returns, episode_end_idxes))
        experience_list.append(offline_data)

    base_ret = return_start_end_list[0][:, 0]
    comp_ret = return_start_end_list[1][:, 0]

    b, c = get_idxes_of_similar_subsets(base_ret, comp_ret, tol=0.01)

    vlt1_samples = return_start_end_list[0][b, :]
    vlt2_samples = return_start_end_list[1][c, :]

    for experience, sample_idxes, uid in zip(
        experience_list,
        [vlt1_samples, vlt2_samples],
        [first_vault_info["uid"], second_vault_info["uid"]],
    ):
        timesteps_written = stitch_vault_from_sampled_episodes_(
            experience=experience,
            len_start_end_sample=sample_idxes,
            dest_vault_name=new_vault_name,
            vault_uid=uid,
            rel_dir=new_rel_dir,
        )

        with open(f"{new_rel_dir}/{new_vault_name}/{uid}/timesteps.pickle", "wb") as f:
            pickle.dump(timesteps_written, f)

        with open(f"{new_rel_dir}/{new_vault_name}/{uid}/returns.pickle", "wb") as f:
            pickle.dump(sample_idxes[:, 0], f)

    return
