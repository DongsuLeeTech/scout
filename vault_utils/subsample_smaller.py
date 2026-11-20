from typing import Optional

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import flashbax as fbx
from flashbax.vault import Vault
import flashbax
from vault_utils.download_vault import (
    check_directory_exists_and_not_empty,
    get_available_uids,
)

from typing import Dict
from chex import Array

def get_length_start_end(experience: Dict[str, Array], done_flags: tuple = ("terminals",)) -> Array:
    if len(done_flags)==1:
        terminal_flag = experience[done_flags[0]][0, :, ...].all(axis=-1)
    elif len(done_flags)==2:
        done_1 = experience[done_flags[0]][0, :, ...].all(axis=-1)
        done_2 = experience[done_flags[1]][0, :, ...].all(axis=-1)

        terminal_flag = jnp.logical_or(done_1,done_2)

    term_idxes = np.argwhere(terminal_flag)

    start_idxes = np.zeros_like(term_idxes)
    start_idxes[1:] = term_idxes[:-1]

    lengths = term_idxes - start_idxes

    len_start_end = np.concatenate((lengths, start_idxes, term_idxes), axis=1)

    return len_start_end


def select_episodes_uniformly_up_to_n_transitions(len_start_end: Array, n: int) -> Array:
    shuffled_idxes = np.arange(len_start_end.shape[0])
    np.random.shuffle(shuffled_idxes)

    shuffled_lengths = len_start_end[shuffled_idxes, 0]
    shuffled_cumsum_of_eps_lengths = np.cumsum(shuffled_lengths)

    selected_idxes = shuffled_idxes[np.where(shuffled_cumsum_of_eps_lengths < n)]
    print(shuffled_cumsum_of_eps_lengths[-1])

    randomly_sampled_len_start_end = len_start_end[selected_idxes, :]

    return randomly_sampled_len_start_end


def stitch_vault_from_sampled_episodes_(
    experience: Dict[str, Array],
    len_start_end_sample: Array,
    dest_vault_name: str,
    vault_uid: str,
    rel_dir: str,
    n: int = 500_000,
) -> int:
    if check_directory_exists_and_not_empty(f"{rel_dir}/{dest_vault_name}/{vault_uid}/"):
        print(f"Vault '{rel_dir}/{dest_vault_name}.vlt/{vault_uid}' already exists.")
        return 0

    dest_buffer = fbx.make_trajectory_buffer(
        sample_batch_size=1,
        sample_sequence_length=1,
        period=1,
        max_length_time_axis=n,
        min_length_time_axis=100,
        add_batch_size=experience["actions"].shape[0],
    )

    dummy_experience = jax.tree_map(lambda x: x[0, 0, ...], experience)

    dest_state = dest_buffer.init(dummy_experience)
    buffer_add = jax.jit(dest_buffer.add, donate_argnums=0)
    dest_vault = flashbax.vault.Vault(
        experience_structure=dest_state.experience,
        vault_name=dest_vault_name,
        vault_uid=vault_uid,
        rel_dir=rel_dir,
    )

    for start, end in zip(len_start_end_sample[:, 1], len_start_end_sample[:, 2]):
        sample_experience = jax.tree_util.tree_map(
            lambda x: x[:, int(start) : int(end + 1), ...],  # noqa
            experience,
        )
        dest_state = buffer_add(dest_state, sample_experience)

    timesteps_written = dest_vault.write(dest_state)

    print(timesteps_written)

    return int(timesteps_written)


def subsample_smaller_vault(
    vaults_dir: str,
    vault_name: str,
    vault_uids: Optional[list] = None,
    target_number_of_transitions: int = 500000,
) -> str:
    if not check_directory_exists_and_not_empty(f"./{vaults_dir}/{vault_name}"):
        print(f"Vault './{vaults_dir}/{vault_name}' does not exist and cannot be subsampled.")
        return f"./{vaults_dir}/{vault_name}"

    if vault_uids is None:
        vault_uids = get_available_uids(f"./{vaults_dir}/{vault_name}")

    new_vault_name = (
        vault_name.removesuffix(".vlt") + "_" + str(target_number_of_transitions) + ".vlt"
    )

    if check_directory_exists_and_not_empty(f"./{vaults_dir}/{new_vault_name}"):
        print(
            f"Vault '{vaults_dir}/{new_vault_name.removesuffix('.vlt')}' already exists. To subsample from scratch, please remove the current subsampled vault from its directory."  # noqa
        )
        return f"./{vaults_dir}/{vault_name}"

    for vault_uid in vault_uids:
        vlt = Vault(rel_dir=vaults_dir, vault_name=vault_name, vault_uid=vault_uid)

        all_data = vlt.read()
        offline_data = all_data.experience

        len_start_end = get_length_start_end(offline_data)

        len_start_end_sample = select_episodes_uniformly_up_to_n_transitions(
            len_start_end, target_number_of_transitions
        )

        timesteps_written = stitch_vault_from_sampled_episodes_(
            offline_data,
            len_start_end_sample,
            new_vault_name,
            vault_uid,
            vaults_dir,
            target_number_of_transitions,
        )

        with open(f"{vaults_dir}/{new_vault_name}/{vault_uid}/timesteps.pickle", "wb") as f:
            pickle.dump(timesteps_written, f)

    return new_vault_name
