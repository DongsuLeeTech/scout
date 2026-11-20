from typing import Dict, List, Tuple, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from chex import Array
from flashbax.vault import Vault
import numpy as np
from vault_utils.download_vault import get_available_uids
from tabulate import tabulate


def get_structure_descriptors(
    experience: Dict[str, Array], n_head: int = 1, done_flags: tuple = ("terminals",),
) -> Tuple[Dict[str, Array], Dict[str, Array], int]:
    struct = jax.tree_map(lambda x: x.shape, experience)

    head = jax.tree_map(lambda x: x[0, :n_head, ...], experience)

    if len(done_flags)==1:
        terminal_flag = experience[done_flags[0]][0, :, ...].all(axis=-1)
    elif len(done_flags)==2:
        done_1 = experience[done_flags[0]][0, :, ...].all(axis=-1)
        done_2 = experience[done_flags[1]][0, :, ...].all(axis=-1)

        terminal_flag = jnp.logical_or(done_1,done_2)
    else:
        print("Too many done flags. Please revise.")
        return struct, head, 0

    num_episodes = int(jnp.sum(terminal_flag))

    return struct, head, num_episodes


def describe_structure(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    n_head: int = 0,
    done_flags: tuple = ("terminals",),
) -> Dict[str, Array]:
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    heads = {}
    structs = {}
    single_values = []
    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience
        n_trans = exp["actions"].shape[1]

        struct, head, n_traj = get_structure_descriptors(exp, n_head, done_flags)

        print(str(uid) + "\n-----")
        for key, val in struct.items():
            print(f"{str(key)+':': <15}{str(val): >15}")
        print("\n")

        heads[uid] = head
        structs[uid] = struct

        single_values.append([uid, n_trans, n_traj])

    print(tabulate(single_values, headers=["Uid", "Transitions", "Trajectories"]))

    return heads


def get_episode_return_descriptors(
    experience: Dict[str, Array], done_flags: tuple = ("terminals",),
) -> Tuple[float, float, float, float, Array]:
    episode_returns = calculate_returns(experience, done_flags = done_flags)

    mean = jnp.mean(episode_returns)
    stddev = jnp.std(episode_returns)

    return mean, stddev, jnp.max(episode_returns), jnp.min(episode_returns), episode_returns


def plot_eps_returns_violin(
    all_uid_returns: Dict[str, Array], vault_name: str, save_path: str = ""
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    sns.violinplot(data=list(all_uid_returns.values()), inner="point")
    plt.xlabel("Dataset Quality")
    plt.ylabel("Episode Returns")
    plt.xticks(range(len(all_uid_returns)), list(all_uid_returns.keys()))
    if len(save_path) > 0:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.title(f"Violin Distributions of Returns for {vault_name}")
    plt.show()
    return


def plot_eps_returns_hist(
    all_uid_returns: Dict[str, Array],
    vault_name: str,
    n_bins: int,
    min_return: float,
    max_return: float,
    save_path: str = "",
) -> None:
    vault_uids = list(all_uid_returns.keys())

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(
        1,
        len(vault_uids),
        figsize=(0.5 + 2.5 * len(vault_uids), 3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    colors = sns.color_palette()

    for i, uid in enumerate(vault_uids):
        counts, bins = np.histogram(
            all_uid_returns[uid], bins=n_bins, range=(min_return, max_return)
        )
        ax[0, i].stairs(counts, bins, fill=True, color=colors[i])
        ax[0, i].set_title(uid)
        ax[0, i].set_xlabel("Episode return")
    ax[0, 0].set_ylabel("Frequency")
    fig.tight_layout()
    if len(save_path) > 0:
        plt.savefig(save_path, bbox_inches="tight")
    fig.suptitle(f"Histogram of distributions of episode returns for {vault_name}")
    fig.tight_layout()
    plt.show()
    return


def describe_episode_returns(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    plot_hist: bool = True,
    save_hist: bool = False,
    plot_violin: bool = True,
    save_violin: bool = False,
    plot_saving_rel_dir: str = "vaults",
    n_bins: Optional[int] = 50,
    done_flags: tuple = ("terminals",),
) -> None:
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    single_values = []
    all_uid_eps_returns = {}
    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience

        mean, stddev, max_ret, min_ret, episode_returns = get_episode_return_descriptors(exp, done_flags)
        all_uid_eps_returns[uid] = episode_returns

        single_values.append([uid, mean, stddev, max_ret, min_ret])

    print(tabulate(single_values, headers=["Uid", "Mean", "Stddev", "Max", "Min"]))

    if plot_saving_rel_dir == "vaults":
        plot_saving_rel_dir = rel_dir

    if plot_hist:
        min_of_all = min([x[-1] for x in single_values])
        max_of_all = max([x[-2] for x in single_values])
        if save_hist:
            plot_eps_returns_hist(
                all_uid_eps_returns,
                vault_name,
                n_bins,
                min_of_all,
                max_of_all,
                save_path=f"{plot_saving_rel_dir}/{vault_name.removesuffix('.vlt')}_histogram.pdf",
            )
        else:
            plot_eps_returns_hist(all_uid_eps_returns, vault_name, n_bins, min_of_all, max_of_all)

    if plot_violin:
        if save_violin:
            plot_eps_returns_violin(
                all_uid_eps_returns,
                vault_name,
                save_path=f"{plot_saving_rel_dir}/{vault_name.removesuffix('.vlt')}_violin_plot.pdf",
            )
        else:
            plot_eps_returns_violin(all_uid_eps_returns, vault_name)

    return


def calculate_returns(
    experience: Dict[str, Array], reward_key: str = "rewards", done_flags: tuple = ("terminals",),
) -> Array:
    experience_one_agent = jax.tree_map(lambda x: x[0, :, 0, ...], experience)
    rewards = experience_one_agent[reward_key]
    
    if len(done_flags)==1:
        terminals = jnp.array(experience[done_flags[0]][0].all(axis=-1).squeeze(), dtype=jnp.float32)
    elif len(done_flags)==2:
        done_1 = jnp.array(experience[done_flags[0]][0].all(axis=-1).squeeze(), dtype=jnp.float32)
        done_2 = jnp.array(experience[done_flags[1]][0].all(axis=-1).squeeze(), dtype=jnp.float32)
        terminals = jnp.logical_or(done_1,done_2).astype(jnp.float32)

    def sum_rewards(terminals: Array, rewards: Array) -> Array:
        def scan_fn(carry: Array, inputs: Array) -> Array:
            terminal, reward = inputs
            new_carry = carry + reward
            new_carry = jax.lax.cond(
                terminal.all(),
                lambda _: reward,
                lambda _: new_carry,
                operand=None,
            )
            return new_carry, new_carry

        _, cumulative_rewards = jax.lax.scan(
            scan_fn,
            jnp.zeros_like(terminals[0]),
            (
                jnp.pad(terminals, ((1, 0)))[:-1],
                rewards,
            ),
        )
        return cumulative_rewards[terminals == 1]

    episode_returns = sum_rewards(terminals, rewards)
    return episode_returns


def get_saco(experience: Dict[str, Array]) -> Tuple[float, Array, Array]:
    states = experience["infos"]["state"]

    num_tot = states.shape[1]

    reshaped_actions = experience["actions"].reshape((*experience["actions"].shape[:2], -1))
    reshaped_states = states.reshape((*experience["infos"]["state"].shape[:2], -1))
    state_pairs = np.concatenate((reshaped_states, reshaped_actions), axis=-1)

    unique_vals, counts = np.unique(state_pairs, axis=1, return_counts=True)
    count_vals, count_freq = np.unique(counts, return_counts=True)

    saco = unique_vals.shape[1] / num_tot
    return saco, count_vals, count_freq


def plot_count_frequencies(
    all_count_vals: Dict[str, Array], all_count_freq: Dict[str, Array], save_path: str = ""
) -> None:
    vault_uids = list(all_count_vals.keys())
    colors = sns.color_palette()

    for i, uid in enumerate(vault_uids):
        plt.scatter(
            np.log(all_count_vals[uid].astype(float)),
            np.log(all_count_freq[uid].astype(float)),
            label=uid,
            color=colors[i],
        )

    plt.xlabel("Count (log base 10)")
    plt.ylabel("Frequency of count (log base 10)")
    plt.legend()

    if len(save_path) > 0:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.title("Frequency of unique pair counts power law")
    plt.show()
    return


def describe_coverage(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    plot_count_freq: bool = True,
    save_plot: bool = False,
) -> None:
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    single_values = []
    all_uid_count_vals = {}
    all_uid_count_freq = {}
    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience

        saco, count_vals, count_freq = get_saco(exp)
        all_uid_count_freq[uid] = count_freq
        all_uid_count_vals[uid] = count_vals

        single_values.append([uid, saco])

    print(tabulate(single_values, headers=["Uid", "Joint SACo"]))

    if plot_count_freq:
        if save_plot:
            plot_count_frequencies(
                all_uid_count_vals,
                all_uid_count_freq,
                save_path=f"{rel_dir}/{vault_name}/count_frequency_loglog.pdf",
            )
        else:
            plot_count_frequencies(all_uid_count_vals, all_uid_count_freq)

    return


def descriptive_summary(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    plot_hist: bool = True,
    save_hist: bool = False,
    n_bins: int = 40,
    done_flags: tuple = ("terminals",),
) -> Dict[str, Array]:
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    all_returns = {}
    single_values = []
    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience

        saco, _, _ = get_saco(exp)
        mean, stddev, max_ret, min_ret, episode_returns = get_episode_return_descriptors(exp, done_flags)
        n_traj = len(episode_returns)
        n_trans = exp["actions"].shape[1]

        single_values.append([uid, mean, stddev, min_ret, max_ret, n_trans, n_traj, saco])
        all_returns[uid] = episode_returns

    print(
        tabulate(
            single_values,
            headers=[
                "Uid",
                "Mean",
                "Stddev",
                "Min return",
                "Max return",
                "Transitions",
                "Trajectories",
                "Joint SACo",
            ],
        )
    )

    if plot_hist:
        min_of_all = min([x[3] for x in single_values])
        max_of_all = max([x[4] for x in single_values])
        if save_hist:
            plot_eps_returns_hist(
                all_returns,
                vault_name,
                n_bins,
                min_of_all,
                max_of_all,
                save_path=f"{rel_dir}/{vault_name}/histogram.pdf",
            )
        else:
            plot_eps_returns_hist(all_returns, vault_name, n_bins, min_of_all, max_of_all)

    return all_returns
