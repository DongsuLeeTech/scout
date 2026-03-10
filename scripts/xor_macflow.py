"""
XOR toy example to stress IGM breakdown with MACFlow.

What it does
- Creates a stateless 2-agent XOR environment (continuous actions in [-1, 1]).
- Generates an offline dataset concentrated on the two XOR modes: (+1, -1) and (-1, +1).
- Trains MACFlow agent on the dataset.
- Evaluates (1) joint-flow sampling and (2) factored one-step policy sampling.
- Saves metrics CSV and an optional figure showing action samples and performance.

Run
  python scripts/xor_macflow.py --steps 2000 --batch_size 256 --seed 0 --save_dir exp/xor_macflow

Notes
- Uses agents/macflow.MADFlowAgent (continuous actions).
- To illustrate coupling, we compare rewards using joint-flow vs factorized sampling.
"""

import argparse
import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from agents.macflow import MADFlowAgent, get_config
from utils.loggers import CsvLogger, get_exp_name
from util import batch_concat_agent_id_to_obs


def xor_reward(a1, a2):
    """Reward = 1.0 if signs differ; else 0.0.

    Args:
        a1: scalar action of agent 1
        a2: scalar action of agent 2
    Returns:
        float reward in {0.0, 1.0}
    """
    return 1.0 if np.sign(a1) != np.sign(a2) else 0.0


def make_xor_dataset(n_samples=50000, T=2):
    """Create an offline dataset of XOR coordination.

    Shapes follow the repo convention: (B, T, N, ...)
    - observations: trivial zeros of shape (B, T, 2, 1)
    - actions: only two modes (+1, -1) or (-1, +1) at t=0 (replicated to t=1 for stability)
    - rewards: r_t is XOR of signs at that time step
    - terminals: mark last step as terminal to keep TD shapes safe
    """
    B = n_samples
    N = 2
    obs = np.zeros((B, T, N, 1), dtype=np.float32)
    acts = np.zeros((B, T, N), dtype=np.float32)
    rews = np.zeros((B, T, N), dtype=np.float32)
    terms = np.zeros((B, T, N), dtype=np.float32)

    for i in range(B):
        if np.random.rand() < 0.5:
            a0 = np.array([+1.0, -1.0], dtype=np.float32)
        else:
            a0 = np.array([-1.0, +1.0], dtype=np.float32)

        # Set actions for t=0 and replicate to t=1 to avoid empty time-slices
        acts[i, 0] = a0
        acts[i, 1] = a0

        # Rewards for both timesteps (duplicated)
        r = xor_reward(a0[0], a0[1])
        rews[i, 0] = r
        rews[i, 1] = r

        # Mark last step as terminal
        terms[i, 1] = 1.0

    batch = {
        "observations": jnp.array(obs),
        "actions": jnp.array(acts),
        "rewards": jnp.array(rews),
        "terminals": jnp.array(terms),
    }
    return batch


def create_agent(dummy_batch, seed=0):
    cfg = get_config()
    # Minimal config suitable for a stateless toy
    cfg["encoder"] = None
    cfg["alpha"] = 1.0
    cfg["flow_steps"] = 8
    cfg["discount"] = 0.0
    cfg["normalize_q_loss"] = False
    cfg["q_agg"] = "mean"
    # Shrink networks for speed
    cfg["actor_hidden_dims"] = (128, 128)
    cfg["value_hidden_dims"] = (128, 128)

    ex_obs = dummy_batch["observations"]  # (B, T, N, O)
    ex_act = dummy_batch["actions"]       # (B, T, N)
    agent = MADFlowAgent.create(
        seed=seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        agent_names=("agent_1", "agent_2"),
        config=cfg,
    )
    return agent


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 256


def train_macflow(agent, dataset, steps=2000, batch_size=256, logger: CsvLogger = None):
    data_size = dataset["actions"].shape[0]
    for step in range(steps):
        idx = np.random.randint(0, data_size, size=batch_size)
        batch = {k: v[idx] for k, v in dataset.items()}
        agent, info = agent.update(batch, step)

        if logger is not None and (step % 50 == 0 or step == steps - 1):
            log_row = {k: float(v) for k, v in info.items()}
            logger.log(log_row, step)

        if step % 200 == 0:
            print(f"[train {step:05d}] actor_loss={float(info['actor/actor_loss']):.4f} \n"
                  f"               bc_loss={float(info['actor/bc_flow_loss']):.4f} distill={float(info['actor/distill_loss']):.4f}")

    return agent


def eval_joint(agent, n_samples=2000, rng=None):
    # Build a single zero-observation with agent ids concatenated
    obs = jnp.zeros((1, 1, 2, 1), dtype=jnp.float32)  # (B=1, T=1, N=2, O=1)
    obs = batch_concat_agent_id_to_obs(obs)

    # For joint-flow, draw one noise sample per agent
    if rng is None:
        rng = agent.rng
    rng, sub = jax.random.split(rng)

    total = 0.0
    for _ in range(n_samples):
        noises = jax.random.normal(sub, (1, 1, 2, agent.config['action_dim']))
        acts = agent.compute_flow_actions(obs, noises)
        a = np.array(acts[0, 0])  # (N,)
        total += xor_reward(a[0], a[1])
        rng, sub = jax.random.split(rng)
    return total / n_samples


def eval_factored(agent, n_samples=2000, rng=None):
    obs = jnp.zeros((1, 1, 2, 1), dtype=jnp.float32)
    obs = batch_concat_agent_id_to_obs(obs)

    if rng is None:
        rng = agent.rng
    rng, sub = jax.random.split(rng)

    total = 0.0
    for _ in range(n_samples):
        acts = agent.sample_actions(obs, seed=sub)[0, 0]  # (N,)
        a = np.array(acts)
        total += xor_reward(a[0], a[1])
        rng, sub = jax.random.split(rng)
    return total / n_samples


def estimate_coupling(agent, samples=2000, rng=None):
    obs = jnp.zeros((1, 1, 2, 1), dtype=jnp.float32)
    obs = batch_concat_agent_id_to_obs(obs)

    if rng is None:
        rng = agent.rng
    rng, sub = jax.random.split(rng)

    joint = []
    factored = []
    for _ in range(samples):
        n = jax.random.normal(sub, (1, 1, 2, agent.config['action_dim']))
        a_joint = np.array(agent.compute_flow_actions(obs, n)[0, 0])
        a_fact = np.array(agent.sample_actions(obs, seed=sub)[0, 0])
        joint.append(a_joint)
        factored.append(a_fact)
        rng, sub = jax.random.split(rng)

    joint = np.asarray(joint)
    factored = np.asarray(factored)

    w2 = float(np.sqrt(np.mean(np.sum((joint - factored) ** 2, axis=1))))
    corr = float(np.corrcoef(joint[:, 0], joint[:, 1])[0, 1])
    return w2, corr, joint, factored


def maybe_save_figure(save_dir, joint_samples, factored_samples):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping figure.")
        return

    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.scatter(joint_samples[:, 0], joint_samples[:, 1], s=6, alpha=0.6, label='joint')
    ax1.set_title('Joint-flow samples')
    ax1.set_xlabel('a1')
    ax1.set_ylabel('a2')
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.grid(True, alpha=0.2)

    ax2.scatter(factored_samples[:, 0], factored_samples[:, 1], s=6, alpha=0.6, color='orange', label='factored')
    ax2.set_title('Factorized samples')
    ax2.set_xlabel('a1')
    ax2.set_ylabel('a2')
    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2])
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(save_dir, 'samples_scatter.pdf')
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved figure to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_dataset', type=int, default=50000)
    parser.add_argument('--save_dir', type=str, default='exp/xor_macflow')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    exp_name = get_exp_name(args.seed)
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    csv_path = os.path.join(exp_dir, 'metrics.csv')
    logger = CsvLogger(csv_path)

    print("Building XOR dataset...")
    dataset = make_xor_dataset(n_samples=args.n_dataset, T=2)

    print("Creating agent...")
    agent = create_agent(dataset, seed=args.seed)

    print("Training MACFlow...")
    t0 = time.time()
    agent = train_macflow(agent, dataset, steps=args.steps, batch_size=args.batch_size, logger=logger)
    print(f"Training complete in {time.time() - t0:.1f}s")

    print("Evaluating joint sampling vs factored policy...")
    r_joint = float(eval_joint(agent, n_samples=2000))
    r_fact = float(eval_factored(agent, n_samples=2000))
    w2, corr, joint_samps, fact_samps = estimate_coupling(agent, samples=3000)

    # Save metrics
    summary = {
        'metric/joint_reward': r_joint,
        'metric/factored_reward': r_fact,
        'metric/w2_proxy': w2,
        'metric/corr_proxy': corr,
    }
    logger.log(summary, step=args.steps)
    logger.close()

    # Save samples for inspection
    np.save(os.path.join(exp_dir, 'joint_samples.npy'), joint_samps)
    np.save(os.path.join(exp_dir, 'factored_samples.npy'), fact_samps)

    # Optional figure
    maybe_save_figure(exp_dir, joint_samps, fact_samps)

    print("\n====================== RESULTS ======================")
    print(f"Joint-flow reward          ≈ {r_joint:.4f}")
    print(f"Factorized policy reward   ≈ {r_fact:.4f}")
    print(f"W2 distance (proxy)        ≈ {w2:.4f}")
    print(f"Correlation (MI proxy)     ≈ {corr:.4f}")
    print(f"Saved logs to              : {csv_path}")
    print(f"Saved samples              : joint_samples.npy, factored_samples.npy in {exp_dir}")
    print("=====================================================")


if __name__ == '__main__':
    main()
