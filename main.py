"""Unified entry point for offline (and optional offline-to-online) MARL training.

Supports both discrete (SMAC) and continuous (MAMuJoCo) environments with
automatic action-type detection.

Agents: macflow, scout, scout_team, bc
"""
import os
import sys
import json
import time
import random
import concurrent.futures

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags, config_dict

from envs.environments import get_environment
from utils.replay_buffers import FlashbaxReplayBuffer
from vault_utils.download_vault import download_and_unzip_vault
from utils.loggers import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    setup_wandb,
    get_wandb_video,
)

# ========================== Agent Registry ========================== #
AGENT_MAP = {
    'macflow':    ('agents.macflow',     'MACFlowAgent'),
    'scout':      ('agents.scout',       'ScoutAgent'),
    'scout_v1':   ('agents.scout',       'ScoutAgent'),
    'scout_v2':   ('agents.scout',       'ScoutAgent'),
    'scout_eff':      ('agents.scout_efficiency', 'ScoutEfficientAgent'),
    'scout_eff_v1':   ('agents.scout_efficiency', 'ScoutEfficientAgent'),
    'scout_eff_v2':   ('agents.scout_efficiency', 'ScoutEfficientAgent'),
    'scout_team': ('agents.scout_team',  'ScoutTeamAgent'),
    'bc':         ('agents.bc',          'MABCAgent'),
    'odis':       ('agents.odis',        'ODISAgent'),
}

AGENT_CONFIG_MAP = {
    'macflow':    'agents/macflow.py',
    'scout':      'agents/scout.py',
    'scout_v1':   'agents/scout.py:v1',
    'scout_v2':   'agents/scout.py:v2',
    'scout_eff':      'agents/scout_efficiency.py',
    'scout_eff_v1':   'agents/scout_efficiency.py:v1',
    'scout_eff_v2':   'agents/scout_efficiency.py:v2',
    'scout_team': 'agents/scout_team.py',
    'bc':         'agents/bc.py',
    'odis':       'agents/odis.py',
}

# ========================== Flags ========================== #
# Pre-parse agent_name from sys.argv so we can define the correct config flag
# BEFORE app.run() parses all flags (required for --agent.XXX overrides to work).
_agent_name = 'macflow'
for _i, _arg in enumerate(sys.argv):
    if _arg.startswith('--agent_name='):
        _agent_name = _arg.split('=', 1)[1]
    elif _arg == '--agent_name' and _i + 1 < len(sys.argv):
        _agent_name = sys.argv[_i + 1]

FLAGS = flags.FLAGS
flags.DEFINE_string('agent_name',      'macflow',   f'Agent name: {list(AGENT_MAP.keys())}')
flags.DEFINE_string('run_group',       'Debug',     'Run group')
flags.DEFINE_integer('seed',           0,           'Random seed')
flags.DEFINE_string('env',             'smac_v1',   'Environment family')
flags.DEFINE_string('source',          'og_marl',   'Dataset source')
flags.DEFINE_string('scenario',        '3m',        'Scenario / map name')
flags.DEFINE_string('dataset',         'Good',      'Dataset quality')
flags.DEFINE_string('save_dir',        'exp/',      'Save directory')
flags.DEFINE_string('project_name',    'test',      'WandB project name')
flags.DEFINE_string('data_dir',        './data/',   'Dataset base directory')

flags.DEFINE_integer('offline_steps',    500_000,   'Offline gradient steps')
flags.DEFINE_integer('online_steps',     0,         'Online fine-tuning steps (0 = offline only)')
flags.DEFINE_integer('sequence_length',  20,        'Replay sequence length')
flags.DEFINE_integer('sample_period',    1,         'Sample period')
flags.DEFINE_integer('buffer_size',      2_000_000, 'Max transitions in buffer')
flags.DEFINE_integer('batch_size',       32,        'Batch size')

flags.DEFINE_integer('log_interval',   50_000,     'Log every N steps')
flags.DEFINE_integer('eval_interval',  100_000,    'Evaluate every N steps')
flags.DEFINE_integer('save_interval',  1_000_001,  'Save checkpoint every N steps')
flags.DEFINE_integer('num_eval_episodes', 10,      'Number of evaluation episodes')
flags.DEFINE_integer('num_eval_workers', 10,       'Parallel eval workers (threads)')

# Define agent config flag BEFORE app.run() so --agent.XXX overrides are recognized
_config_path = AGENT_CONFIG_MAP.get(_agent_name, 'agents/macflow.py')
config_flags.DEFINE_config_file('agent', _config_path, lock_config=False)


# ========================== Main ========================== #
def main(_):
    agent_name = FLAGS.agent_name
    if agent_name not in AGENT_MAP:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose from {list(AGENT_MAP.keys())}")

    # Import agent class
    module_path, class_name = AGENT_MAP[agent_name]
    import importlib
    agent_module = importlib.import_module(module_path)
    agent_cls = getattr(agent_module, class_name)

    cfg = FLAGS.agent
    if hasattr(cfg, 'copy_and_resolve_references'):
        cfg = cfg.copy_and_resolve_references()
    else:
        cfg = config_dict.ConfigDict(cfg.to_dict())

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project=FLAGS.project_name, group=FLAGS.run_group, name=exp_name)

    save_root = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(save_root, exist_ok=True)
    json.dump(get_flag_dict(), open(os.path.join(save_root, 'flags.json'), 'w'))

    # -------- Environment & Buffer --------
    env_factory = lambda s: get_environment(FLAGS.source, FLAGS.env, FLAGS.scenario, s)
    env = env_factory(FLAGS.seed)
    agent_names = list(env.agents)

    buffer = FlashbaxReplayBuffer(
        sequence_length=FLAGS.sequence_length,
        batch_size=FLAGS.batch_size,
        sample_period=FLAGS.sample_period,
        seed=FLAGS.seed,
        max_size=FLAGS.buffer_size,
    )

    if FLAGS.data_dir in ('/datastor1/dongsu', '/datastor1/dongsu/'):
        download_and_unzip_vault(FLAGS.source, FLAGS.env, FLAGS.scenario,
                                 dataset_base_dir=FLAGS.data_dir)
        buffer.populate_from_vault(FLAGS.source, FLAGS.env, FLAGS.scenario,
                                   str(FLAGS.dataset), rel_dir=FLAGS.data_dir)
    else:
        download_and_unzip_vault(FLAGS.source, FLAGS.env, FLAGS.scenario)
        buffer.populate_from_vault(FLAGS.source, FLAGS.env, FLAGS.scenario,
                                   str(FLAGS.dataset))

    example = buffer.sample()
    ex_obs = jnp.asarray(example['observations'])
    ex_act = jnp.asarray(example['actions'])

    # -------- Auto-detect discrete vs continuous --------
    # Discrete actions have shape (B, T, N) with integer type -> legals gives action_dim
    # Continuous actions have shape (B, T, N, A) with float type
    is_discrete = (len(ex_act.shape) == 3)

    if is_discrete:
        # For discrete envs, use legals shape for action_dim
        ex_act = jnp.asarray(example['infos']['legals'])

    # Set is_discrete in config
    cfg['is_discrete'] = is_discrete
    if is_discrete:
        cfg['agent_name'] = agent_name

    # Pass total_steps for LR schedule
    cfg['total_steps'] = FLAGS.offline_steps + FLAGS.online_steps

    # Pass scenario to agent config (needed by ODIS decomposer)
    if agent_name == 'odis':
        cfg['scenario'] = FLAGS.scenario

    # -------- Create Agent --------
    agent = agent_cls.create(
        seed=FLAGS.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        agent_names=agent_names,
        config=cfg,
    )

    # -------- Determine VGF eval steps --------
    # Any agent starting with 'scout' gets multi-step VGF evaluation
    has_vgf = agent_name.startswith('scout')
    if has_vgf:
        eval_vgf_steps_list = [0, 1, 3, 5, 10]
    else:
        eval_vgf_steps_list = [None]

    # -------- Pre-create eval environment pool --------
    n_eval_eps = FLAGS.num_eval_episodes
    eval_env_pool = []
    for i in range(n_eval_eps):
        try:
            eval_env_pool.append(env_factory(FLAGS.seed + 1000 + i))
        except Exception:
            eval_env_pool = []
            break

    # -------- Logger --------
    train_csv = CsvLogger(os.path.join(save_root, 'train.csv'))
    eval_csv = CsvLogger(os.path.join(save_root, 'eval.csv'))
    t0 = time.time()
    last = t0

    # -------- Offline Training --------
    for step in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), dynamic_ncols=True, desc='Offline'):
        batch = buffer.sample()
        agent, info = agent.update(batch, step)

        if step % FLAGS.log_interval == 0:
            metrics = {f'train/{k}': float(v) for k, v in info.items()}
            metrics['time/iter_s'] = (time.time() - last) / FLAGS.log_interval
            wandb.log(metrics, step=step)
            train_csv.log(metrics, step=step)
            last = time.time()

        if step % FLAGS.eval_interval == 0:
            eval_metrics = {}
            for eval_idx, eval_steps in enumerate(eval_vgf_steps_list):
                eval_ret = _evaluate(
                    agent, env_factory, is_discrete,
                    n_eps=n_eval_eps,
                    seed=FLAGS.seed + eval_idx,
                    eval_vgf_steps=eval_steps,
                    num_workers=FLAGS.num_eval_workers,
                    env_pool=eval_env_pool if eval_env_pool else None,
                )
                eval_metrics.update(eval_ret)
            wandb.log(eval_metrics, step=step)
            eval_csv.log(eval_metrics, step=step)

        if step % FLAGS.save_interval == 0:
            agent.network.save(os.path.join(save_root, f'ckpt_{step}.npz'))

    # -------- Online Fine-tuning (optional) --------
    if FLAGS.online_steps > 0:
        online_env = env_factory(FLAGS.seed + 9999)
        for step in tqdm.tqdm(
            range(FLAGS.offline_steps + 1, FLAGS.offline_steps + FLAGS.online_steps + 1),
            dynamic_ncols=True, desc='Online',
        ):
            # Collect data from environment and add to buffer
            # (placeholder - actual online data collection varies by env)
            batch = buffer.sample()
            agent, info = agent.update(batch, step)

            if step % FLAGS.log_interval == 0:
                metrics = {f'train/{k}': float(v) for k, v in info.items()}
                metrics['time/iter_s'] = (time.time() - last) / FLAGS.log_interval
                wandb.log(metrics, step=step)
                train_csv.log(metrics, step=step)
                last = time.time()

            if step % FLAGS.eval_interval == 0:
                eval_metrics = {}
                for eval_idx, eval_steps in enumerate(eval_vgf_steps_list):
                    eval_ret = _evaluate(
                        agent, env_factory, is_discrete,
                        n_eps=n_eval_eps,
                        seed=FLAGS.seed + eval_idx,
                        eval_vgf_steps=eval_steps,
                        num_workers=FLAGS.num_eval_workers,
                        env_pool=eval_env_pool if eval_env_pool else None,
                    )
                    eval_metrics.update(eval_ret)
                wandb.log(eval_metrics, step=step)
                eval_csv.log(eval_metrics, step=step)

    # -------- Cleanup --------
    for e in eval_env_pool:
        if hasattr(e, 'close'):
            try:
                e.close()
            except Exception:
                pass
    train_csv.close()
    eval_csv.close()


# ========================== Evaluation ========================== #
def _evaluate(agent, env_factory, is_discrete, n_eps=10, seed=0,
              eval_vgf_steps=None, num_workers=1, env_pool=None):
    """Evaluate agent across episodes, handling both discrete and continuous.

    Args:
        env_pool: Optional list of pre-created environments to reuse (avoids
            expensive SC2 process startup). When provided, environments are
            reset and reused instead of creating new ones.
    """
    eval_results = [None] * n_eps  # (episode_return, battle_won)

    def _run_episode(ep_idx):
        # Use pool env if available, otherwise create new
        if env_pool and ep_idx < len(env_pool):
            env = env_pool[ep_idx]
            close_env = False
        else:
            env = env_factory(seed + ep_idx)
            close_env = True

        observations, infos = env.reset()
        done = False
        episode_return = 0.0
        battle_won = False
        carry = None
        reset_mask = None

        while not done:
            if is_discrete:
                # Discrete evaluation: use sample_actions_with_carry
                if hasattr(agent, 'sample_actions_with_carry'):
                    actions, carry = agent.sample_actions_with_carry(
                        observations, carry,
                        jax.random.PRNGKey(seed + ep_idx),
                        infos.get('legals'),
                        reset_mask=reset_mask,
                        eval_vgf_steps=eval_vgf_steps,
                    )
                else:
                    actions = agent.sample_actions(
                        observations, jax.random.PRNGKey(seed + ep_idx),
                        legal_actions=infos.get('legals'),
                        eval_vgf_steps=eval_vgf_steps,
                    )

                # Convert to int for discrete envs
                actions = {
                    name: int(actions[name].item())
                    for name in agent.agent_names
                }
            else:
                # Continuous evaluation
                kwargs = {}
                if eval_vgf_steps is not None:
                    kwargs['eval_vgf_steps'] = eval_vgf_steps
                actions = agent.sample_actions(
                    observations,
                    jax.random.PRNGKey(seed + ep_idx),
                    **kwargs,
                )

            observations, rewards, terminal, truncation, infos = env.step(actions)
            episode_return += np.mean(list(rewards.values()), dtype="float")

            # Track battle_won from SMAC environments
            if infos.get('battle_won', False):
                battle_won = True

            if is_discrete:
                reset_mask = jnp.asarray(
                    [terminal[ag] for ag in agent.agent_names], dtype=jnp.bool_,
                )

            done = all(terminal.values()) or all(truncation.values())

        if close_env and hasattr(env, 'close'):
            try:
                env.close()
            except Exception:
                pass
        return (episode_return, battle_won)

    if env_pool:
        # Sequential when using pool (environments are not thread-safe)
        for i in range(n_eps):
            eval_results[i] = _run_episode(i)
    elif num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_run_episode, i) for i in range(n_eps)]
            for i, fut in enumerate(futures):
                eval_results[i] = fut.result()
    else:
        for i in range(n_eps):
            eval_results[i] = _run_episode(i)

    suffix = "" if eval_vgf_steps is None else f"_steps_{eval_vgf_steps}"
    valid = [r for r in eval_results if r is not None]
    if not valid:
        valid = [(0.0, False)]
    returns = [r[0] for r in valid]
    wins = [r[1] for r in valid]
    return {
        f"evaluation/mean_episode_return{suffix}": np.mean(returns),
        f"evaluation/max_episode_return{suffix}": np.max(returns),
        f"evaluation/min_episode_return{suffix}": np.min(returns),
        f"evaluation/win_rate{suffix}": np.mean(wins),
        f"evaluation/wins{suffix}": int(np.sum(wins)),
    }


if __name__ == '__main__':
    os.environ["SUPPRESS_GR_PROMPT"] = "1"
    app.run(main)
