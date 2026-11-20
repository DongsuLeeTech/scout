<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>SCOUT</h1>
    </summary>
  </ul>
</div>

</div>

## Overview

SCOUT is an offline multi-agent RL algorithm
that combines an expressive *flow-matching* behavioral policy
with test-time *value gradient flow* (VGF) refinement.
Instead of distilling the flow policy into a Q-guided one-step actor,
SCOUT samples a set of action particles from the flow policy,
moves them toward high-value regions with a few Stein variational gradient steps on the learned critic,
and executes the best particle.
The same agent handles both discrete (SMAC) and continuous (MAMuJoCo, MPE) action spaces.

## Installation

SCOUT requires Python 3.9+ and is based on JAX. The main dependencies are
`jax`, `flax`, `optax`, `distrax`, `flashbax`, and `ml-collections`.
To install the core dependencies, simply run:
```bash
pip install -r requirements/datasets.txt
```

Each environment suite is installed separately:
```bash
# SMAC (v1)
pip install -r install_environments/requirements/smacv1.txt
bash install_environments/smacv1.sh
# SMACv2
pip install -r install_environments/requirements/smacv2.txt
bash install_environments/smacv2.sh
# MAMuJoCo (MuJoCo 2.1.0)
bash install_environments/mujoco210.sh
pip install -r install_environments/requirements/mamujoco210.txt
# MPE
pip install -r install_environments/requirements/pettingzoo.txt
```

> [!NOTE]
> The SMAC scripts only add the SMAC maps to `$SC2PATH` (`/data/StarCraftII` by default).
> You need to install StarCraft II (4.10) there beforehand.

## Usage

The main implementation of SCOUT is in [agents/scout.py](agents/scout.py),
and a faster variant with a DQN-style critic for discrete actions is in [agents/scout_efficiency.py](agents/scout_efficiency.py).
Our implementations of two baselines (MACFlow and BC) can also be found in the same directory.
All agents are trained with a single entry point, [main.py](main.py),
and the action type (discrete or continuous) is detected automatically from the dataset.
We use `scout_eff` for discrete-action tasks (SMAC and SMACv2) and `scout` for continuous-action tasks (MAMuJoCo and MPE).
Here are some example commands:
```bash
# SCOUT on SMAC 5m_vs_6m (OG-MARL, Good)
python main.py --agent_name=scout_eff --env=smac_v1 --source=og_marl --scenario=5m_vs_6m --dataset=Good --agent.train_vgf_steps=1
# SCOUT on SMACv2 terran_5_vs_5 (OG-MARL, Replay)
python main.py --agent_name=scout_eff --env=smac_v2 --source=og_marl --scenario=terran_5_vs_5 --dataset=Replay --agent.train_vgf_steps=1
# SCOUT on MAMuJoCo 6halfcheetah (OMIGA, Medium-Replay)
python main.py --agent_name=scout --env=mamujoco --source=omiga --scenario=6halfcheetah --dataset=Medium-Replay --agent.train_vgf_steps=1
# SCOUT on MPE simple_spread (OMAR, Medium)
python main.py --agent_name=scout --env=mpe --source=omar --scenario=simple_spread --dataset=Medium --agent.train_vgf_steps=1
# SCOUT on SMAC 3s5z_vs_3s6z (offline-to-online RL)
python main.py --agent_name=scout_eff --env=smac_v1 --source=og_marl --scenario=3s5z_vs_3s6z --dataset=Good --online_steps=500000 --agent.train_vgf_steps=1
```

A task is specified by four flags:

| Flag | Description | Examples |
| --- | --- | --- |
| `--env` | Environment suite | `smac_v1`, `smac_v2`, `mamujoco`, `mpe` |
| `--source` | Dataset provider | `og_marl`, `omiga`, `omar` |
| `--scenario` | Map or task | `3m`, `5m_vs_6m`, `terran_5_vs_5`, `2ant`, `6halfcheetah`, `simple_spread` |
| `--dataset` | Dataset quality | `Good`, `Medium`, `Poor`, `Expert`, `Medium-Replay` |

The following agents are available through `--agent_name`:

| Agent | Description |
| --- | --- |
| `scout` | SCOUT ([agents/scout.py](agents/scout.py)). |
| `scout_eff` | Efficient SCOUT ([agents/scout_efficiency.py](agents/scout_efficiency.py)). |
| `macflow` | MACFlow ([agents/macflow.py](agents/macflow.py)): flow BC, one-step distillation, and Q-guidance. |
| `bc` | Behavioral cloning ([agents/bc.py](agents/bc.py)). |

Agent hyperparameters are defined in `get_config()` of each agent file and can be overridden with `--agent.*`:
```bash
python main.py --agent_name=scout_eff --env=smac_v1 --source=og_marl --scenario=3m --dataset=Good --agent.train_vgf_steps=2 --agent.lvc_coef=0.5
```

### Datasets

Datasets are [OG-MARL](https://huggingface.co/datasets/InstaDeepAI/og-marl) vaults.
With the default `--data_dir=./data/`, they are automatically downloaded from Hugging Face on the first run
and stored in `./data/<source>/<env>/<scenario>.vlt`.
If you set `--data_dir` to another directory, the vault is expected to already exist there.
The complete list of available `source`, `env`, and `scenario` combinations
is in [vault_utils/download_vault.py](vault_utils/download_vault.py).

### Logging and evaluation

Training and evaluation metrics are logged to Weights & Biases (`--project_name`, `--run_group`)
and to `train.csv` and `eval.csv` under `--save_dir`.
For SCOUT agents, every evaluation reports returns (and win rates on SMAC)
with `0`, `1`, `3`, `5`, and `10` VGF refinement steps,
logged as `evaluation/mean_episode_return_steps_{k}` and `evaluation/win_rate_steps_{k}`.
Checkpoints are saved every `--save_interval` steps
and can be restored with `--restore_path` and `--restore_epoch`.

## Tips for hyperparameter tuning

Here are some general tips for SCOUT's hyperparameter tuning for new tasks:

* The most important hyperparameter of SCOUT is the number of VGF steps used for critic training (`--agent.train_vgf_steps`).
  We tune it in `{1, 2, 3}` for `scout_eff` and `{0, 1, 2, 3}` for `scout`, and keep the other hyperparameters at their default values.
* The number of test-time VGF steps does not need separate runs,
  since every evaluation reports results with `0`, `1`, `3`, `5`, and `10` steps.
* `--agent.lvc_coef` controls the strength of local value calibration of the critic
  (default: `0.5` for `scout_eff` and `0` for `scout`).
* `--agent.vgf_lr` and `--agent.vgf_alpha` control the VGF step size and the strength of the repulsion between particles,
  and `--agent.vgf_particles` controls the number of action particles (default: `10`).
* For other hyperparameters, you may use the default values in `agents/scout.py` and `agents/scout_efficiency.py`.
