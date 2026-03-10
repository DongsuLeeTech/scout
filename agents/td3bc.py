import copy
from typing import Any, Dict, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value
from util import *

class MATD3BCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    agent_names: Sequence[str] = nonpytree_field()
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        qs = self.network.seimport copy
from typing import Any, Dict, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value
from util import *

class MATD3BCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    agent_names: Sequence[str] = nonpytree_field()
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        qs = self.network.select("critic")(batch['observations'][:-1], actions=batch['actions'][:-1], params=grad_params)

        pi_dist = self.network.select("actor")(batch['observations'][1:])
        pi_actions = pi_dist.mode()

        target_qs = self.network.select("target_critic")(batch['observations'][1:], actions=pi_actions)
        target_qs = target_qs.min(axis=0)
        targets = batch['rewards'][:-1] + (1.0 - batch['terminals'][1:]) * self.config['discount'] * target_qs

        critic_loss_1 = jnp.mean(0.5 * (qs[0] - targets) ** 2)
        critic_loss_2 = jnp.mean(0.5 * (qs[1] - targets) ** 2)
        critic_loss = (critic_loss_1 + critic_loss_2) / 2

        metrics = {
            'critic_loss_1': critic_loss_1,
            'critic_loss_2': critic_loss_2,
            'critic_loss': critic_loss
        }

        return critic_loss, metrics

    def actor_loss(self,batch, grad_params):
        pi_dist = self.network.select("actor")(batch['observations'], params=grad_params)
        pi_actions = pi_dist.mode()

        pi_qs = self.network.select("critic")(batch['observations'], actions=pi_actions)
        pi_q = pi_qs.min(axis=0)

        bc_loss = jnp.mean((pi_actions - batch['actions']) ** 2)
        q_loss = - self.config['bc_alpha'] / jnp.mean(jnp.abs(pi_q)) * jnp.mean(pi_q)
        actor_loss = bc_loss + q_loss

        metrics = {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'q_loss': q_loss
        }

        return actor_loss, metrics

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = self.rng if rng is None else rng

        observations = batch["observations"]  # (B,T,N,O)
        actions = batch["actions"]  # (B,T,N,A)
        rewards = batch["rewards"]  # (B,T,N)
        terminals = jnp.array(batch["terminals"], "float32")  # (B,T,N)

        # Make time-major
        observations = batch_concat_agent_id_to_obs(observations)
        observations = switch_two_leading_dims(observations)

        replay_actions = switch_two_leading_dims(actions)
        rewards = switch_two_leading_dims(rewards)
        terminals = switch_two_leading_dims(terminals)

        batch["observations"] = observations
        batch["actions"] = replay_actions
        batch["rewards"] = rewards
        batch["terminals"] = terminals

        c_loss, c_info = self.critic_loss(batch, grad_params)
        p_loss, p_info = self.actor_loss(batch, grad_params)

        for k, v in c_info.items():
            info[f'critic/{k}'] = v
        for k, v in p_info.items():
            info[f'policy/{k}'] = v

        loss = c_loss + p_loss
        info['loss'] = loss
        return loss, info

    # --------------------------------------------------------------------- #
    #  Update
    # --------------------------------------------------------------------- #
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, step):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    # --------------------------------------------------------------------- #
    #  Acting
    # --------------------------------------------------------------------- #
    @jax.jit
    def sample_actions(self, observations: Dict[str, jnp.ndarray], seed, temperature=0.0):
        rngs = jax.random.split(seed, len(self.agent_names))
        acts = {}
        for r, i, agent in zip(rngs, range(self.config['num_agents']), self.agent_names):
            agent_observation = concat_agent_id_to_obs(observations[agent], i, self.config['num_agents'])
            dist = self.network.select('actor')(agent_observation)
            acts[agent] = dist.mode()
        return acts

    # --------------------------------------------------------------------- #
    #  Factory
    # --------------------------------------------------------------------- #
    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        agent_names,
        config,
    ):
        """Instantiate new MABCAgent."""
        master_rng = jax.random.PRNGKey(seed)
        master_rng, big_init_rng = jax.random.split(master_rng, 2)

        # ------------------------------------------------- encoders (optional)
        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['policy'] = encoder_module()

        # ------------------------------------------------- per-agent policies
        networks: Dict[str, Any] = {}
        network_args: Dict[str, Any] = {}

        cfg = dict(config)
        policy_def = Actor(
            hidden_dims=config['hidden_dims'],
            action_dim=ex_actions.shape[-1],
            layer_norm=config['layer_norm'],
            encoder=encoders.get('policy')
        )
        critic_def = Value(
            hidden_dims=config['hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('q'),
        )
        ex_obs_with_id = batch_concat_agent_id_to_obs(ex_observations)
        network_info = dict(
            critic=(critic_def, (ex_obs_with_id, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_obs_with_id, ex_actions)),
            actor=(policy_def, (ex_obs_with_id, )),
        )

        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_obs_with_id,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(big_init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ex_obs_with_id.shape[-1]
        config['action_dim'] = ex_actions.shape[-1]
        config['num_agents'] = len(agent_names)

        return cls(
            rng=master_rng,
            network=network,
            agent_names=tuple(agent_names),
            config=flax.core.FrozenDict(**config),
        )


# --------------------------------------------------------------------------- #
#  Default config helper
# --------------------------------------------------------------------------- #
def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='td3bc',
            lr=3e-4,
            hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            encoder=None,
            discount=0.99,
            bc_alpha=2.5,
            tau=0.005,
        )
    )lect("critic")(batch['observations'][:-1], actions=batch['actions'][:-1], params=grad_params)

        pi_dist = self.network.select("actor")(batch['observations'][1:])
        pi_actions = pi_dist.mode()

        target_qs = self.network.select("target_critic")(batch['observations'][1:], actions=pi_actions)
        target_qs = target_qs.min(axis=0)
        targets = batch['rewards'][:-1] + (1.0 - batch['terminals'][1:]) * self.config['discount'] * target_qs

        critic_loss_1 = jnp.mean(0.5 * (qs[0] - targets) ** 2)
        critic_loss_2 = jnp.mean(0.5 * (qs[1] - targets) ** 2)
        critic_loss = (critic_loss_1 + critic_loss_2) / 2

        metrics = {
            'critic_loss_1': critic_loss_1,
            'critic_loss_2': critic_loss_2,
            'critic_loss': critic_loss
        }

        return critic_loss, metrics

    def actor_loss(self,batch, grad_params):
        pi_dist = self.network.select("actor")(batch['observations'], params=grad_params)
        pi_actions = pi_dist.mode()

        pi_qs = self.network.select("critic")(batch['observations'], actions=pi_actions)
        pi_q = pi_qs.min(axis=0)

        bc_loss = jnp.mean((pi_actions - batch['actions']) ** 2)
        q_loss = - self.config['bc_alpha'] / jnp.mean(jnp.abs(pi_q)) * jnp.mean(pi_q)
        actor_loss = bc_loss + q_loss

        metrics = {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'q_loss': q_loss
        }

        return actor_loss, metrics

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = self.rng if rng is None else rng

        observations = batch["observations"]  # (B,T,N,O)
        actions = batch["actions"]  # (B,T,N,A)
        rewards = batch["rewards"]  # (B,T,N)
        terminals = jnp.array(batch["terminals"], "float32")  # (B,T,N)

        # Make time-major
        observations = batch_concat_agent_id_to_obs(observations)
        observations = switch_two_leading_dims(observations)

        replay_actions = switch_two_leading_dims(actions)
        rewards = switch_two_leading_dims(rewards)
        terminals = switch_two_leading_dims(terminals)

        batch["observations"] = observations
        batch["actions"] = replay_actions
        batch["rewards"] = rewards
        batch["terminals"] = terminals

        c_loss, c_info = self.critic_loss(batch, grad_params)
        p_loss, p_info = self.actor_loss(batch, grad_params)

        for k, v in c_info.items():
            info[f'critic/{k}'] = v
        for k, v in p_info.items():
            info[f'policy/{k}'] = v

        loss = c_loss + p_loss
        info['loss'] = loss
        return loss, info

    # --------------------------------------------------------------------- #
    #  Update
    # --------------------------------------------------------------------- #
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, step):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    # --------------------------------------------------------------------- #
    #  Acting
    # --------------------------------------------------------------------- #
    @jax.jit
    def sample_actions(self, observations: Dict[str, jnp.ndarray], seed, temperature=0.0):
        rngs = jax.random.split(seed, len(self.agent_names))
        acts = {}
        for r, i, agent in zip(rngs, range(self.config['num_agents']), self.agent_names):
            agent_observation = concat_agent_id_to_obs(observations[agent], i, self.config['num_agents'])
            dist = self.network.select('actor')(agent_observation)
            acts[agent] = dist.mode()
        return acts

    # --------------------------------------------------------------------- #
    #  Factory
    # --------------------------------------------------------------------- #
    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        agent_names,
        config,
    ):
        """Instantiate new MABCAgent."""
        master_rng = jax.random.PRNGKey(seed)
        master_rng, big_init_rng = jax.random.split(master_rng, 2)

        # ------------------------------------------------- encoders (optional)
        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['policy'] = encoder_module()

        # ------------------------------------------------- per-agent policies
        networks: Dict[str, Any] = {}
        network_args: Dict[str, Any] = {}

        cfg = dict(config)
        policy_def = Actor(
            hidden_dims=config['hidden_dims'],
            action_dim=ex_actions.shape[-1],
            layer_norm=config['layer_norm'],
            encoder=encoders.get('policy')
        )
        critic_def = Value(
            hidden_dims=config['hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('q'),
        )
        ex_obs_with_id = batch_concat_agent_id_to_obs(ex_observations)
        network_info = dict(
            critic=(critic_def, (ex_obs_with_id, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_obs_with_id, ex_actions)),
            actor=(policy_def, (ex_obs_with_id, )),
        )

        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_obs_with_id,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(big_init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ex_obs_with_id.shape[-1]
        config['action_dim'] = ex_actions.shape[-1]
        config['num_agents'] = len(agent_names)

        return cls(
            rng=master_rng,
            network=network,
            agent_names=tuple(agent_names),
            config=flax.core.FrozenDict(**config),
        )


# --------------------------------------------------------------------------- #
#  Default config helper
# --------------------------------------------------------------------------- #
def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='td3bc',
            lr=3e-4,
            hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            encoder=None,
            discount=0.99,
            bc_alpha=2.5,
            tau=0.005,
        )
    )