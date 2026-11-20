import copy
from typing import Any, Dict, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, DiscreteActor
from util import *

class MABCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    agent_names: Sequence[str] = nonpytree_field()
    config: Any = nonpytree_field()

    def bc_loss(self, batch, grad_params):
        metrics = {}
        is_discrete = self.config.get('is_discrete', False)

        if is_discrete:
            dist = self.network.select("actor")(batch['observations'], params=grad_params)
            logits = dist.logits

            legals = batch['legals']
            logits = jnp.where(legals.astype(bool), logits, -1e9)

            actions_int = batch['actions'].astype(jnp.int32)
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, actions_int)
            )
            metrics['bc_loss'] = loss
        else:
            dist = self.network.select("actor")(batch['observations'], params=grad_params)
            actions = dist.mode()
            loss = jnp.mean((actions - batch['actions']) ** 2)
            metrics['bc_loss'] = loss

        return loss, metrics

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        is_discrete = self.config.get('is_discrete', False)

        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminals = jnp.array(batch["terminals"], "float32")

        observations = batch_concat_agent_id_to_obs(observations)
        observations = switch_two_leading_dims(observations)

        replay_actions = switch_two_leading_dims(actions)
        rewards = switch_two_leading_dims(rewards)
        terminals = switch_two_leading_dims(terminals)

        inner_batch = {
            "observations": observations,
            "actions": replay_actions,
            "rewards": rewards,
            "terminals": terminals,
        }

        if is_discrete:
            legals = batch['infos']['legals']
            inner_batch['legals'] = switch_two_leading_dims(legals)

        loss, metrics = self.bc_loss(inner_batch, grad_params)
        metrics['loss'] = loss
        return loss, metrics

    @jax.jit
    def update(self, batch, step=None):
        next_rng, step_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=step_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(rng=next_rng, network=new_network), info

    @jax.jit
    def sample_actions(self, observations: Dict[str, jnp.ndarray], seed,
                       legal_actions=None, **kwargs):
        is_discrete = self.config.get('is_discrete', False)
        rngs = jax.random.split(seed, len(self.agent_names))
        acts = {}
        for r, i, agent in zip(rngs, range(self.config['num_agents']), self.agent_names):
            agent_observation = concat_agent_id_to_obs(observations[agent], i, self.config['num_agents'])
            dist = self.network.select('actor')(agent_observation)

            if is_discrete:
                logits = dist.logits
                if legal_actions is not None and agent in legal_actions:
                    mask = legal_actions[agent].astype(bool)
                    logits = jnp.where(mask, logits, -1e9)
                acts[agent] = jnp.argmax(logits, axis=-1)
            else:
                acts[agent] = dist.mode()
        return acts

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        agent_names,
        config,
    ):
        master_rng = jax.random.PRNGKey(seed)
        master_rng, big_init_rng = jax.random.split(master_rng)

        is_discrete = config.get('is_discrete', False)

        encoder_def = None
        if config['encoder'] is not None:
            encoder_def = encoder_modules[config['encoder']]()

        networks: Dict[str, Any] = {}
        network_args: Dict[str, Any] = {}

        cfg = dict(config)
        action_dim = ex_actions.shape[-1]

        if is_discrete:
            policy_def = DiscreteActor(
                hidden_dims=config['policy_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                encoder=copy.deepcopy(encoder_def) if encoder_def else None,
            )
        else:
            policy_def = Actor(
                hidden_dims=config['policy_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                encoder=copy.deepcopy(encoder_def) if encoder_def else None,
            )

        ex_obs_with_id = batch_concat_agent_id_to_obs(ex_observations)
        networks["actor"] = policy_def
        network_args["actor"] = (ex_obs_with_id,)

        net_def = ModuleDict(networks)
        net_params = net_def.init(big_init_rng, **network_args)['params']
        train_state = TrainState.create(
            net_def,
            net_params,
            tx=optax.adam(config['lr']),
        )

        cfg.update({
            "ob_dims": ex_obs_with_id.shape[-1],
            "action_dim": action_dim,
            "num_agents": len(agent_names),
        })

        return cls(
            rng=master_rng,
            network=train_state,
            agent_names=tuple(agent_names),
            config=flax.core.FrozenDict(cfg),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='bc',
            lr=3e-4,
            policy_hidden_dims=(512, 512, 512, 512),
            layer_norm=False,
            encoder=None,
        )
    )
