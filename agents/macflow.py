import copy
from typing import Any, Dict, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from util import *


class MACFlowAgent(flax.struct.PyTreeNode):
    """Unified MACFlow agent supporting both discrete and continuous actions.

    Set config['is_discrete'] = True for discrete (SMAC-style) environments
    or False for continuous (MAMuJoCo-style) environments.
    """
    rng: Any
    network: Any
    agent_names: Sequence[str] = nonpytree_field()
    config: Any = nonpytree_field()

    # -------------------- Time Embedding (discrete only) -------------------- #
    def _time_sin_embed(self, ts):
        kfreq = int(self.config.get('t_embed_frequencies', 8))
        freqs = jnp.asarray([2 ** i for i in range(kfreq)], dtype=ts.dtype) * jnp.pi
        ang = ts * freqs
        return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)

    # -------------------- Critic Loss -------------------- #
    def critic_loss(self, batch, grad_params, rng):
        is_discrete = self.config['is_discrete']
        q_name = 'q' if is_discrete else 'critic'
        target_q_name = 'target_q' if is_discrete else 'target_critic'

        rng, sample_rng = jax.random.split(rng)

        if is_discrete:
            next_actions_idx = self._sample_actions_train(batch['observations'][1:], seed=sample_rng)
            next_actions = jax.nn.one_hot(next_actions_idx, self.config['action_dim'])
        else:
            next_actions = self.sample_actions_batch(batch['observations'][1:], seed=sample_rng)
            next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select(target_q_name)(batch['observations'][1:], actions=next_actions)
        if self.config.get('q_agg', 'min') == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'][:-1] + self.config['discount'] * (1.0 - batch['terminals'][1:]) * next_q

        if is_discrete:
            cur_actions = jax.nn.one_hot(batch['actions'][:-1], self.config['action_dim'])
        else:
            cur_actions = batch['actions'][:-1]

        q = self.network.select(q_name)(batch['observations'][:-1], actions=cur_actions, params=grad_params)
        if self.config.get('q_agg', 'min') == 'min':
            q_cur = q.min(axis=0)
        else:
            q_cur = q.mean(axis=0)

        if is_discrete:
            mixed_target_q = jnp.sum(target_q, axis=-1)
            mixed_q = jnp.sum(q_cur, axis=-1)
            critic_loss = 0.5 * jnp.mean(jnp.square(mixed_q - mixed_target_q))
        else:
            mixed_target_q = target_q.mean(axis=-1)
            mixed_q = q.mean(axis=-1)
            critic_loss = jnp.square((mixed_q - mixed_target_q)).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': mixed_q.mean(),
            'q_max': mixed_q.max(),
            'q_min': mixed_q.min(),
        }

    # -------------------- Actor Loss -------------------- #
    def actor_loss(self, batch, grad_params, rng):
        is_discrete = self.config['is_discrete']
        q_name = 'q' if is_discrete else 'critic'

        rng, x_rng, t_rng = jax.random.split(rng, 3)

        if is_discrete:
            T, B, N = batch['actions'].shape
            action_dim = self.config['action_dim']
            obs_slice = batch['observations'][:-1]
            act_slice = batch['actions'][:-1]
            x_1 = jax.nn.one_hot(act_slice, action_dim)
            x_0 = jax.random.normal(x_rng, (*x_1.shape[:-1], action_dim))
            t_scalar = jax.random.uniform(t_rng, (*x_1.shape[:-1], 1))
            x_t = (1 - t_scalar) * x_0 + t_scalar * x_1
            vel = x_1 - x_0

            t_embed = self._time_sin_embed(t_scalar)
            pred = self.network.select('actor_bc_flow')(
                obs_slice, x_t, t_embed, params=grad_params,
                is_encoded=self.config.get('use_lstm', False),
            )
        else:
            obs_slice = batch['observations']
            x_0 = jax.random.normal(x_rng, (*batch['actions'].shape[:-1], self.config['action_dim']))
            x_1 = batch['actions']
            t_scalar = jax.random.uniform(t_rng, (*batch['actions'].shape[:-1], 1))
            x_t = (1 - t_scalar) * x_0 + t_scalar * x_1
            vel = x_1 - x_0

            pred = self.network.select('actor_bc_flow')(obs_slice, x_t, t_scalar, params=grad_params)

        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss
        rng, noise_rng = jax.random.split(rng)
        if is_discrete:
            noises = jax.random.normal(noise_rng, (*x_1.shape[:-1], action_dim))
            target_flow_actions = self.compute_flow_actions(
                obs_slice, noises=noises,
                is_encoded=self.config.get('use_lstm', False),
            )
            actor_logits = self.network.select('actor_onestep_flow')(
                obs_slice, noises, params=grad_params,
                is_encoded=self.config.get('use_lstm', False),
            )
            distill_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(actor_logits, target_flow_actions))
        else:
            noises = jax.random.normal(noise_rng, (*batch['actions'].shape[:-1], self.config['action_dim']))
            target_flow_actions = self.compute_flow_actions(obs_slice, noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(obs_slice, noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q-guidance loss
        if is_discrete:
            actor_actions_idx = jnp.argmax(actor_logits, axis=-1)
            actor_actions_oh = jax.nn.one_hot(actor_actions_idx, action_dim)
            qs_all = self.network.select(q_name)(obs_slice, actions=actor_actions_oh)
            qs = qs_all.mean(axis=0)
            mixed_q_for_actor = qs.mean(axis=-1)
        else:
            actor_actions = jnp.clip(
                self.network.select('actor_onestep_flow')(obs_slice, noises, params=grad_params),
                -1, 1,
            )
            qs = self.network.select(q_name)(obs_slice, actions=actor_actions)
            mixed_q_for_actor = jnp.mean(qs, axis=0)

        q_loss = -mixed_q_for_actor.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(mixed_q_for_actor).mean())
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': mixed_q_for_actor.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = self.rng if rng is None else rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = jnp.array(batch['terminals'], 'float32')

        observations = batch_concat_agent_id_to_obs(observations)

        obs_t = switch_two_leading_dims(observations)
        actions_t = switch_two_leading_dims(actions)
        rewards_t = switch_two_leading_dims(rewards)
        terminals_t = switch_two_leading_dims(terminals)

        if self.config.get('use_lstm', False):
            resets = jnp.zeros_like(terminals_t, dtype=jnp.bool_)
            resets = resets.at[0].set(True)
            resets = resets.at[1:].set(terminals_t[:-1] > 0.5)
            enc_obs_t = self.network.select('seq_encoder')(obs_t, resets)
            obs_in = enc_obs_t
        else:
            obs_in = obs_t

        batch = {
            'observations': obs_in,
            'actions': actions_t,
            'rewards': rewards_t,
            'terminals': terminals_t,
        }

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, step):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        target_name = 'q' if self.config['is_discrete'] else 'critic'
        self.target_update(new_network, target_name)

        return self.replace(network=new_network, rng=new_rng), info

    # -------------------- Helpers (discrete) -------------------- #
    def _masked_softmax(self, logits, mask=None):
        if mask is None:
            return jax.nn.softmax(logits, axis=-1)
        large_neg = -1e9
        masked_logits = jnp.where(mask, logits, large_neg)
        return jax.nn.softmax(masked_logits, axis=-1)

    @jax.jit
    def _sample_actions_train(self, observations, seed=None, temperature=1.0):
        """Sample discrete actions from onestep flow during training."""
        seed = self.rng if seed is None else seed
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(action_seed, (*observations.shape[:3], self.config['action_dim']))
        logits = self.network.select('actor_onestep_flow')(
            observations, noises,
            is_encoded=self.config.get('use_lstm', False),
        )
        logits = logits / jnp.maximum(temperature, 1e-6)
        return jnp.argmax(jax.nn.softmax(logits, axis=-1), axis=-1)

    @jax.jit
    def sample_actions_batch(self, observations, seed=None, temperature=1.0):
        """Sample continuous actions from onestep flow (for batch observations)."""
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(action_seed, (*observations.shape[:3], self.config['action_dim']))
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        return jnp.clip(actions, -1, 1)

    # -------------------- Action Sampling (Evaluation) -------------------- #
    def sample_actions_with_carry(self,
                                  observations: Dict[str, jnp.ndarray],
                                  carry,
                                  seed,
                                  legal_actions=None,
                                  reset_mask=None,
                                  temperature: float = 0.0,
                                  stochastic: bool = False,
                                  **kwargs):
        """Sample actions with carry for discrete evaluation (LSTM support)."""
        rng = seed if seed is not None else self.rng
        action_seed, _ = jax.random.split(rng)
        N = len(self.agent_names)
        obs_with_ids = [concat_agent_id_to_obs(observations[agent], i, N) for i, agent in enumerate(self.agent_names)]
        obs_tensor = jnp.stack(obs_with_ids, axis=0)

        if self.config.get('use_lstm', False):
            obs_seq = obs_tensor[None, None, ...]
            if reset_mask is None:
                resets = jnp.ones((1, 1, N), dtype=jnp.bool_) if carry is None else jnp.zeros((1, 1, N), dtype=jnp.bool_)
            else:
                resets = reset_mask[None, None, :].astype(jnp.bool_)
            enc, new_carry = self.network.select('seq_encoder')(
                obs_seq, resets, initial_carry=carry, return_carry=True,
            )
            obs_tensor = enc[0, 0]
        else:
            new_carry = None

        noises = jnp.zeros((N, self.config['action_dim']))
        logits = self.network.select('actor_onestep_flow')(
            obs_tensor[None, :], noises[None, :],
            is_encoded=self.config.get('use_lstm', False),
        )[0]
        logits = logits / jnp.maximum(temperature, 1e-6)

        if legal_actions is not None:
            masks = jnp.stack([legal_actions[agent].astype(bool) for agent in self.agent_names], axis=0)
            probs = self._masked_softmax(logits, masks)
        else:
            probs = jax.nn.softmax(logits, axis=-1)

        actions = jnp.argmax(probs, axis=-1)
        return {agent: actions[i] for i, agent in enumerate(self.agent_names)}, new_carry

    @jax.jit
    def sample_actions(self, observations, seed, legal_actions=None, temperature=0.0, stochastic=False, **kwargs):
        """Sample actions for evaluation.

        For discrete: observations is Dict[str, ndarray], returns Dict[str, int].
        For continuous: observations is Dict[str, ndarray], returns Dict[str, ndarray].
        """
        if self.config['is_discrete']:
            actions, _ = self.sample_actions_with_carry(
                observations, carry=None, seed=seed,
                legal_actions=legal_actions,
                reset_mask=None, temperature=temperature,
                stochastic=stochastic,
            )
            return actions
        else:
            action_seed, _ = jax.random.split(seed)
            if type(observations) is dict:
                obs_with_ids = [concat_agent_id_to_obs(observations[agent], i, self.config['num_agents']) for i, agent in enumerate(self.agent_names)]
                obs_tensor = jnp.stack(obs_with_ids, axis=0)
                noises = jax.random.normal(action_seed, (self.config['num_agents'], self.config['action_dim']))
                actions = self.network.select('actor_onestep_flow')(obs_tensor, noises)
                actions = jnp.clip(actions, -1, 1)
                return {agent: actions[i] for i, agent in enumerate(self.agent_names)}
            else:
                noises = jax.random.normal(action_seed, (*observations.shape[:3], self.config['action_dim']))
                actions = self.network.select('actor_onestep_flow')(observations, noises)
                return jnp.clip(actions, -1, 1)

    # -------------------- Flow Integration -------------------- #
    @jax.jit
    def compute_flow_actions(self, observations, noises, is_encoded=False):
        """Compute actions via multi-step flow integration.

        For discrete: returns integer action indices.
        For continuous: returns clipped continuous actions.
        """
        is_discrete = self.config['is_discrete']

        if is_discrete:
            if (self.config['encoder'] is not None) and (not self.config.get('use_lstm', False)) and (not is_encoded):
                observations = self.network.select('actor_bc_flow_encoder')(observations)
            steps = int(self.config['flow_steps'])

            def body(actions, i):
                t_scalar = jnp.full((*observations.shape[:-1], 1), i / steps)
                t_embed = self._time_sin_embed(t_scalar)
                vels = self.network.select('actor_bc_flow')(observations, actions, t_embed, is_encoded=True)
                actions = actions + vels / steps
                return actions, None

            actions, _ = jax.lax.scan(body, noises, jnp.arange(steps))
            return jnp.argmax(actions, axis=-1)
        else:
            if self.config['encoder'] is not None:
                observations = self.network.select('actor_bc_flow_encoder')(observations)
            actions = noises
            for i in range(self.config['flow_steps']):
                t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
                vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
                actions = actions + vels / self.config['flow_steps']
            return jnp.clip(actions, -1, 1)

    # -------------------- Factory -------------------- #
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, agent_names, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        is_discrete = config['is_discrete']

        if is_discrete:
            T, B, N, O = ex_observations.shape
            ob_dims = ex_observations.shape[1:]
            action_dim = ex_actions.shape[-1]
        else:
            action_dim = ex_actions.shape[-1]

        encoders = {}
        if is_discrete:
            q_key, target_q_key = 'q', 'target_q'
            if (config['encoder'] is not None) and (not config.get('use_lstm', False)):
                encoder_module = encoder_modules[config['encoder']]
                encoders['q'] = encoder_module()
                encoders['actor_bc_flow'] = encoder_module()
                encoders['actor_onestep_flow'] = encoder_module()
        else:
            q_key, target_q_key = 'critic', 'target_critic'
            if config['encoder'] is not None:
                encoder_module = encoder_modules[config['encoder']]
                encoders['critic'] = encoder_module()
                encoders['actor_bc_flow'] = encoder_module()
                encoders['actor_onestep_flow'] = encoder_module()

        q_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=int(config.get('value_ensembles', 2)),
            encoder=encoders.get(q_key),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        ex_obs_with_id = batch_concat_agent_id_to_obs(ex_observations)

        if is_discrete:
            kfreq = int(config.get('t_embed_frequencies', 8))
            ex_times = jnp.zeros((*ex_actions.shape[:-1], 2 * kfreq), dtype=jnp.float32)

            if config.get('use_lstm', False):
                enc_feat_dim = int(config.get('lstm_hidden_dim', 256))
                ex_obs_init = jnp.zeros((*ex_obs_with_id.shape[:-1], enc_feat_dim), dtype=jnp.float32)
            else:
                ex_obs_init = ex_obs_with_id
        else:
            ex_times = ex_actions[..., :1]
            ex_obs_init = ex_obs_with_id

        network_info = dict()
        network_info[q_key] = (q_def, (ex_obs_init, ex_actions))
        network_info[target_q_key] = (copy.deepcopy(q_def), (ex_obs_init, ex_actions))
        network_info['actor_bc_flow'] = (actor_bc_flow_def, (ex_obs_init, ex_actions, ex_times))
        network_info['actor_onestep_flow'] = (actor_onestep_flow_def, (ex_obs_init, ex_actions))

        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_obs_with_id,))

        if is_discrete and config.get('use_lstm', False):
            from utils.networks import SequenceLSTMEncoder
            point_enc = None
            if config.get('encoder', None) is not None:
                point_enc = encoder_modules[config['encoder']]
            seq_enc_def = SequenceLSTMEncoder(
                hidden_dim=config.get('lstm_hidden_dim', 256),
                num_layers=config.get('lstm_layers', 1),
                pre_mlp_dims=tuple(config.get('lstm_pre_mlp_dims', ())),
                layer_norm=config.get('lstm_layer_norm', False),
                point_encoder=(point_enc() if point_enc is not None else None),
            )
            dummy_resets = jnp.zeros(ex_actions.shape[:-1], dtype=jnp.bool_)
            network_info['seq_encoder'] = (seq_enc_def, (ex_obs_with_id, dummy_resets))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params[f'modules_{target_q_key}'] = params[f'modules_{q_key}']

        config['ob_dims'] = ex_obs_with_id.shape[:-1] if is_discrete else ex_obs_with_id.shape[:-1]
        config['action_dim'] = action_dim
        config['num_agents'] = len(agent_names)

        return cls(
            rng=rng,
            network=network,
            agent_names=tuple(agent_names),
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='macflow',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            is_discrete=False,
            lr=3e-4,
            actor_hidden_dims=(256, 256, 256, 256),
            value_hidden_dims=(256, 256, 256, 256),
            value_ensembles=2,
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=3.0,
            flow_steps=10,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
            # Discrete-only options
            t_embed_frequencies=8,
            use_lstm=False,
            lstm_hidden_dim=64,
            lstm_layers=1,
            lstm_pre_mlp_dims=(128,),
            lstm_layer_norm=True,
        )
    )
