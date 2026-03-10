import copy
from typing import Any, Dict, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, DQN, Value
from util import *


class ScoutEfficientAgent(flax.struct.PyTreeNode):
    """Scout agent with DQN-style Q network for discrete environments.

    Key optimization: For discrete actions, uses DQN(obs) -> Q(s, all_actions)
    instead of Value(obs, one_hot_a) -> Q(s, a). This eliminates the A-fold
    observation expansion in _vgf_phi, _vgf_select, lvc_loss, and critic_loss,
    reducing Q network evaluations by ~A times (e.g., 9x for 3m).

    Continuous environments use the original Value network (unchanged).
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

    # -------------------- VGF Helpers -------------------- #
    def _rbf_kernel(self, X, Y):
        X2 = jnp.sum(X * X, axis=-1, keepdims=True)
        Y2 = jnp.sum(Y * Y, axis=-1, keepdims=True).transpose(0, 2, 1)
        XY = jnp.matmul(X, Y.transpose(0, 2, 1))
        d2 = jnp.maximum(X2 + Y2 - 2.0 * XY, 0.0)
        h = jnp.median(d2, axis=(1, 2)) / (2.0 * jnp.log(X.shape[1] + 1.0))
        sigma = jnp.sqrt(jnp.maximum(h, 1e-12))[:, None, None]
        gamma = 1.0 / (1e-6 + 2.0 * (sigma**2))
        K = jnp.exp(-gamma * d2)
        return K, d2, 2.0 * gamma

    def _vgf_phi(self, obs_rep, particles, alpha):
        is_discrete = self.config['is_discrete']
        q_name = 'q' if is_discrete else 'critic'

        if is_discrete:
            def sum_q(lg):
                obs_f = obs_rep.reshape(-1, obs_rep.shape[-1])
                lg_f = lg.reshape(-1, lg.shape[-1])
                pi = jax.nn.softmax(lg_f, axis=-1)

                # DQN: single forward pass gives Q for ALL actions
                qs = self.network.select(q_name)(obs_f)  # (ensemble, m*p, A)
                q_mean = qs.mean(axis=0)  # (m*p, A)
                batch_mean = jax.lax.stop_gradient(jnp.mean(q_mean))
                batch_std = jax.lax.stop_gradient(jnp.std(q_mean)) + 1e-6
                q_norm = (q_mean - batch_mean) / batch_std

                exp_q = jnp.sum(pi * q_norm, axis=-1)
                return jnp.sum(exp_q)
        else:
            def sum_q(actions):
                obs_f = obs_rep.reshape(-1, obs_rep.shape[-1])
                act_f = actions.reshape(-1, actions.shape[-1])
                qs = self.network.select(q_name)(obs_f, actions=act_f)
                q = qs.mean(axis=0)
                return jnp.sum(q)

        score = jax.grad(sum_q)(particles)
        particles_stop = jax.lax.stop_gradient(particles)
        K_xx, K_dist, K_gamma2 = self._rbf_kernel(particles, particles_stop)
        K_xx = jax.lax.stop_gradient(K_xx)

        def sum_K(x):
            return jnp.sum(self._rbf_kernel(x, particles_stop)[0])

        grad_K = -jax.grad(sum_K)(particles)
        phi = (jnp.matmul(K_xx, score) / alpha + grad_K) / particles.shape[1]
        term_1 = jnp.matmul(grad_K, score.transpose((0, 2, 1)))
        term_2 = alpha * K_xx * (particles.shape[-1] * K_gamma2 - K_dist * K_gamma2**2)
        trace = (term_1 + term_2).mean(axis=(1, 2))
        return phi, trace

    def _vgf_refine(self, obs_rep, init_particles, steps, lr, alpha):
        particles = init_particles
        entropy = jnp.zeros((init_particles.shape[0],), dtype=init_particles.dtype)
        bc_reg = self.config.get('vgf_bc_reg', 0.0)
        grad_clip = self.config.get('vgf_grad_clip', 0.0)
        for _ in range(steps):
            phi, tr = self._vgf_phi(obs_rep, particles, alpha)
            if bc_reg > 0:
                phi = phi - bc_reg * (particles - init_particles)
            if grad_clip > 0:
                phi_norm = jnp.linalg.norm(phi, axis=-1, keepdims=True)
                phi = phi * jnp.minimum(1.0, grad_clip / (phi_norm + 1e-8))
            if self.config['is_discrete']:
                particles = particles + lr * phi
            else:
                particles = jnp.clip(particles + lr * phi, -1, 1)
            entropy = entropy + lr * tr
        return particles, entropy

    def _vgf_select(self, obs_rep, particles, mode='max', module=None, legal_masks=None):
        is_discrete = self.config['is_discrete']
        if module is None:
            module = 'target_q' if is_discrete else 'target_critic'

        if is_discrete:
            obs_f = obs_rep.reshape(-1, obs_rep.shape[-1])
            lg_f = particles.reshape(-1, particles.shape[-1])
            A = lg_f.shape[-1]

            if legal_masks is not None:
                P = particles.shape[1]
                masks_tiled = jnp.repeat(legal_masks[:, None, :].astype(bool), P, axis=1).reshape(-1, A)
                large_neg = jnp.asarray(-1e9, dtype=lg_f.dtype)
                lg_f_eff = jnp.where(masks_tiled, lg_f, large_neg)
            else:
                lg_f_eff = lg_f

            pi = jax.nn.softmax(lg_f_eff, axis=-1)

            # DQN: single forward pass for all Q values
            qs = self.network.select(module)(obs_f)  # (ensemble, m*p, A)
            if self.config.get('q_agg', 'min') == 'min':
                q_mean = qs.min(axis=0)  # (m*p, A)
            else:
                q_mean = qs.mean(axis=0)
            exp_q = jnp.sum(pi * q_mean, axis=-1).reshape(particles.shape[0], particles.shape[1])

            if mode == 'max':
                idx = jnp.argmax(exp_q, axis=1)
                sel_logits = particles[jnp.arange(particles.shape[0]), idx]
            else:
                sel_logits = particles.mean(axis=1)

            if legal_masks is not None:
                masks_mpa = jnp.broadcast_to(legal_masks[:, None, :].astype(bool), (particles.shape[0], particles.shape[1], particles.shape[-1]))
                if mode == 'max':
                    sel_idx = jnp.argmax(exp_q, axis=1)
                    sel_mask = masks_mpa[jnp.arange(particles.shape[0]), sel_idx]
                else:
                    sel_mask = jnp.any(masks_mpa, axis=1)
                large_neg = jnp.asarray(-1e9, dtype=sel_logits.dtype)
                sel_logits = jnp.where(sel_mask, sel_logits, large_neg)

            return jnp.argmax(sel_logits, axis=-1)
        else:
            obs_f = obs_rep.reshape(-1, obs_rep.shape[-1])
            act_f = particles.reshape(-1, particles.shape[-1])
            qs = self.network.select(module)(obs_f, actions=act_f)
            q = qs.mean(axis=0).reshape(particles.shape[0], particles.shape[1])
            if mode == 'max':
                idx = jnp.argmax(q, axis=1)
                sel = particles[jnp.arange(particles.shape[0]), idx]
            else:
                sel = particles.mean(axis=1)
            return sel

    # -------------------- LVC Loss -------------------- #
    def lvc_loss(self, batch, grad_params, rng):
        A = self.config['action_dim']
        obs = batch['observations'][:-1]
        T, B, N, O = obs.shape

        # DQN: single forward pass, no A-fold expansion
        q_all = self.network.select('q')(obs, params=grad_params)  # (ensemble, T, B, N, A)
        q_all = q_all.mean(axis=0)  # (T, B, N, A)
        q_flat = q_all.reshape(-1, A)  # (M, A)

        tau = self.config.get('lvc_temperature', 1.0)
        log_q_probs = jax.nn.log_softmax(q_flat / tau, axis=-1)

        actions_flat = batch['actions'][:-1].reshape(-1)
        eps = self.config.get('lvc_label_smoothing', 0.1)
        beta = (1 - eps) * jax.nn.one_hot(actions_flat, A) + eps / A

        lvc = -jnp.sum(beta * log_q_probs, axis=-1).mean()
        return lvc, {'lvc_loss': lvc}

    # -------------------- Losses -------------------- #
    def critic_loss(self, batch, grad_params, rng):
        is_discrete = self.config['is_discrete']
        q_name = 'q' if is_discrete else 'critic'
        target_q_name = 'target_q' if is_discrete else 'target_critic'

        next_obs = batch['observations'][1:]
        t, b, n, o = next_obs.shape
        m = t * b * n
        next_obs_flat = jnp.reshape(next_obs, (m, o))

        p = self.config['vgf_particles']
        obs_rep = jnp.repeat(next_obs_flat[:, None, :], p, axis=1)
        rng, noise_rng = jax.random.split(rng)

        if is_discrete:
            noises = jax.random.normal(noise_rng, (m * p, self.config['action_dim']))
            bc_particles = self.compute_flow_logits(
                obs_rep.reshape(m * p, o), noises=noises,
                is_encoded=self.config.get('use_lstm', False),
            ).reshape(m, p, -1)
        else:
            noises = jax.random.normal(noise_rng, (m * p, self.config['action_dim']))
            bc_particles = self.compute_flow_actions(
                obs_rep.reshape(m * p, o), noises=noises,
            ).reshape(m, p, -1)

        particles, entropy = self._vgf_refine(
            obs_rep, bc_particles, self.config['train_vgf_steps'],
            self.config['vgf_lr'], self.config['vgf_alpha'],
        )

        if is_discrete:
            next_actions = self._vgf_select(
                obs_rep, particles, mode=self.config['train_particle_select'], module=target_q_name,
            )
            next_actions = jnp.reshape(next_actions, (t, b, n))

            # DQN target: get all Q values, then index by selected action
            next_q_all = self.network.select(target_q_name)(batch['observations'][1:])  # (ensemble, t, b, n, A)
            next_q_all = next_q_all.min(axis=0) if self.config.get('q_agg', 'mean') == 'min' else next_q_all.mean(axis=0)  # (t, b, n, A)
            next_q = jnp.take_along_axis(next_q_all, next_actions[..., None], axis=-1).squeeze(-1)  # (t, b, n)
        else:
            next_actions = self._vgf_select(
                obs_rep, particles, mode=self.config['train_particle_select'], module=target_q_name,
            )
            next_actions_input = jnp.reshape(next_actions, (t, b, n, -1))
            next_actions_input = jnp.clip(next_actions_input, -1, 1)
            next_qs = self.network.select(target_q_name)(batch['observations'][1:], actions=next_actions_input)
            next_q = next_qs.min(axis=0) if self.config.get('q_agg', 'mean') == 'min' else next_qs.mean(axis=0)

        entropy = entropy.reshape(t, b, n)
        target_q = batch['rewards'][:-1] + self.config['discount'] * (1.0 - batch['terminals'][1:]) * (
            next_q + (entropy if self.config.get('use_entropy', False) else 0.0)
        )
        q_clip_val = self.config.get('q_clip', 0)
        if q_clip_val > 0:
            target_q = jnp.clip(target_q, -q_clip_val, q_clip_val)

        if is_discrete:
            # DQN current: get all Q values, then index by taken action
            q_all = self.network.select(q_name)(batch['observations'][:-1], params=grad_params)  # (ensemble, t, b, n, A)
            cur_actions = batch['actions'][:-1]  # (t, b, n) integers
            q_ens = jnp.take_along_axis(q_all, cur_actions[None, ..., None], axis=-1).squeeze(-1)  # (ensemble, t, b, n)

            q_cur = q_ens.min(axis=0) if self.config.get('q_agg', 'mean') == 'min' else q_ens.mean(axis=0)
            mixed_target_q = jnp.sum(target_q, axis=-1)
            mixed_q = jnp.sum(q_cur, axis=-1)

            mu_y = jax.lax.stop_gradient(jnp.mean(mixed_target_q))
            abs_diff = jnp.abs(mixed_target_q - mu_y)
            mad_y = jax.lax.stop_gradient(jnp.mean(abs_diff)) + 1e-6
            q_hat = (mixed_q - mu_y) / mad_y
            y_hat = (mixed_target_q - mu_y) / mad_y
            critic_loss = jnp.mean(0.5 * (q_hat - y_hat) ** 2)
        else:
            cur_actions_input = batch['actions'][:-1]
            q_ens = self.network.select(q_name)(batch['observations'][:-1], actions=cur_actions_input, params=grad_params)
            if self.config.get('critic_loss', 'td') == 'td':
                q1, q2 = q_ens[0], q_ens[1]
                critic_loss = (((target_q - q1) ** 2) + ((target_q - q2) ** 2)).mean()
            else:
                adv = target_q - q_ens
                alpha_val = self.config.get('vgf_alpha', 1.0)
                z = jnp.clip(adv / alpha_val, -5.0, 5.0)
                max_z = jnp.max(z)
                max_z = jnp.where(max_z < -1.0, jnp.asarray(-1.0, dtype=z.dtype), max_z)
                max_z = jax.lax.stop_gradient(max_z)
                critic_loss = (jnp.exp(z - max_z) - z * jnp.exp(-max_z) - jnp.exp(-max_z)).mean()

        q_mean = mixed_q.mean() if is_discrete else q_ens.mean()
        q_max = mixed_q.max() if is_discrete else q_ens.max()
        q_min = mixed_q.min() if is_discrete else q_ens.min()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q_mean,
            'q_max': q_max,
            'q_min': q_min,
            'entropy_q': entropy.mean(),
        }

    def actor_bc_loss(self, batch, grad_params, rng):
        is_discrete = self.config['is_discrete']
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        if is_discrete:
            T, B, N = batch['actions'].shape
            A = self.config['action_dim']
            x_1 = jax.nn.one_hot(batch['actions'], A)
            x_0 = jax.random.normal(x_rng, (*x_1.shape[:-1], A))
            t_scalar = jax.random.uniform(t_rng, (*x_1.shape[:-1], 1))
            x_t = (1 - t_scalar) * x_0 + t_scalar * x_1
            vel = x_1 - x_0

            t_embed = self._time_sin_embed(t_scalar)
            pred = self.network.select('actor_bc_flow')(
                batch['observations'], x_t, t_embed, params=grad_params,
                is_encoded=self.config.get('use_lstm', False),
            )
        else:
            x_0 = jax.random.normal(x_rng, (*batch['actions'].shape[:-1], self.config['action_dim']))
            x_1 = batch['actions']
            t_scalar = jax.random.uniform(t_rng, (*batch['actions'].shape[:-1], 1))
            x_t = (1 - t_scalar) * x_0 + t_scalar * x_1
            vel = x_1 - x_0

            pred = self.network.select('actor_bc')(batch['observations'], x_t, t_scalar, params=grad_params)

        bc_flow_loss = jnp.mean((pred - vel) ** 2)
        return bc_flow_loss, {'bc_flow_loss': bc_flow_loss}

    def actor_bc_loss_joint(self, batch, grad_params, rng):
        is_discrete = self.config['is_discrete']
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        if is_discrete:
            T, B, N = batch['actions'].shape
            A = self.config['action_dim']
            actions_oh = jax.nn.one_hot(batch['actions'], A)
            x_1 = actions_oh.reshape(T, B, 1, N * A)
            x_0 = jax.random.normal(x_rng, x_1.shape)
            t_scalar = jax.random.uniform(t_rng, (*x_1.shape[:-1], 1))
            x_t = (1 - t_scalar) * x_0 + t_scalar * x_1
            vel = x_1 - x_0

            obs_joint = batch['observations'].reshape(T, B, 1, -1)

            t_embed = self._time_sin_embed(t_scalar)
            pred = self.network.select('actor_bc_flow_joint')(
                obs_joint, x_t, t_embed, params=grad_params,
                is_encoded=self.config.get('use_lstm', False),
            )
        else:
            T, B, N, A = batch['actions'].shape
            x_1 = batch['actions'].reshape(T, B, 1, N * A)
            x_0 = jax.random.normal(x_rng, x_1.shape)
            t_scalar = jax.random.uniform(t_rng, (*x_1.shape[:-1], 1))
            x_t = (1 - t_scalar) * x_0 + t_scalar * x_1
            vel = x_1 - x_0

            obs_joint = batch['observations'].reshape(T, B, 1, -1)
            pred = self.network.select('actor_bc_joint')(obs_joint, x_t, t_scalar, params=grad_params)

        joint_bc_loss = jnp.mean((pred - vel) ** 2)
        return joint_bc_loss, {'joint_bc_flow_loss': joint_bc_loss}

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

        if self.config.get('joint_flow_bc', False):
            bc_loss, bc_info = self.actor_bc_loss_joint(batch, grad_params, actor_rng)
        else:
            bc_loss, bc_info = self.actor_bc_loss(batch, grad_params, actor_rng)
        for k, v in bc_info.items():
            info[f'actor/{k}'] = v

        critic_coef = self.config.get('critic_coef', 1.0)
        loss = critic_coef * critic_loss + bc_loss

        lvc_coef = self.config.get('lvc_coef', 0.0)
        if lvc_coef > 0 and self.config['is_discrete']:
            lvc_val, lvc_info = self.lvc_loss(batch, grad_params, rng)
            for k, v in lvc_info.items():
                info[f'lvc/{k}'] = v
            loss = loss + lvc_coef * lvc_val

        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, step):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        target_name = 'q' if self.config['is_discrete'] else 'critic'
        target_reset_interval = self.config.get('target_reset_interval', 0)
        if target_reset_interval > 0:
            do_hard = (step % target_reset_interval == 0)
            new_target = jax.tree_util.tree_map(
                lambda p, tp: jnp.where(do_hard, p, p * self.config['tau'] + tp * (1 - self.config['tau'])),
                new_network.params[f'modules_{target_name}'],
                new_network.params[f'modules_target_{target_name}'],
            )
            new_network.params[f'modules_target_{target_name}'] = new_target
        else:
            self.target_update(new_network, target_name)

        return self.replace(network=new_network, rng=new_rng), info

    # -------------------- Action Sampling (Evaluation) -------------------- #
    def _masked_softmax(self, logits, mask=None):
        if mask is None:
            return jax.nn.softmax(logits, axis=-1)
        large_neg = -1e9
        masked_logits = jnp.where(mask, logits, large_neg)
        return jax.nn.softmax(masked_logits, axis=-1)

    def sample_actions_with_carry(self,
                                  observations: Dict[str, jnp.ndarray],
                                  carry,
                                  seed,
                                  legal_actions=None,
                                  reset_mask=None,
                                  eval_vgf_steps=None,
                                  temperature: float = 0.0,
                                  stochastic: bool = False):
        rng = seed if seed is not None else self.rng
        action_seed, _ = jax.random.split(rng)
        N = len(self.agent_names)
        obs_with_ids = [concat_agent_id_to_obs(observations[agent], i, self.config['num_agents']) for i, agent in enumerate(self.agent_names)]
        obs_tensor = jnp.stack(obs_with_ids, axis=0)

        if self.config.get('use_lstm', False):
            obs_seq = obs_tensor[None, None, ...]
            if reset_mask is None:
                resets = jnp.ones((1, 1, N), dtype=jnp.bool_) if carry is None else jnp.zeros((1, 1, N), dtype=jnp.bool_)
            else:
                resets = reset_mask[None, None, :].astype(jnp.bool_)
            enc, new_carry = self.network.select('seq_encoder')(obs_seq, resets, initial_carry=carry, return_carry=True)
            obs_tensor = enc[0, 0]
        else:
            new_carry = None

        p = self.config['vgf_particles']
        steps = self.config['eval_vgf_steps'] if eval_vgf_steps is None else eval_vgf_steps
        obs_rep = jnp.repeat(obs_tensor[:, None, :], p, axis=1)
        noises = jax.random.normal(action_seed, (obs_tensor.shape[0] * p, self.config['action_dim']))

        bc_logits = self.compute_flow_logits(
            obs_rep.reshape(-1, obs_tensor.shape[-1]),
            noises=noises,
            is_encoded=self.config.get('use_lstm', False),
        ).reshape(obs_tensor.shape[0], p, -1)

        if legal_actions is not None:
            masks = jnp.stack([legal_actions[agent].astype(bool) for agent in self.agent_names], axis=0)
        else:
            masks = None

        particles, _ = self._vgf_refine(obs_rep, bc_logits, steps, self.config['vgf_lr'], self.config['vgf_alpha'])
        actions = self._vgf_select(obs_rep, particles, mode=self.config['eval_particle_select'], module='q', legal_masks=masks)
        return {agent: actions[i] for i, agent in enumerate(self.agent_names)}, new_carry

    @jax.jit
    def sample_actions(self, observations: Dict[str, jnp.ndarray], seed, legal_actions=None, eval_vgf_steps=None, temperature=0.0, stochastic=False):
        if self.config['is_discrete']:
            actions, _ = self.sample_actions_with_carry(
                observations, carry=None, seed=seed,
                legal_actions=legal_actions, reset_mask=None,
                eval_vgf_steps=eval_vgf_steps,
                temperature=temperature, stochastic=stochastic,
            )
            return actions
        else:
            rng = seed if seed is not None else self.rng
            action_seed, _ = jax.random.split(rng)
            obs_with_ids = [concat_agent_id_to_obs(observations[agent], i, self.config['num_agents']) for i, agent in enumerate(self.agent_names)]
            obs_tensor = jnp.stack(obs_with_ids, axis=0)
            p = self.config['vgf_particles']
            steps = self.config['eval_vgf_steps'] if eval_vgf_steps is None else eval_vgf_steps
            obs_rep = jnp.repeat(obs_tensor[:, None, :], p, axis=1)
            noises = jax.random.normal(action_seed, (obs_tensor.shape[0] * p, self.config['action_dim']))
            bc = self.compute_flow_actions(obs_rep.reshape(-1, obs_tensor.shape[-1]), noises=noises).reshape(obs_tensor.shape[0], p, -1)
            particles, _ = self._vgf_refine(obs_rep, bc, steps, self.config['vgf_lr'], self.config['vgf_alpha'])
            actions = self._vgf_select(obs_rep, particles, mode=self.config['eval_particle_select'], module='critic')
            actions = jnp.clip(actions, -1, 1)
            return {agent: actions[i] for i, agent in enumerate(self.agent_names)}

    # -------------------- Flow Integration (lax.scan) -------------------- #
    @jax.jit
    def compute_flow_logits(self, observations, noises, is_encoded=False):
        if (self.config['encoder'] is not None) and (not self.config.get('use_lstm', False)) and (not is_encoded):
            observations = self.network.select('actor_bc_flow_encoder')(observations)

        flow_steps = self.config['flow_steps']

        def scan_body(actions, i):
            t_scalar = jnp.full((*observations.shape[:-1], 1), i / flow_steps)
            t_embed = self._time_sin_embed(t_scalar)
            vels = self.network.select('actor_bc_flow')(observations, actions, t_embed, is_encoded=True)
            actions = actions + vels / flow_steps
            return actions, None

        actions, _ = jax.lax.scan(scan_body, noises, jnp.arange(flow_steps))
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_encoder')(observations)

        flow_steps = self.config['flow_steps']

        def scan_body(actions, i):
            t = jnp.full((*observations.shape[:-1], 1), i / flow_steps)
            vels = self.network.select('actor_bc')(observations, actions, t, is_encoded=True)
            actions = actions + vels / flow_steps
            return actions, None

        actions, _ = jax.lax.scan(scan_body, noises, jnp.arange(flow_steps))
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

        encoders = dict()
        if is_discrete:
            q_key, target_q_key = 'q', 'target_q'
            actor_bc_key = 'actor_bc_flow'
            if (config['encoder'] is not None) and (not config.get('use_lstm', False)):
                encoder_module = encoder_modules[config['encoder']]
                encoders['q'] = encoder_module()
                encoders['actor_bc_flow'] = encoder_module()
        else:
            q_key, target_q_key = 'critic', 'target_critic'
            actor_bc_key = 'actor_bc'
            if config['encoder'] is not None:
                encoder_module = encoder_modules[config['encoder']]
                encoders['critic'] = encoder_module()
                encoders['actor_bc'] = encoder_module()

        # DQN for discrete (obs -> all Q values), Value for continuous
        if is_discrete:
            q_def = DQN(
                hidden_dims=config.get('value_hidden_dims', config.get('critic_hidden_dims', (256, 256, 256, 256))),
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                num_ensembles=2,
                encoder=encoders.get(q_key),
            )
        else:
            q_def = Value(
                hidden_dims=config.get('value_hidden_dims', config.get('critic_hidden_dims', (256, 256, 256, 256))),
                layer_norm=config['layer_norm'],
                num_ensembles=2,
                encoder=encoders.get(q_key),
            )

        actor_bc_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get(actor_bc_key),
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

        # DQN init: only obs (no actions needed)
        if is_discrete:
            network_info = dict(
                **{q_key: (q_def, (ex_obs_init,))},
                **{target_q_key: (copy.deepcopy(q_def), (ex_obs_init,))},
                **{actor_bc_key: (actor_bc_def, (ex_obs_init, ex_actions, ex_times))},
            )
        else:
            network_info = dict(
                **{q_key: (q_def, (ex_obs_init, ex_actions))},
                **{target_q_key: (copy.deepcopy(q_def), (ex_obs_init, ex_actions))},
                **{actor_bc_key: (actor_bc_def, (ex_obs_init, ex_actions, ex_times))},
            )

        if config.get('joint_flow_bc', False):
            n_agents = len(agent_names)
            joint_action_dim = action_dim * n_agents
            if is_discrete:
                joint_obs_dim = ex_obs_init.shape[-1] * n_agents
                joint_bc_key = 'actor_bc_flow_joint'
            else:
                joint_obs_dim = ex_obs_init.shape[-1] * n_agents
                joint_bc_key = 'actor_bc_joint'
            joint_bc_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=joint_action_dim,
                layer_norm=config['actor_layer_norm'],
            )
            ex_joint_obs = jnp.zeros((1, 1, 1, joint_obs_dim), dtype=jnp.float32)
            ex_joint_act = jnp.zeros((1, 1, 1, joint_action_dim), dtype=jnp.float32)
            if is_discrete:
                kfreq = int(config.get('t_embed_frequencies', 8))
                ex_joint_times = jnp.zeros((1, 1, 1, 2 * kfreq), dtype=jnp.float32)
            else:
                ex_joint_times = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
            network_info[joint_bc_key] = (joint_bc_def, (ex_joint_obs, ex_joint_act, ex_joint_times))

        if encoders.get(actor_bc_key) is not None:
            encoder_key = f'{actor_bc_key}_encoder'
            network_info[encoder_key] = (encoders.get(actor_bc_key), (ex_obs_with_id,))

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
        network_params = network_def.init(init_rng, **network_args)['params']
        lr_schedule = config.get('lr_schedule', 'constant')
        if lr_schedule == 'cosine':
            total_steps = int(config.get('total_steps', 500_000))
            warmup_steps = int(config.get('lr_warmup_steps', 0))
            lr_fn = optax.warmup_cosine_decay_schedule(
                init_value=0.0 if warmup_steps > 0 else config['lr'],
                peak_value=config['lr'],
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_value=config['lr'] * 0.01,
            )
        else:
            lr_fn = config['lr']

        max_grad_norm = config.get('max_grad_norm', 0)
        if max_grad_norm > 0:
            network_tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr_fn),
            )
        else:
            network_tx = optax.adam(lr_fn)
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params[f'modules_{target_q_key}'] = params[f'modules_{q_key}']

        config['ob_dims'] = ex_obs_with_id.shape[:-1]
        config['action_dim'] = action_dim
        config['num_agents'] = len(agent_names)

        return cls(
            rng=rng,
            network=network,
            agent_names=tuple(agent_names),
            config=flax.core.FrozenDict(**config),
        )


def get_config(variant=None):
    if variant == 'v1':
        return _get_v1_config()
    if variant == 'v2':
        return _get_v2_config()
    return ml_collections.ConfigDict(
        dict(
            agent_name='scout_eff',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            is_discrete=False,
            lr=3e-4,
            actor_hidden_dims=(256, 256, 256, 256),
            value_hidden_dims=(256, 256, 256, 256),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            critic_loss='td',
            flow_steps=10,
            encoder=ml_collections.config_dict.placeholder(str),
            vgf_lr=1e-3,
            vgf_alpha=1.0,
            vgf_particles=10,
            train_vgf_steps=3,
            eval_vgf_steps=0,
            train_particle_select='mean',
            eval_particle_select='max',
            use_entropy=False,
            max_grad_norm=1.0,
            critic_coef=1.0,
            q_clip=0.0,
            vgf_bc_reg=0.1,
            vgf_grad_clip=1.0,
            lvc_coef=0.5,
            lvc_temperature=1.0,
            lvc_label_smoothing=0.1,
            joint_flow_bc=False,
            target_reset_interval=0,
            lr_schedule='constant',
            lr_warmup_steps=0,
            total_steps=500000,
            t_embed_frequencies=8,
            use_lstm=False,
            lstm_hidden_dim=128,
            lstm_layers=1,
            lstm_pre_mlp_dims=(128,),
            lstm_layer_norm=True,
        )
    )


def _get_v1_config():
    config = get_config()
    config.agent_name = 'scout_eff_v1'
    config.train_vgf_steps = 3
    config.vgf_lr = 1e-3
    config.vgf_bc_reg = 0.1
    config.vgf_grad_clip = 1.0
    config.lvc_coef = 0.5
    config.max_grad_norm = 1.0
    return config


def _get_v2_config():
    config = get_config()
    config.agent_name = 'scout_eff_v2'
    config.train_vgf_steps = 1
    config.vgf_lr = 3e-3
    config.vgf_bc_reg = 0.05
    config.vgf_grad_clip = 1.0
    config.lvc_coef = 0.0
    config.lr_schedule = 'cosine'
    config.lr_warmup_steps = 5000
    config.max_grad_norm = 1.0
    return config
