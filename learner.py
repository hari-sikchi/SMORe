from typing import Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import policy
import value_net
from actor import update_gcrl as awr_update_actor
from actor import update_gcrl_det as awr_update_actor_det
from common import GCRLBatch, Batch, MixedBatch, InfoDict, Model, PRNGKey
from critic_gcrl import update_v_smore, update_q_smore, update_q_smore_stable
from functools import partial


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=['double', 'vanilla', 'args'])
def _update_jit_stable(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: MixedBatch,expert_batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float, loss_temp: float,alpha: float, beta: float, double: bool, vanilla: bool, args,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    # Set up combined batches for expert data and observation data
    combined_observations = jnp.concatenate((batch.observations,expert_batch.observations),axis=0)
    combined_actions = jnp.concatenate((batch.actions,expert_batch.actions),axis=0)
    combined_achieved_goals = jnp.concatenate((batch.achieved_goals,expert_batch.achieved_goals),axis=0)
    combined_goals = jnp.concatenate((batch.goals,expert_batch.goals),axis=0)
    combined_next_observations = jnp.concatenate((batch.next_observations,expert_batch.next_observations),axis=0)
    combined_rewards = jnp.concatenate((batch.rewards,expert_batch.rewards),axis=0)
    combined_batch = GCRLBatch(observations=combined_observations, actions=combined_actions, next_observations=combined_next_observations,achieved_goals=combined_achieved_goals,goals=combined_goals, rewards=combined_rewards)
    goal_transition_indicator = jnp.concatenate((jnp.zeros(batch.observations.shape[0]),jnp.ones(expert_batch.observations.shape[0])),axis=0)
    
    is_expert_mask = batch.is_expert
    key, rng = jax.random.split(rng)
    for i in range(args.num_v_updates):
        new_value, value_info = update_v_smore(target_critic, value, combined_batch,is_expert_mask, expectile, loss_temp, alpha,beta, double, vanilla, key, args,goal_transition_indicator)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, combined_batch,is_expert_mask, temperature, double)

    # new_actor, actor_info = awr_update_actor_det(key, actor, target_critic,
    #                                          new_value, combined_batch,is_expert_mask, temperature, double)

    new_critic, critic_info = update_q_smore_stable(critic, new_value, combined_batch,is_expert_mask, discount, double, key, loss_temp, args,goal_transition_indicator)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


@partial(jax.jit, static_argnames=['double', 'vanilla', 'args'])
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: MixedBatch,expert_batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float, loss_temp: float,alpha: float, beta: float, double: bool, vanilla: bool, args,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    # Set up combined batches for expert data and observation data
    combined_observations = jnp.concatenate((batch.observations,expert_batch.observations),axis=0)
    combined_actions = jnp.concatenate((batch.actions,expert_batch.actions),axis=0)
    combined_achieved_goals = jnp.concatenate((batch.achieved_goals,expert_batch.achieved_goals),axis=0)
    combined_goals = jnp.concatenate((batch.goals,expert_batch.goals),axis=0)
    combined_next_observations = jnp.concatenate((batch.next_observations,expert_batch.next_observations),axis=0)
    combined_rewards = jnp.concatenate((batch.rewards,expert_batch.rewards),axis=0)
    combined_batch = GCRLBatch(observations=combined_observations, actions=combined_actions, next_observations=combined_next_observations,achieved_goals=combined_achieved_goals,goals=combined_goals, rewards=combined_rewards)
    goal_transition_indicator = jnp.concatenate((jnp.zeros(batch.observations.shape[0]),jnp.ones(expert_batch.observations.shape[0])),axis=0)
    
    is_expert_mask = batch.is_expert
    key, rng = jax.random.split(rng)
    for i in range(args.num_v_updates):
        new_value, value_info = update_v_smore(target_critic, value, combined_batch,is_expert_mask, expectile, loss_temp, alpha,beta, double, vanilla, key, args,goal_transition_indicator)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, combined_batch,is_expert_mask, temperature, double)

    # new_actor, actor_info = awr_update_actor_det(key, actor, target_critic,
    #                                          new_value, combined_batch,is_expert_mask, temperature, double)

    new_critic, critic_info = update_q_smore(critic, new_value, combined_batch,is_expert_mask, discount, double, key, loss_temp, args,goal_transition_indicator)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }

class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 layernorm: bool = False,
                 value_dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 loss_temp: float = 1.0,
                 double_q: bool = True,
                 vanilla: bool = True,
                 opt_decay_schedule: str = "cosine",
                 loss_type: str = "smore_stable",
                 alpha: float = 0.7,
                 beta: float = 0.8,
                 args=None):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.loss_temp = loss_temp
        self.double_q = double_q
        self.vanilla = vanilla
        self.alpha = alpha
        self.loss_type = loss_type
        self.beta=beta
        self.args = args
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)
        # actor_def = policy.DetPolicy(hidden_dims,
        #                                     action_dim,
        #                                     log_std_scale=1e-3,
        #                                     log_std_min=-5.0,
        #                                     dropout_rate=dropout_rate,
        #                                     state_dependent_std=False,
        #                                     tanh_squash_distribution=False)
        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = value_net.DoubleCritic(hidden_dims)

        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = value_net.ValueCritic(hidden_dims,
                                          layer_norm=layernorm,
                                          dropout_rate=value_dropout_rate)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: MixedBatch, expert_batch: Batch) -> InfoDict:
        if self.loss_type == "smore_stable":
            new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit_stable(
                self.rng, self.actor, self.critic, self.value, self.target_critic,
                batch,expert_batch, self.discount, self.tau, self.expectile, self.temperature, self.loss_temp, self.alpha,self.beta,  self.double_q, self.vanilla, self.args)
        else:
            new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
                self.rng, self.actor, self.critic, self.value, self.target_critic,
                batch,expert_batch, self.discount, self.tau, self.expectile, self.temperature, self.loss_temp, self.alpha,self.beta,  self.double_q, self.vanilla, self.args)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info

    def load(self, save_dir: str):
        self.actor = self.actor.load(os.path.join(save_dir, 'actor'))
        self.critic = self.critic.load(os.path.join(save_dir, 'critic'))
        self.value = self.value.load(os.path.join(save_dir, 'value'))
        self.target_critic = self.target_critic.load(os.path.join(save_dir, 'critic'))

    def save(self, save_dir: str):
        self.actor.save(os.path.join(save_dir, 'actor'))
        self.critic.save(os.path.join(save_dir, 'critic'))
        self.value.save(os.path.join(save_dir, 'value'))
