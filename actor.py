from typing import Tuple
import jax
import jax.numpy as jnp
from common import Batch, InfoDict, Model, Params, PRNGKey


def update(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float, double: bool) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info



def update_gcrl(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, is_expert_mask, temperature: float, double: bool) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs)[:exp_a.shape[0]//2].mean()
        loss_dict = {'actor_loss': actor_loss, 'adv': q - v}
        return actor_loss, loss_dict

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info['clipped_adv'] = exp_a.mean()
    return new_actor, info





def update_gcrl_det(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, is_expert_mask, temperature: float, double: bool) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        actor_loss = (exp_a.reshape(-1,1) * ((dist.mean() - batch.actions)**2)).mean()
        loss_dict = {'actor_loss': actor_loss, 'adv': q - v}
        return actor_loss, loss_dict

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info['clipped_adv'] = exp_a.mean()
    return new_actor, info

