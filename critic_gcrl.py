from typing import Tuple
import jax.numpy as jnp
import jax
from functools import partial
from common import GCRLBatch, Batch, InfoDict, Model, Params, PRNGKey

def grad_norm(model, params, obs, action, lambda_=10):

    @partial(jax.vmap, in_axes=(0, 0))
    @partial(jax.jacrev, argnums=1)
    def input_grad_fn(obs, action):
        return model.apply({'params': params}, obs, action)

    def grad_pen_fn(grad):
        # We use gradient penalties inspired from WGAN-LP loss which penalizes grad_norm > 1
        penalty = jnp.maximum(jnp.linalg.norm(grad1, axis=-1) - 1, 0)**2
        return penalty

    grad1, grad2 = input_grad_fn(obs, action)

    return grad_pen_fn(grad1), grad_pen_fn(grad2)


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return (weight * (diff**2))[:diff.shape[0]//2]

def update_v_smore(critic: Model, value: Model, batch: Batch, is_expert_mask,
             expectile: float, loss_temp: float, alpha:float, beta:float, double: bool, vanilla: bool, key: PRNGKey, args, goal_transition_indicator=None) -> Tuple[Model, InfoDict]:

    rng1, rng2 = jax.random.split(key)
    obs = batch.observations
    acts = batch.actions

    q1, q2 = critic(obs, acts) # this is target critic
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, obs)
        if vanilla:
            value_loss = expectile_loss(q - v,beta).mean()

        return value_loss, {
            'unseen_v_expert': (v[:v.shape[0]//2]*is_expert_mask).sum()/is_expert_mask.sum(),
            'unseen_v_suboptimal':(v[:v.shape[0]//2]*(1-is_expert_mask)).sum()/(1-is_expert_mask).sum(),
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)
    
    return new_value, info


def update_q_smore_stable(critic: Model, target_value: Model, batch: Batch, is_expert_mask,
             discount: float, double: bool, key: PRNGKey, loss_temp: float,  args, goal_transition_indicator=None) -> Tuple[Model, InfoDict]:
    
    next_v = target_value(batch.next_observations)
    neg_r = -2
    target_q = neg_r + discount  * next_v 

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        acts = batch.actions
        
        q1, q2 = critic.apply({'params': critic_params}, batch.observations, acts)
        v = target_value(batch.observations)

        def stable_loss(q,q_target,*args):
            loss_dict = {}
            alpha_coeff = (1-discount)*1e-4
            loss = ((q[q.shape[0]//2:]+neg_r/(1-discount))**2).mean()+alpha_coeff*((q[:q.shape[0]//2]-0)**2).mean()+0.5*((q[:q.shape[0]//2]-q_target[:q.shape[0]//2])**2).mean()
            loss_dict['critic_loss'] = loss

            return loss.mean(), loss_dict

        def original_loss(q,q_target,*args):
            loss_dict = {}
            alpha_coeff = 1e-4

            loss = -(q[q.shape[0]//2:]).mean()+alpha_coeff*(q[:q.shape[0]//2]).mean()+0.5*((q[:q.shape[0]//2]-q_target[:q.shape[0]//2])**2).mean()
            loss_dict['critic_loss'] = loss

            return loss.mean(), loss_dict
        
        critic_loss = stable_loss
        if double:
            loss1, dict1 = critic_loss(q1, target_q, v, loss_temp)
            loss2, dict2 = critic_loss(q2, target_q, v, loss_temp)

            critic_loss = (loss1 + loss2).mean()
            for k, v in dict2.items():
                dict1[k] += v
            loss_dict = dict1
        else:
            critic_loss, loss_dict = critic_loss(q1, target_q,  v, loss_temp)

        if args.grad_pen:
            # print("Using grad_pen")
            lambda_ =args.lambda_gp
            q1_grad, q2_grad = grad_norm(critic, critic_params, batch.observations, acts)
            loss_dict['q1_grad'] = q1_grad.mean()
            loss_dict['q2_grad'] = q2_grad.mean()

            if double:
                gp_loss = (q1_grad + q2_grad).mean()
            else:
                gp_loss = q1_grad.mean()

            critic_loss += lambda_ * gp_loss
        if goal_transition_indicator is not None:
            loss_dict.update({
                'unseen_q_expert':(q1*goal_transition_indicator).mean(),
                'unseen_q_suboptimal':(q1*(1-goal_transition_indicator)).mean(),
                'q1': q1.mean(),
                'q2': q2.mean()
            })
        else:
            loss_dict.update({
                'unseen_q_expert':(q1[:q1.shape[0]//2]*is_expert_mask).sum()/is_expert_mask.sum(),
                'unseen_q_suboptimal':(q1[:q1.shape[0]//2]*(1-is_expert_mask)).sum()/(1-is_expert_mask).sum(),
                'q1': q1.mean(),
                'q2': q2.mean()
            })
        return critic_loss, loss_dict

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def update_q_smore(critic: Model, target_value: Model, batch: Batch, is_expert_mask,
             discount: float, double: bool, key: PRNGKey, loss_temp: float, args, goal_transition_indicator=None) -> Tuple[Model, InfoDict]:
    
    next_v = target_value(batch.next_observations)
    neg_r = -2

    target_q = discount  * next_v
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        acts = batch.actions
        
        q1, q2 = critic.apply({'params': critic_params}, batch.observations, acts)
        v = target_value(batch.observations)

        def stable_loss(q,q_target,*args):
            loss_dict = {}
            alpha_coeff = (1-discount)*1e-4
            loss = ((q[q.shape[0]//2:]-neg_r/(1-discount))**2).mean()+alpha_coeff*((q[:q.shape[0]//2]-0)**2).mean()+0.5*((q[:q.shape[0]//2]-q_target[:q.shape[0]//2])**2).mean()
            loss_dict['critic_loss'] = loss

            return loss.mean(), loss_dict

        def original_loss(q,q_target,*args):
            loss_dict = {}
            alpha_coeff = 1e-4

            loss = -(q[q.shape[0]//2:]).mean()+alpha_coeff*(q[:q.shape[0]//2]).mean()+0.5*((q[:q.shape[0]//2]-q_target[:q.shape[0]//2])**2).mean()
            loss_dict['critic_loss'] = loss

            return loss.mean(), loss_dict

        critic_loss = original_loss
        if double:
            loss1, dict1 = critic_loss(q1, target_q, v, loss_temp)
            loss2, dict2 = critic_loss(q2, target_q, v, loss_temp)

            critic_loss = (loss1 + loss2).mean()
            for k, v in dict2.items():
                dict1[k] += v
            loss_dict = dict1
        else:
            critic_loss, loss_dict = critic_loss(q1, target_q,  v, loss_temp)

        if args.grad_pen:
            # print("Using grad_pen")
            lambda_ =args.lambda_gp
            q1_grad, q2_grad = grad_norm(critic, critic_params, batch.observations, acts)
            loss_dict['q1_grad'] = q1_grad.mean()
            loss_dict['q2_grad'] = q2_grad.mean()

            if double:
                gp_loss = (q1_grad + q2_grad).mean()
            else:
                gp_loss = q1_grad.mean()

            critic_loss += lambda_ * gp_loss
        if goal_transition_indicator is not None:
            loss_dict.update({
                'unseen_q_expert':(q1*goal_transition_indicator).mean(),
                'unseen_q_suboptimal':(q1*(1-goal_transition_indicator)).mean(),
                'q1': q1.mean(),
                'q2': q2.mean()
            })
        else:
            loss_dict.update({
                'unseen_q_expert':(q1[:q1.shape[0]//2]*is_expert_mask).sum()/is_expert_mask.sum(),
                'unseen_q_suboptimal':(q1[:q1.shape[0]//2]*(1-is_expert_mask)).sum()/(1-is_expert_mask).sum(),
                'q1': q1.mean(),
                'q2': q2.mean()
            })
        return critic_loss, loss_dict

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info

def huber_loss(x, delta: float = 1.):
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
    See "Robust Estimation of a Location Parameter" by Huber.
    (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
    Args:
    x: a vector of arbitrary shape.
    delta: the bounds for the huber loss transformation, defaults at 1.
    Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
    Returns:
    a vector of same shape of `x`.
    """
    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear
