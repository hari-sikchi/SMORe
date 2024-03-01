import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from jax.config import config
# config.update('jax_disable_jit', True)
import numpy as np
import sys
from absl import app, flags
from ml_collections import config_flags 
from dataclasses import dataclass
import wrappers
from evaluation import evaluate, evaluate_scalar_env, evaluate_mujoco_env
from learner import Learner
from logging_utils.logx import EpochLogger
from dataset_utils import make_env_and_dataset
import os.path

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-medium-v2', 'Environment name.')
flags.DEFINE_string('exp_name', 'dump', 'Epoch logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float('temp', 1.0, 'Loss temperature')
flags.DEFINE_boolean('double', True, 'Use double q-learning')
flags.DEFINE_integer('max_steps', int(3e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('grad_pen', False, 'Add a gradient penalty to critic network')
flags.DEFINE_float('lambda_gp', 1, 'Gradient penalty coefficient')
flags.DEFINE_float('max_clip', 7., 'Loss clip value')
flags.DEFINE_integer('num_v_updates', 1, 'Number of value updates per iter')
flags.DEFINE_float('alpha', 0.8, 'f-maximization strength')
flags.DEFINE_float('beta', 0.1, 'imitation strength vs bellman strength')
flags.DEFINE_float('expert_fraction', 0.1, 'Fraction of expert trajectories in the dataset')
flags.DEFINE_integer('max_episode_steps', 50, 'Max episode steps')
flags.DEFINE_boolean('normalize_observations', False, 'Normalizes observations and goals')
flags.DEFINE_string('loss_type', 'smore_stable', 'Either smore or smore_stable')
flags.DEFINE_boolean('noisy_env', False, 'Add noise to actions')
flags.DEFINE_float('noise_eps', 0.5, 'Noise std for actions')


config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

@dataclass(frozen=True)
class ConfigArgs:
    grad_pen: bool
    lambda_gp: int
    max_clip: float
    num_v_updates: int



def main(_):

    # Set up logging
    if FLAGS.noisy_env:
        exp_id = f"results_smore/expert_{str(FLAGS.expert_fraction)}/gcrl2/{FLAGS.env_name}_noise_{FLAGS.noise_eps}/" + FLAGS.exp_name 
    else:   
        exp_id = f"results_smore/expert_{str(FLAGS.expert_fraction)}/gcrl2_HER_ratio_0.8/{FLAGS.env_name}/" + FLAGS.exp_name
    log_folder = exp_id + '/'+FLAGS.exp_name+'_s'+str(FLAGS.seed) 
    logger_kwargs={'output_dir':log_folder, 'exp_name':FLAGS.exp_name}
    e_logger = EpochLogger(**logger_kwargs)


    # Generate datasets for offline GCRL (Follow protocal similar to https://arxiv.org/abs/2206.03023 by mixing expert and suboptimal trajectories)
    env, dataset, normalization_dict, goal_indices, desired_goal = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, flags.FLAGS)

    kwargs = dict(FLAGS.config)

    args = ConfigArgs(grad_pen=FLAGS.grad_pen,
                      lambda_gp=FLAGS.lambda_gp,
                      max_clip=FLAGS.max_clip,
                      num_v_updates=FLAGS.num_v_updates)
    if 'pen' in FLAGS.env_name or 'door' in FLAGS.env_name or 'halfcheetah' in FLAGS.env_name or 'ant' in FLAGS.env_name or 'hopper' in FLAGS.env_name or 'walker' in FLAGS.env_name :
        agent = Learner(FLAGS.seed,
                        np.concatenate((env.observation_space.sample()[np.newaxis],
                            env.observation_space.sample()[-1:][np.newaxis]),axis=1),
                        env.action_space.sample()[np.newaxis],
                        max_steps=FLAGS.max_steps,
                        loss_temp=FLAGS.temp,
                        double_q=FLAGS.double,
                        vanilla=True,
                        alpha = FLAGS.alpha,
                        beta = FLAGS.beta,
                        loss_type = FLAGS.loss_type,
                        args=args,
                        **kwargs)
    else:
        agent = Learner(FLAGS.seed,
                        np.concatenate((env.observation_space.sample()['observation'][np.newaxis],
                            env.observation_space.sample()['achieved_goal'][np.newaxis]),axis=1),
                        env.action_space.sample()[np.newaxis],
                        max_steps=FLAGS.max_steps,
                        loss_temp=FLAGS.temp,
                        double_q=FLAGS.double,
                        vanilla=True,
                        alpha = FLAGS.alpha,
                        beta = FLAGS.beta,
                        loss_type = FLAGS.loss_type,
                        args=args,
                        **kwargs)

    best_eval_returns = -np.inf
    eval_returns = []
    for i in range(1, FLAGS.max_steps + 1): 
        batch = dataset.sample(int(FLAGS.batch_size),sampling_mode='HER')
        # Sample a goal transition batch
        gt_batch = dataset.sample(int(FLAGS.batch_size), sampling_mode='HER_goal_entry_distribution')
        update_info = agent.update(batch, gt_batch)

        if i % FLAGS.eval_interval == 0:
            if 'halfcheetah' in FLAGS.env_name or 'ant' in FLAGS.env_name:
                eval_stats = evaluate_mujoco_env(agent, env, FLAGS.eval_episodes, goal_indices=goal_indices, desired_goal=desired_goal)
                e_logger.log_tabular('Iterations', i)
                e_logger.log_tabular('Discounted Return', eval_stats['discounted_return'])
                e_logger.log_tabular('Undiscounted Return', eval_stats['undiscounted_return'])
                e_logger.dump_tabular()
            elif 'Hand' in FLAGS.env_name or 'Fetch' in FLAGS.env_name: # or 'Reacher' in FLAGS.env_name:
                eval_stats = evaluate(agent, env, FLAGS.eval_episodes, normalization_dict, max_episode_steps=FLAGS.max_episode_steps, make_gif=False)
                if eval_stats['Test/success_rate'] >= best_eval_returns:
                    # Store best eval returns
                    best_eval_returns = eval_stats['Test/success_rate']
                e_logger.log_tabular('Iterations', i)
                e_logger.log_tabular('Test/final_distance', eval_stats['Test/final_distance'])
                e_logger.log_tabular('Test/success_rate', eval_stats['Test/success_rate'])
                e_logger.log_tabular('Test/discounted_return', eval_stats['Test/discounted_return'])
                e_logger.log_tabular('Test/undiscounted_return', eval_stats['Test/undiscounted_return'])
                e_logger.log_tabular('UnseenExpertV', update_info['unseen_v_expert'].item())
                e_logger.log_tabular('UnseenRandomV', update_info['unseen_v_suboptimal'].item())
                e_logger.log_tabular('UnseenExpertQ', update_info['unseen_q_expert'].item())
                e_logger.log_tabular('UnseenRandomQ', update_info['unseen_q_suboptimal'].item())
                e_logger.log_tabular('ClippedAdv', update_info['clipped_adv'].mean().item())
                e_logger.dump_tabular()
                eval_returns.append((i, eval_stats['Test/success_rate']))
                print("Iterations: {} Average Success Rate: {}".format(i,eval_stats['Test/success_rate']))
            else:
                eval_stats = evaluate_scalar_env(agent, env, FLAGS.eval_episodes, normalization_dict, max_episode_steps=FLAGS.max_episode_steps, make_gif=False)

                if eval_stats['Test/success_rate'] >= best_eval_returns:
                    # Store best eval returns
                    best_eval_returns = eval_stats['Test/success_rate']
                e_logger.log_tabular('Iterations', i)
                e_logger.log_tabular('Test/final_distance', eval_stats['Test/final_distance'])
                e_logger.log_tabular('Test/success_rate', eval_stats['Test/success_rate'])
                e_logger.log_tabular('Test/discounted_return', eval_stats['Test/discounted_return'])
                e_logger.log_tabular('Test/undiscounted_return', eval_stats['Test/undiscounted_return'])
                e_logger.log_tabular('UnseenExpertV', update_info['unseen_v_expert'].item())
                e_logger.log_tabular('UnseenRandomV', update_info['unseen_v_suboptimal'].item())
                e_logger.log_tabular('UnseenExpertQ', update_info['unseen_q_expert'].item())
                e_logger.log_tabular('UnseenRandomQ', update_info['unseen_q_suboptimal'].item())
                e_logger.log_tabular('ClippedAdv', update_info['clipped_adv'].mean().item())
                e_logger.dump_tabular()
                eval_returns.append((i, eval_stats['Test/success_rate']))
                print("Iterations: {} Average Success Rate: {}".format(i,eval_stats['Test/success_rate']))

    sys.exit(0)
    os._exit(0)
    raise SystemExit


if __name__ == '__main__':
    app.run(main)
