from typing import Dict
import os
import flax.linen as nn
import gym
import numpy as np

def discounted_return(rewards, gamma, reward_offset=True):
    N, T = rewards.shape[0], rewards.shape[1]
    if reward_offset:
        rewards += 1   # positive offset as used in previous works.

    discount_weights = np.power(gamma, np.arange(T)).reshape(1, T)
    dis_return = (rewards * discount_weights).sum(axis=1)
    undis_return = rewards.sum(axis=1)
    return dis_return, undis_return

def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int, normalization_dict, max_episode_steps=50, verbose: bool = False, make_gif=False) -> Dict[str, float]:
    total_obs, total_g, total_ag, total_rewards, total_success_rate = [], [], [], [], []

    for i in range(num_episodes):
        per_obs, per_g, per_ag, per_rewards, per_success_rate = [], [], [], [], []
        # import ipdb;ipdb.set_trace()
        observation = [env.reset()]
        if 'observations' in normalization_dict:
            obs = (observation[0]['observation'] - normalization_dict['observations']['mean']) / (normalization_dict['observations']['std']+1e-6)
            ag = (observation[0]['achieved_goal'] - normalization_dict['goals']['mean']) / (normalization_dict['goals']['std']+1e-6)
            g = (observation[0]['desired_goal'] - normalization_dict['goals']['mean']) / (normalization_dict['goals']['std']+1e-6)
        else:
            obs = observation[0]['observation']
            ag = observation[0]['achieved_goal']
            g = observation[0]['desired_goal']
        
        imgs = []
        for _ in range(max_episode_steps):
            input_tensor = np.concatenate((obs, g), axis=0)
            actions = agent.sample_actions(input_tensor, temperature=0.0)
            # actions = self._deterministic_action(input_tensor)
            # convert the actions
            actions = actions.squeeze()
            observation_new, reward, _, info = env.step(actions)
            if 'score/success' in info:
                info['is_success'] = float(info['score/success'])
           
            per_obs.append(obs)
            per_g.append(g)
            per_ag.append(ag)
            per_rewards.append(reward)
            per_success_rate.append(info['is_success'])
            if 'observations' in normalization_dict:
                obs = (observation_new['observation'] - normalization_dict['observations']['mean']) / (normalization_dict['observations']['std']+1e-6)
                ag = (observation_new['achieved_goal'] - normalization_dict['goals']['mean']) / (normalization_dict['goals']['std']+1e-6)
                g = (observation_new['desired_goal'] - normalization_dict['goals']['mean']) / (normalization_dict['goals']['std']+1e-6)
            else:
                obs = observation_new['observation']
                ag = observation_new['achieved_goal']
                g = observation_new['desired_goal']
        total_obs.append(per_obs)
        total_g.append(per_g)
        total_ag.append(per_ag)
        total_rewards.append(per_rewards)
        total_success_rate.append(per_success_rate)

    total_obs = np.array(total_obs)
    total_g = np.array(total_g)
    total_ag = np.array(total_ag)
    total_rewards = np.array(total_rewards)
    total_success_rate = np.array(total_success_rate)
    dis_return, undis_return = discounted_return(total_rewards, 0.99)

    local_discounted_return = np.mean(dis_return)
    local_undiscounted_return = np.mean(undis_return)
    local_distances = np.mean(np.linalg.norm(total_ag[:, -1] - total_g[:, -1], axis=1))
    local_success_rate = np.mean(total_success_rate[:, -1])
    print("Finished evaluation")
    results = {'Test/final_distance': local_distances, 
                'Test/success_rate': local_success_rate,
                'Test/discounted_return': local_discounted_return,
                'Test/undiscounted_return': local_undiscounted_return}
    return results



def evaluate_scalar_env(agent: nn.Module, env: gym.Env,
             num_episodes: int, normalization_dict, max_episode_steps=50, verbose: bool = False, make_gif=False) -> Dict[str, float]:
    total_obs, total_g, total_ag, total_rewards, total_success_rate = [], [], [], [], []

    for i in range(num_episodes):
        per_obs, per_g, per_ag, per_rewards, per_success_rate = [], [], [], [], []
        
        observation =env.reset()
        if 'observations' in normalization_dict:
            obs = (observation['observation'] - normalization_dict['observations']['mean']) / (normalization_dict['observations']['std']+1e-6)
            ag = (observation['achieved_goal'] - normalization_dict['goals']['mean']) / (normalization_dict['goals']['std']+1e-6)
            g = (observation['desired_goal'] - normalization_dict['goals']['mean']) / (normalization_dict['goals']['std']+1e-6)
        else:
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
        
        imgs = []
        for _ in range(max_episode_steps):
            input_tensor = np.concatenate((obs, g), axis=0)
            actions = agent.sample_actions(input_tensor, temperature=0.0)
            # actions = self._deterministic_action(input_tensor)
            # convert the actions
            actions = actions.squeeze()
            observation_new, reward, _, info = env.step(actions)
            if 'score/success' in info:
                info['is_success'] = float(info['score/success'])
            if make_gif:
                img = env.render("rgb_array")
                imgs.append(img)
            per_obs.append(obs)
            per_g.append(g)
            per_ag.append(ag)
            per_rewards.append(reward)
            per_success_rate.append(info['is_success'])
            if 'observations' in normalization_dict:
                obs = (observation_new['observation'] - normalization_dict['observations']['mean']) / (normalization_dict['observations']['std']+1e-6)
                ag = (observation_new['achieved_goal'] - normalization_dict['goals']['mean']) / (normalization_dict['goals']['std']+1e-6)
                g = (observation_new['desired_goal'] - normalization_dict['goals']['mean']) / (normalization_dict['goals']['std']+1e-6)
            else:
                obs = observation_new['observation']
                ag = observation_new['achieved_goal']
                g = observation_new['desired_goal']
        total_obs.append(per_obs)
        total_g.append(per_g)
        total_ag.append(per_ag)
        total_rewards.append(per_rewards)
        total_success_rate.append(per_success_rate)

    total_obs = np.array(total_obs)
    total_g = np.array(total_g)
    total_ag = np.array(total_ag)
    total_rewards = np.array(total_rewards)
    total_success_rate = np.array(total_success_rate)
    dis_return, undis_return = discounted_return(total_rewards, 0.99)

    local_discounted_return = np.mean(dis_return)
    local_undiscounted_return = np.mean(undis_return)
    local_distances = np.mean(np.linalg.norm(total_ag[:, -1] - total_g[:, -1], axis=1))
    local_success_rate = np.mean(total_success_rate[:, -1])
    results = {'Test/final_distance': local_distances, 
                'Test/success_rate': local_success_rate,
                'Test/discounted_return': local_discounted_return,
                'Test/undiscounted_return': local_undiscounted_return}
    return results



def evaluate_mujoco_env(agent: nn.Module, env: gym.Env,
             num_episodes: int, verbose: bool = False, goal_indices=None, desired_goal=None) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    dis_returns = []
    undis_returns = []
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        observation = observation #[0]
        # g = observation[-3:]*0
        # g = np.array([11.0])
        g = desired_goal
        t=0
        ep_ret = 0
        rewards = []
        while not done and t<=1000:
            input_tensor = np.concatenate((observation, g), axis=0)
            action = agent.sample_actions(input_tensor, temperature=0.0)
            observation, reward, done, info = env.env.step(action)
            sparse_reward = int(np.linalg.norm(observation[goal_indices]-desired_goal)<0.5)
            rewards.append(sparse_reward)
            t+=1
            ep_ret+=reward
        dis_return, undis_return = discounted_return(np.array(rewards).reshape(1,-1), 0.999,reward_offset=False)
        dis_returns.append(dis_return)
        undis_returns.append(undis_return)


    stats['discounted_return'] = np.array(dis_returns).mean()
    stats['undiscounted_return'] = np.array(undis_returns).mean()
    return stats