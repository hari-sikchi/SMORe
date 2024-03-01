import collections
from typing import Optional
import gym
import numpy as np
from tqdm import tqdm
import copy
import h5py
from envs.multi_world_wrapper import *
from typing import Tuple
import os
import d4rl


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

MixedBatch = collections.namedtuple(
    'MixedBatch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations','is_expert'])


GCRLBatch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'next_observations', 'achieved_goals', 'goals', 'rewards'])


GCRLMixedBatch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'next_observations', 'achieved_goals', 'goals','is_expert','rewards'])

GCRLMixedBatchFB = collections.namedtuple(
    'Batch',
    ['observations','observations_pure_goals', 'actions', 'next_observations', 'achieved_goals', 'goals','is_expert','rewards'])



def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def convert_to_full_trajectories_with_absorbing_state(dataset, max_len):
    max_size = 2000000
    observations = [[]]
    actions = [[]]
    rewards = [[]]
    terminals = [[]]
    next_observations = [[]]
    for i in range(dataset['observations'].shape[0]):
        observations[-1].append(dataset['observations'][i])
        actions[-1].append(dataset['actions'][i])
        rewards[-1].append(dataset['rewards'][i])
        terminals[-1].append(dataset['terminals'][i])
        next_observations[-1].append(dataset['next_observations'][i])
        if dataset['terminals'][i]==1.0 or len(observations[-1])==max_len:
            if len(observations[-1])!=max_len:
                for j in range(max_len-len(observations[-1])):
                    observations[-1].append(dataset['observations'][i])
                    actions[-1].append(dataset['actions'][i])
                    rewards[-1].append(dataset['rewards'][i])
                    terminals[-1].append(dataset['terminals'][i])
                    next_observations[-1].append(dataset['next_observations'][i])
            observations.append([])
            actions.append([])
            rewards.append([])
            terminals.append([])
            next_observations.append([])
    observations = np.vstack(observations[:-1]).reshape(-1, dataset['observations'].shape[1])
    actions = np.vstack(actions[:-1]).reshape(-1, dataset['actions'].shape[1])
    rewards = np.vstack(rewards[:-1]).reshape(-1)
    terminals = np.vstack(terminals[:-1]).reshape(-1)
    next_observations = np.vstack(next_observations[:-1]).reshape(-1, dataset['next_observations'].shape[1])
    full_traj_dataset = {'observations':observations[:max_size],'actions':actions[:max_size],'rewards':rewards[:max_size],'terminals':terminals[:max_size],'next_observations':next_observations[:max_size]}
    return full_traj_dataset

def split_into_full_trajectories(o,a,u,g, terminals, max_len):
    trajs_o = [[]]
    trajs_a = [[]]
    trajs_u = [[]]
    trajs_g = [[]]
    for i in tqdm(range(len(o))):
        trajs_o[-1].append(o[i])
        trajs_a[-1].append(a[i])
        trajs_u[-1].append(u[i])
        trajs_g[-1].append(g[i])
        if terminals[i] == 1.0 and i + 1 < len(o) or len(trajs_o[-1])==max_len:
            if len(trajs_o[-1])!=max_len:
                trajs_o.pop()
                trajs_g.pop()
                trajs_u.pop()
                trajs_a.pop()

            trajs_o.append([])
            trajs_a.append([])
            trajs_u.append([])
            trajs_g.append([])

    if len(trajs_o[-1])!=max_len:
        trajs_o.pop()
        trajs_g.pop()
        trajs_u.pop()
        trajs_a.pop()

    return trajs_o,trajs_a,trajs_u,trajs_g

def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)




class GCRLDataset(object):
    def __init__(self, observations: np.ndarray,next_observations: np.ndarray, actions: np.ndarray,
                 goals: np.ndarray, achieved_goals: np.ndarray,
                 size: int):
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.goals = goals
        self.achieved_goals = achieved_goals
        self.obs_goals_cat = np.concatenate([observations, goals], axis=1)
        self.next_obs_goals_cat = np.concatenate([next_observations, goals], axis=1)
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        
        return  GCRLBatch(observations=self.obs_goals_cat[indx],
                     actions=self.actions[indx],
                     next_observations = self.next_obs_goals_cat[indx],
                     achieved_goals=self.achieved_goals[indx],
                     goals=self.goals[indx])



class GCRLDataset(object):
    def __init__(self, observations: np.ndarray,next_observations: np.ndarray, actions: np.ndarray,
                 goals: np.ndarray, achieved_goals: np.ndarray,
                 size: int):
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.goals = goals
        self.achieved_goals = achieved_goals
        self.obs_goals_cat = np.concatenate([observations, goals], axis=1)
        self.next_obs_goals_cat = np.concatenate([next_observations, goals], axis=1)
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        
        return  GCRLBatch(observations=self.obs_goals_cat[indx],
                     actions=self.actions[indx],
                     next_observations = self.next_obs_goals_cat[indx],
                     achieved_goals=self.achieved_goals[indx],
                     goals=self.goals[indx])


class GCRLMixedDataset(object):
    def __init__(self, observations: np.ndarray,next_observations: np.ndarray, actions: np.ndarray,
                 goals: np.ndarray, achieved_goals: np.ndarray,next_achieved_goals: np.ndarray, is_expert: np.ndarray,
                 size: int, distance_threshold: int, max_steps:int, normalization_info, rewards=None):
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.goals = goals
        self.achieved_goals = achieved_goals
        self.next_achieved_goals=next_achieved_goals
        self.obs_goals_cat = np.concatenate([observations, goals], axis=1)
        self.next_obs_goals_cat = np.concatenate([next_observations, goals], axis=1)
        if rewards is None:
            self.rewards = (np.linalg.norm(self.next_achieved_goals - self.goals,axis=-1) < distance_threshold).astype(float)
        else:
            self.rewards = rewards
        self.is_expert = is_expert

        self.observation_trajectories = observations.reshape(-1, max_steps, observations.shape[1])
        self.next_observation_trajectories = next_observations.reshape(-1, max_steps, observations.shape[1])
        self.normalization_info = normalization_info
        
        if rewards is not None:
            self.goal_reached_ids= np.where(rewards>=1)[0]
        else:
            if 'goals' in self.normalization_info:
                self.goal_reached_ids = np.where(np.linalg.norm(self.achieved_goals*self.normalization_info['goals']['std'] - self.goals*self.normalization_info['goals']['std'],axis=-1) < distance_threshold)[0]
            else:
                self.goal_reached_ids = np.where(np.linalg.norm(self.achieved_goals - self.goals,axis=-1) < distance_threshold)[0]
        self.goal_traj_ids = self.goal_reached_ids // (max_steps-1)
        self.max_steps = max_steps
        self.size = size
        self.p_future = 0.8

    def sample(self, batch_size: int, sampling_mode='HER') -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        if sampling_mode == 'HER_goal_entry_distribution':
            # test goal distribution entry
            indx = np.random.randint(len(self.goal_reached_ids), size=batch_size)
            goals_idx  = self.goal_reached_ids[indx]
            traj_idx = np.random.randint(self.observation_trajectories.shape[0], size=batch_size)
            time_idx = np.random.randint(self.observation_trajectories.shape[1]-1, size=batch_size)
            indx = traj_idx*(self.max_steps-1)+time_idx
            uniform_sample = np.random.uniform(0,1,batch_size)
            future = uniform_sample < self.p_future
            future_time_idx = ((self.max_steps-3-time_idx)* np.random.uniform(0,1,batch_size)).astype(int)+1+time_idx
            future_goal_idx = traj_idx*(self.max_steps-1)+future_time_idx
            combined_idx = future.reshape(-1)*future_goal_idx + (1-future).reshape(-1)*goals_idx
            combined_goals = future.reshape(-1,1)*self.achieved_goals[future_goal_idx] + (1-future).reshape(-1,1)*self.goals[goals_idx]
            return  GCRLMixedBatch(observations=np.concatenate((self.observations[combined_idx-1],combined_goals),axis=1),
                     actions=self.actions[combined_idx-1],
                     next_observations = np.concatenate((self.next_observations[combined_idx-1],combined_goals),axis=1),
                     achieved_goals=self.achieved_goals[combined_idx-1],
                     goals=combined_goals,
                     is_expert=self.is_expert[combined_idx-1],
                     rewards = self.rewards[combined_idx-1])
        elif sampling_mode == 'HER':
            traj_idx = np.random.randint(self.observation_trajectories.shape[0], size=batch_size)
            time_idx = np.random.randint(self.observation_trajectories.shape[1]-1, size=batch_size)
            indx = traj_idx*(self.max_steps-1)+time_idx
            uniform_sample = np.random.uniform(0,1,batch_size)
            future = uniform_sample < self.p_future
            future_goal_idx = ((self.max_steps-3-time_idx)* np.random.uniform(0,1,batch_size)).astype(int)+1+time_idx
            goal_idx = traj_idx*(self.max_steps-1)+future_goal_idx
            goals = future.reshape(-1,1)*self.achieved_goals[goal_idx] + (1-future).reshape(-1,1)*self.goals[indx]
            return  GCRLMixedBatch(observations=np.concatenate((self.observations[indx],goals),axis=1),
                     actions=self.actions[indx],
                     next_observations = np.concatenate((self.next_observations[indx],goals),axis=1),
                     achieved_goals=self.achieved_goals[indx],
                     goals=goals,
                     is_expert=self.is_expert[indx],
                     rewards = self.rewards[indx])

        else:
            raise NotImplementedError
        

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def get_full_envname(name):
    dic = {
        'PointReach': 'Point2DLargeEnv-v1',
        'PointRooms': 'Point2D-FourRoom-v1',
        'Reacher':'Reacher-v2',
        'SawyerReach': 'SawyerReachXYZEnv-v1',
        'SawyerDoor': 'SawyerDoor-v0',
        'FetchReach':'FetchReach-v1',
        'FetchPush': 'FetchPush-v1',
        'FetchSlide': 'FetchSlide-v1',
        'FetchPick': 'FetchPickAndPlace-v1',
        'HandReach':'HandReach-v0',
        'DClawTurn': 'DClawTurn-v0',
    }
    if name in dic.keys():
        return dic[name]
    else:
        return name
    
def preproc_o( o):
        clip_obs = 200
        o = np.clip(o, -clip_obs, clip_obs)
        return o

def make_env_and_dataset(env_name: str,
                         seed: int, flags) -> Tuple[gym.Env, GCRLDataset]:

    env = gym.make(get_full_envname(env_name))
    if 'halfcheetah' in env_name  or 'ant' in env_name or 'hopper' in env_name or 'walker' in env_name or 'offline' in env_name:
        pass
    else:
        default_max_episode_steps = 50
        if env_name.startswith('Fetch'):
            env._max_episode_steps = 50
            env = FetchGoalWrapper(env)
        elif env_name.startswith('HandManipulate'):
            env._max_episode_steps = 100
        elif env_name.startswith('Point'):
            env = PointGoalWrapper(env)
            env.env._max_episode_steps = 50
        elif env_name.startswith('Sawyer'): 
            env = SawyerGoalWrapper(env)
        elif env_name.startswith('Reacher'):
            env = ReacherGoalWrapper(env)
        if hasattr(env, '_max_episode_steps'):
            max_episode_steps = env._max_episode_steps
        else:
            max_episode_steps = default_max_episode_steps # otherwise use defaulit max episode steps
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if flags.noisy_env:
        env = NoisyAction(env, noise_eps=flags.noise_eps)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    goal_indices = None
    desired_goal = None

    if 'halfcheetah' in env_name or 'ant' in env_name or 'hopper' in env_name or 'walker' in env_name and 'antmaze' not in env_name:
        expert_env = gym.make(env_name.split('-')[0]+'-expert-v2')
        expert_dataset = d4rl.qlearning_dataset(expert_env)
        suboptimal_dataset = d4rl.qlearning_dataset(env)
        
        expert_dataset=convert_to_full_trajectories_with_absorbing_state(expert_dataset,env._max_episode_steps)
        print("Suboptimal dataset size before: {}".format(suboptimal_dataset['observations'].shape[0]))
        suboptimal_dataset = convert_to_full_trajectories_with_absorbing_state(suboptimal_dataset,env._max_episode_steps)
        print("Suboptimal dataset size after: {}".format(suboptimal_dataset['observations'].shape[0]))
        dataset = {}
        
        episode_len = 1000
        expert_trajs = 30
        expert_transitions = episode_len * expert_trajs
        
        dataset['observations'] = np.concatenate([expert_dataset['observations'][:expert_transitions],suboptimal_dataset['observations']],axis=0)
        dataset['actions'] = np.concatenate([expert_dataset['actions'][:expert_transitions],suboptimal_dataset['actions']],axis=0)
        dataset['next_observations'] = np.concatenate([expert_dataset['next_observations'][:expert_transitions],suboptimal_dataset['next_observations']],axis=0)
        dataset['rewards'] = np.concatenate([expert_dataset['rewards'][:expert_transitions],suboptimal_dataset['rewards']],axis=0)
        if 'halfcheetah' in env_name:
            suboptimal_dataset_achieved_goals = suboptimal_dataset['observations'][:,8:9]
            suboptimal_dataset_goals = suboptimal_dataset['observations'][:,8:9]*0+11
            distance_threshold = 0.5
            expert_dataset_achieved_goals = expert_dataset['observations'][:,8:9]
            expert_dataset_goals = expert_dataset['observations'][:,8:9]*0+11
            goal_indices = 8
            desired_goal = np.array([11.0])
            dataset['achieved_goals'] = np.concatenate([expert_dataset['observations'][:expert_transitions,8:9],suboptimal_dataset['observations'][:,8:9]],axis=0)
            dataset['goals'] = np.concatenate([expert_dataset['observations'][:expert_transitions,8:9],suboptimal_dataset['observations'][:,8:9]],axis=0)*0
            dataset['next_achieved_goals'] = dataset['achieved_goals'] # TODO: Unused, but change this later if needed 
        elif 'ant' in env_name:

            suboptimal_dataset_achieved_goals = suboptimal_dataset['observations'][:,13:14]
            suboptimal_dataset_goals = suboptimal_dataset['observations'][:,13:14]*0+5.0
            distance_threshold = 0.5
            goal_indices = 13
            desired_goal = np.array([5.0])
            expert_dataset_achieved_goals = expert_dataset['observations'][:,13:14]
            expert_dataset_goals = expert_dataset['observations'][:,13:14]*0+5.0

            dataset['achieved_goals'] = np.concatenate([expert_dataset['observations'][:expert_transitions,13:14],suboptimal_dataset['observations'][:,13:14]],axis=0)
            dataset['goals'] = np.concatenate([expert_dataset['observations'][:expert_transitions,13:14],suboptimal_dataset['observations'][:,13:14]],axis=0)*0
            dataset['next_achieved_goals'] = dataset['achieved_goals'] # TODO: Unused, but change this later if needed 
        dataset['is_expert'] = np.concatenate((np.ones(expert_dataset['actions'].shape[0]),np.zeros(suboptimal_dataset['actions'].shape[0])),axis=0)
        traj_o, traj_a, traj_u, traj_g = split_into_full_trajectories(suboptimal_dataset['observations'],suboptimal_dataset_achieved_goals,suboptimal_dataset['actions'], suboptimal_dataset_goals, suboptimal_dataset['terminals'], 1000)
        save_suboptimal_dataset = {'o':traj_o,'ag':traj_a ,'g':traj_g ,'u':traj_u}
        traj_o, traj_a, traj_u, traj_g = split_into_full_trajectories(expert_dataset['observations'],expert_dataset_achieved_goals,expert_dataset['actions'], expert_dataset_goals, expert_dataset['terminals'],1000)
        save_expert_dataset = {'o':traj_o,'ag':traj_a ,'g':traj_g ,'u':traj_u}

        if not os.path.isfile('/data/harshit_sikchi/work/GoFAR/offline_data/expert/'+env_name+'/buffer.pkl'):
            import pickle
            os.makedirs('/data/harshit_sikchi/work/GoFAR/offline_data/expert/'+env_name)
            os.makedirs('/data/harshit_sikchi/work/GoFAR/offline_data/random/'+env_name)
            with open('/data/harshit_sikchi/work/GoFAR/offline_data/expert/'+env_name+'/buffer.pkl', 'wb') as handle:
                pickle.dump(save_expert_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('/data/harshit_sikchi/work/GoFAR/offline_data/random/'+env_name+'/buffer.pkl', 'wb') as handle:
                pickle.dump(save_suboptimal_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        dataset_obj = GCRLMixedDataset(observations=dataset['observations'],next_observations=dataset['next_observations'],actions = dataset['actions'],goals=dataset['goals'], achieved_goals=dataset['achieved_goals'],next_achieved_goals=dataset['next_achieved_goals'],is_expert=dataset['is_expert'],size=dataset['achieved_goals'].shape[0], distance_threshold=0.5,max_steps=1000,normalization_info={})
        expert_dataset_obj  = None
        normalization_dict = {}
    else:
        if 'FetchReach' in env_name:
            load_path_expert = '/data/harshit_sikchi/work/GoFAR/offline_data/random/FetchReach/'
            load_path_random = '/data/harshit_sikchi/work/GoFAR/offline_data/random/FetchReach/'
        elif 'Hand' in env_name or 'Pick' in  env_name or 'Slide' in env_name or 'Push' in env_name:
            load_path_expert = f'/data/harshit_sikchi/work/gofarther_dev/offline_data_wgcsl/hard_tasks_2e6/expert/{env_name}/'
            load_path_random = f'/data/harshit_sikchi/work/gofarther_dev/offline_data_wgcsl/hard_tasks_2e6/random/{env_name}/'
        else:
            load_path_expert = f'/data/harshit_sikchi/work/gofarther_dev/offline_data_wgcsl/expert/{env_name}/'
            load_path_random = f'/data/harshit_sikchi/work/gofarther_dev/offline_data_wgcsl/random/{env_name}/'

        if flags.noisy_env:
            buffer_name = 'buffer-noise{}'.format(flags.noise_eps)
        else:
            buffer_name = 'buffer'
        random_percent = 1.0-flags.expert_fraction
        expert_percent = flags.expert_fraction
        expert_filepath=os.path.join(load_path_expert, f'{buffer_name}.pkl')
        random_filepath = os.path.join(load_path_random, f'{buffer_name}.pkl')
        key_map = {'o': 'observations', 'ag': 'achieved_goals', 'g': 'goals', 'u':'actions'}
        import pickle
        expert_dataset = {}
        random_dataset = {}
        dataset = {}
        with open(expert_filepath, "rb") as fp_expert:  
            with open(random_filepath, "rb") as fp_random:  
                data_expert = pickle.load(fp_expert)  
                data_random = pickle.load(fp_random)  
                size_expert = data_expert['o'].shape[0]
                size_random = data_random['o'].shape[0]
                current_size = int(size_expert*expert_percent + size_random*random_percent)
                size = current_size
                split_point = int(size_expert*expert_percent)
                normalization_dict = {} 
                for key in data_expert.keys():
                    if key=='o':
                        expert_dataset[key_map[key]] =  data_expert[key][:split_point][:,:-1,:].reshape(-1,data_expert[key].shape[2])   
                        random_dataset[key_map[key]] =  data_random[key][:size - split_point][:,:-1,:].reshape(-1,data_random[key].shape[2])
                        expert_dataset['next_observations'] =  data_expert[key][:split_point][:,1:,:].reshape(-1,data_expert[key].shape[2])   
                        random_dataset['next_observations'] =  data_random[key][:size - split_point][:,1:,:].reshape(-1,data_random[key].shape[2])
                        
                        
                        dataset[key_map[key]] = np.concatenate((expert_dataset[key_map[key]],random_dataset[key_map[key]]),axis=0)
                        dataset['next_observations'] = np.concatenate((expert_dataset['next_observations'],random_dataset['next_observations']),axis=0)

                        dataset[key_map[key]] = preproc_o(dataset[key_map[key]])
                        dataset['next_observations'] = preproc_o(dataset['next_observations'])

                        if flags.normalize_observations:
                            # Calculating the mean and std of the observations
                            normalization_dict[key_map[key]] = {'mean': np.mean(dataset[key_map[key]],axis=0), 'std': np.std(dataset[key_map[key]],axis=0)}
                            normalization_dict[key_map[key]]['std'][normalization_dict[key_map[key]]['std']<1e-2]=1e-2
                            # Normalizing the observations from mixed dataset
                            dataset[key_map[key]] = (dataset[key_map[key]] - normalization_dict[key_map[key]]['mean'])/(normalization_dict[key_map[key]]['std'])
                            dataset['next_observations'] = (dataset['next_observations'] - normalization_dict[key_map[key]]['mean'])/(normalization_dict[key_map[key]]['std'])

                            # Normalizing the next observation from expert dataset
                            expert_dataset[key_map[key]] = (expert_dataset[key_map[key]] - normalization_dict[key_map[key]]['mean'])/(normalization_dict[key_map[key]]['std'])
                            expert_dataset['next_observations'] = (expert_dataset['next_observations'] - normalization_dict[key_map[key]]['mean'])/(normalization_dict[key_map[key]]['std'])

                
                    else:
                        expert_dataset[key_map[key]] =  data_expert[key][:split_point][:,:,:].reshape(-1,data_expert[key].shape[2])   
                        random_dataset[key_map[key]] =  data_random[key][:size - split_point][:,:,:].reshape(-1,data_random[key].shape[2])
                        dataset[key_map[key]] = np.concatenate((expert_dataset[key_map[key]],random_dataset[key_map[key]]),axis=0)
                        dataset[key_map[key]] = preproc_o(dataset[key_map[key]])
                        
                        if flags.normalize_observations:
                            if key == 'g':
                                # Calculating the mean and std of the actions
                                normalization_dict[key_map[key]] = {'mean': np.mean(dataset[key_map[key]],axis=0), 'std': np.std(dataset[key_map[key]],axis=0)}
                                normalization_dict[key_map[key]]['std'][normalization_dict[key_map[key]]['std']<1e-2]=1e-2
                                # Normalizing the actions
                                dataset[key_map[key]] = (dataset[key_map[key]] - normalization_dict[key_map[key]]['mean'])/(normalization_dict[key_map[key]]['std']+1e-6)
                
                for key in data_expert.keys():
                    if key == 'ag':
                        expert_dataset[key_map[key]] =  data_expert[key][:split_point][:,:-1,:].reshape(-1,data_expert[key].shape[2])   
                        random_dataset[key_map[key]] =  data_random[key][:size - split_point][:,:-1,:].reshape(-1,data_random[key].shape[2])
                        dataset[key_map[key]] = np.concatenate((expert_dataset[key_map[key]],random_dataset[key_map[key]]),axis=0)
                        dataset[key_map[key]] = preproc_o(dataset[key_map[key]])
                        
                        expert_dataset['next_achieved_goals'] =  data_expert[key][:split_point][:,1:,:].reshape(-1,data_expert[key].shape[2])   
                        random_dataset['next_achieved_goals'] =  data_random[key][:size - split_point][:,1:,:].reshape(-1,data_random[key].shape[2])
                        dataset['next_achieved_goals'] = np.concatenate((expert_dataset['next_achieved_goals'],random_dataset['next_achieved_goals']),axis=0)
                        dataset['next_achieved_goals'] = preproc_o(dataset['next_achieved_goals'])
                        # # Calculating the mean and std of the achieved goals
                        if flags.normalize_observations:
                        # Normalizing the achieved goals
                            dataset[key_map[key]] = (dataset[key_map[key]] - normalization_dict[key_map['g']]['mean'])/(normalization_dict[key_map['g']]['std']+1e-6)

        
        dataset['is_expert'] = np.concatenate((np.ones(expert_dataset['actions'].shape[0]),np.zeros(random_dataset['actions'].shape[0])),axis=0)
        if hasattr(env, 'threshold'):
            dataset_obj = GCRLMixedDataset(observations=dataset['observations'],next_observations=dataset['next_observations'],actions = dataset['actions'],goals=dataset['goals'], achieved_goals=dataset['achieved_goals'],next_achieved_goals=dataset['next_achieved_goals'],is_expert=dataset['is_expert'],size=dataset['achieved_goals'].shape[0], distance_threshold=env.threshold,max_steps=env._max_episode_steps,normalization_info=normalization_dict)
        elif hasattr(env, 'indicator_threshold'):
            dataset_obj = GCRLMixedDataset(observations=dataset['observations'],next_observations=dataset['next_observations'],actions = dataset['actions'],goals=dataset['goals'], achieved_goals=dataset['achieved_goals'],next_achieved_goals=dataset['next_achieved_goals'],is_expert=dataset['is_expert'],size=dataset['achieved_goals'].shape[0], distance_threshold=env.indicator_threshold,max_steps=env._max_episode_steps,normalization_info=normalization_dict)
        elif hasattr(env, 'target_radius'):
            dataset_obj = GCRLMixedDataset(observations=dataset['observations'],next_observations=dataset['next_observations'],actions = dataset['actions'],goals=dataset['goals'], achieved_goals=dataset['achieved_goals'],next_achieved_goals=dataset['next_achieved_goals'],is_expert=dataset['is_expert'],size=dataset['achieved_goals'].shape[0], distance_threshold=env.target_radius,max_steps=env._max_episode_steps,normalization_info=normalization_dict)
        else:
            dataset_obj = GCRLMixedDataset(observations=dataset['observations'],next_observations=dataset['next_observations'],actions = dataset['actions'],goals=dataset['goals'], achieved_goals=dataset['achieved_goals'],next_achieved_goals=dataset['next_achieved_goals'],is_expert=dataset['is_expert'],size=dataset['achieved_goals'].shape[0], distance_threshold=env.distance_threshold,max_steps=env._max_episode_steps,normalization_info=normalization_dict)
    return env, dataset_obj, normalization_dict, goal_indices, desired_goal

