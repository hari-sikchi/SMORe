import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

class GoalContinuousGrid(gym.Env):
    '''
    Continuous grid world environment with goal.
    '''

    def __init__(self, r=None, size_x=3, size_y=3, T=5,  random_born=True, state_indices=[0,1],
        terminal_states=[], seed=0, add_time=False, **kwargs):
        self.size_x = size_x
        self.size_y = size_y
        self.terminal_states = terminal_states
        # self.r = r
        self.range_x = (-size_x/2, size_x/2)
        self.range_y = (-size_y/2, size_y/2)
        # self.random_act_prob = random_act_prob
        # self.sigma = sigma
        self.add_time = add_time
        # self.prior_reward_weight = prior_reward_weight
        self.mode = 'train'
        self.goal = np.array([1.2,1.2])
        self.T = T
        if not add_time:
            self.observation_space = Box(low=np.array([0,0]),high=np.array([size_x,size_y]),dtype=np.float32)
        else:
            self.observation_space = Box(low=np.array([0,0,1]),high=np.array([size_x,size_y,T]),dtype=np.float32)

        self.action_space = Box(low=np.array([-1,-1]),high=np.array([1,1]),dtype=np.float32)

        self.seed(seed)
        self.action_space.seed(seed)
        self.random_born = random_born

        # if self.r is not None:
        #     n = 100
        #     x = np.linspace(0, self.size_x, n)
        #     y = np.linspace(0, self.size_y, n)
        #     xx, yy = np.meshgrid(x, y)
        #     zz = np.stack([xx.flatten(), yy.flatten()], axis=1)
        #     all_reward = self.r(zz)
        #     self.min_prior_reward, self.max_prior_reward = np.min(all_reward), np.max(all_reward)
        #     self.prior_reward_range = self.max_prior_reward - self.min_prior_reward


    def set_reward_function(self,r):
        self.r = r

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def set_state(self,s):
        self.s = s.copy().reshape(1,-1)

    def reset(self, n=1):
        if self.random_born:
            self.s = np.random.uniform((self.range_x[0],self.range_y[0]),(self.range_x[1],self.range_y[1]),size=(n, 2))
        else:
            choice = np.random.randint(0,4)
            if choice == 0:
                self.s = np.array([1.23,1.23])
            elif choice==1:
                self.s = np.array([1.23,-1.23])
            elif choice==2:
                self.s = np.array([-1.23,1.23])
            elif choice==3:
                self.s = np.array([-1.23,-1.23])
            # self.s = np.zeros((n, 2), dtype=np.float32)

        self.n = n
        self.t = 0
        if not self.add_time:
            return self.s.copy()
        else:
            ret = np.zeros((n, 3))
            ret[:, :2] = self.s.copy()
            ret[:, 2] = self.t
            return ret

    def step(self, action):
        # change_action_prob = (np.random.uniform(0, 1, size=(self.n)) < self.random_act_prob).reshape(-1,1)
        # action = change_action_prob * (action + self.sigma * np.random.randn(self.n, 2)) \
        #         + (1-change_action_prob) * action

        old_s = self.s.copy()
        # print("Old s: ", old_s)
        self.s += action   
        # print("New s: ", self.s)
        self.s[:,0] = np.clip(self.s[:,0],self.range_x[0],self.range_x[1])
        self.s[:,1] = np.clip(self.s[:,1],self.range_y[0],self.range_y[1])
        # print("Clipped s: ", self.s)
        new_s = self.s.copy()

        self.t += 1
        done = (self.t >= self.T) 

        reward = 0
        # reward = np.logical_and(self.s[:, 0] > self.size_x - 0.05, self.s[:, 1] > self.size_y - 0.05).astype(np.float32)
        # reward_distractor = np.logical_and(self.s[:, 0] > self.size_x - 0.05, self.s[:, 1] < 0.05).astype(np.float32) * 0.1
        # reward_distractor2 = np.logical_and(self.s[:, 1] > self.size_y - 0.05, self.s[:, 0] < 0.05).astype(np.float32) * 0.1
        # reward += reward_distractor
        # reward += reward_distractor2

        

        if not self.add_time:
            return self.s.copy(), reward, done, {}
        else:
            ret = np.zeros((self.s.shape[0], 3))
            ret[:, :2] = self.s.copy()
            ret[:, 2] = self.t
            return ret, reward, done, {}

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'
