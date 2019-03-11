import gym
import quanser_robots
from quanser_robots.cartpole import ctrl
from quanser_robots.common import Logger, GentlyTerminating

import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt

import torch
import torch.nn, torch.optim
import torch.utils.data as Data


env = gym.make('Levitation-v0')
env.seed(0)
np.random.seed(2)
torch.manual_seed(0)

print("state range: ", env.observation_space.high, env.observation_space.low)
print("action range: ", env.action_space.high, env.action_space.low)
print("reward range", env.reward_range)

class Policy:
    def __init__(self):
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )   # TODO weigh initialization
        
        self.net = net
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=(0.985))
    
    def choose_action(self, s):
        output = self.net(s).detach_().numpy()
        mu = output[0,0]
        sigma = output[0,1]
        return np.random.normal(size = (1,)) * sigma + mu
    
    def train(self, states, actions, R_whole):  
        states, actions, R_whole = torch.tensor(states, dtype=torch.float),\
            torch.tensor(actions, dtype=torch.float), torch.tensor(R_whole, dtype=torch.float)
        dataset = Data.TensorDataset(states, actions, R_whole)
        loader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=2) 
        
        epoch_loss = 0.
        self.lr_scheduler.step()
        for i, (s, a, R) in enumerate(loader):
            para_distri = self.net(s)
            mu, sigma = para_distri[:, [0]], para_distri[:, [1]]
            log_pi = -torch.log(sigma) - 0.5 * ((a - mu)/sigma) ** 2  
            loss = log_pi.view(1, -1) @ R.view(-1, 1)
            
            epoch_loss += s.shape[0] * loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()    
            self.optimizer.step()      
        
        print("the loss ", epoch_loss)
        return
        
def sample_episode(env, action_policy, gamma):
    lists_a, lists_s, lists_s_ne, lists_r = [], [], [], []
    done = False
    s = env.reset()
    step_counter = 0
    while not done:
        a = action_policy(torch.tensor(s.reshape(1,-1), dtype = torch.float))
        s_ne, r, done, _ = env.step(a)
        lists_a.append(a), lists_s.append(s), lists_s_ne.append(s_ne), lists_r.append(r)
        
        step_counter+=1
        s = s_ne
    
    assert step_counter == 5000 # not sure if the stepsize is fixed of 5000
    
    actions = np.array(lists_a)
    states = np.array(lists_s)
    rewards = np.array(lists_r)    
    gamma_array = np.ones_like(rewards) * gamma
    gamma_array = np.cumproduct(gamma_array)
    R_fun = gamma_array * rewards
    
    print("mean_reward in this sampling ", np.sum(rewards)/step_counter)
    return states, actions.reshape(-1,1), R_fun.reshape(-1,1), np.sum(rewards)/step_counter

policy = Policy()
for i in range(10):
    states, actions, R_fun, mean_r = sample_episode(env, policy.choose_action, 0.99)
    
    policy.train(states, actions, R_fun)  
