import torch
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from ppo.ppo_nn import PPONN
from ppo.ppo_nn import ValueNN

class PPO:

    def __init__(self, env, steps_per_episode):
        self.env = env
        self.number_of_episodes = 1000
        self.steps_per_episode = steps_per_episode
        self.min_vol, self.max_vol = env.action_space.low[0], env.action_space.high[0]
        self.observation_space = env.observation_space.shape[0]
        self.ppo_nn = PPONN(self.observation_space)
        self.value_nn = PPONN(self.observation_space)

    def main(self):
        for episode in range(1, self.number_of_episodes):
            done = False
            episode_rwd = 0
            self.obs = self.env.reset()
            while not done:
                obs, rwd, done, _ = self.env.step(self.get_action(self.obs))

                self.obs = obs

                episode_rwd += rwd
                self.env.render()
            self.value_nn.minimize_loss(old_state=self.obs, new_state=obs, rwd=rwd)
            self.ppo_nn.minimize_loss(old_state=self.obs, new_state=obs, rwd=rwd)
            print('Reward this episode was: ' + str(episode_rwd))


    def get_action(self, obs):
        return (torch.randn(1) + self.ppo_nn(torch.Tensor(obs)).detach()).numpy()


