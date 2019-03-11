import numpy as np
import torch as torch
import matplotlib.pyplot as plt
from copy import copy
from lspi.memory import LSPIMemory
from lspi.policy import Policy
from lspi.lstd import LSTDQ


class LSPI:

    # Initialize lspi
    # D = Source of Samples (s,a,r,s')
    # k = Number of basis functions
    # bas_fun = Basis functions
    # gamma = Discount factor
    # epsilon = stopping criterion
    # w_0 = initial policy
    def __init__(self, env, steps_per_episode):
        self.env = env
        self.obs_space = self.env.observation_space
        self.obs_space_size = self.env.observation_space.shape[0]
        self.loadNoSample = self.env.env.stabilization

        self.stab = True
        if self.stab:
            self.obs_space_size = self.env.observation_space.shape[0]-1

        # For memory initialisation
        self.starting_episodes = 10
        self.starting_steps = steps_per_episode
        self.max_size = 100000
        self.memory = LSPIMemory(self.max_size)

        # define action_space
        self.min_vol = -9.0  # Minimal voltage to apply
        self.max_vol = 9.0  # Maximal voltage to apply
        self.size_action_space = 3
        self.action_space = np.arange(
            self.min_vol, self.max_vol+1,
            (self.max_vol-self.min_vol)/(self.size_action_space-1), dtype=float)  # Discrete action space
        self.action_space = torch.tensor(self.action_space)
        print(self.action_space)

        # Define Basis Function
        self.number_means = 3
        self.rbf_means = self.number_means * self.obs_space_size
        self.gamma = 0.8
        self.lstdq = LSTDQ(self.gamma, self.obs_space_size, self.stab)

        self.initial_policy = Policy(self.action_space, self.obs_space_size, self.number_means, self.obs_space,
                                     self.stab)
        self.changing_policy = copy(self.initial_policy)
        self.epsilon = 0.0001

        # Using the policy
        self.test_episodes = 200
        self.test_timesteps = steps_per_episode
        self.still_learning = False
        self.old_weights = torch.Tensor([])

        # For Plotting and evaluation
        self.rew_overall = 0
        self.rew_episodes_len = 10
        self.rew_episodes = np.zeros(self.rew_episodes_len)
        self.rwd_episodes_array = np.empty(0)
        self.rwd_overall_array = np.empty(0)

    def main(self):
        self.sample_data()

        print('Memory generation finished')
        # Memory is full with training data

        episode = self.lspi_main()

        self.play_with_policy(episode)

        return 0

    def lspi_main(self):
        samples = torch.stack(self.memory.get_all_samples())

        print('#++++++++++++++++++++++++++++#')
        print(samples.shape)
        print('#++++++++++++++++++++++++++++#')

        i = 5
        if self.stab:
            i = 4
        self.get_means(samples[:, 0:i])

        distance = float('inf')
        iteration = 1

        new_weights = self.changing_policy.get_weights()
        distances = np.empty(0)

        # samples = torch.stack(self.memory.get_x_samples(15000))

        while distance > self.epsilon:

            self.changing_policy.set_weights(new_weights)
            new_weights = self.lstdq.learn(samples, self.changing_policy).view(-1, 1)
            print(torch.norm(new_weights))

            print('#+++++++++++++++++++++#')
            print('Lspi_mainn/function')
            print('#+++++++++++++++++++++#')
            old_weights = self.changing_policy.get_weights()

            new_distance = torch.dist(old_weights, new_weights)
            distance = new_distance
            print('Distance: ' + str(distance))

            if iteration != 1:
                distances = np.append(distances, distance)
                self.plot_distance(distances)

            self.test_episodes = iteration + 1
            self.play_with_policy(iteration)

            self.test_episodes = 200

            iteration += 1
            torch.save(self.changing_policy.get_weights(), 'weights.pt')
        print('# +++++++++++++++++++')
        print('Weights found')
        print('# +++++++++++++++++++')
        torch.save(self.changing_policy.get_weights(), 'weights.pt')
        return iteration

    def get_means(self, samples):

        mins, _ = torch.min(samples, dim=0)
        maxs, _ = torch.max(samples, dim=0)
        step_size = (maxs-mins)/(self.number_means+1)

        means_list = []
        no_grid_list = []

        for i in range(0, mins.shape[0]):
            grid = np.linspace(mins[i].item(), maxs[i].item(), num=self.number_means+1)
            no_grid = torch.from_numpy(grid)
            means = []
            for j in range(0, grid.shape[0]-1):
                means.append((grid[j]+grid[j+1])/2)
            means = torch.tensor(means).view(-1, 1)
            no_grid_list.append(means.view(1, -1))
            if i == 0:
                means_list = means
            else:
                new_list = torch.zeros(means_list.shape[0]*self.number_means, i+1)
                for j in range(0, means_list.shape[0]):
                    new_list[j*self.number_means:j*self.number_means+self.number_means, 0:-1] = means_list[j, :]
                    for n in range(0, means.shape[0]):
                        new_list[j*self.number_means + n, -1] = means[n]
                means_list = new_list
            # new_mean = torch.arange(mins[i].item(), maxs[i].item(), step_size[i].item())

        no_grid_list_tens = torch.cat(no_grid_list)
        means = torch.t(means_list)

        torch.save(no_grid_list_tens, "means.pt")

        self.changing_policy.bas_fun.set_beta(step_size)

        self.changing_policy.bas_fun.set_means(means, no_grid_list_tens)

    def sample_data(self):
        if not self.loadNoSample:
            try:
                memory = torch.load('samples.pt')
                list = []
                for i in range(0, int(memory.shape[0]/2)):
                    list.append(memory[i, :])
                self.memory.set_memory(list)
            except:
                print('No samples found')
        else:
            print('Hi')
            for episode in range(1, self.starting_episodes):
                obs = self.env.reset()
                if self.stab:
                    obs = np.delete(obs, 2)

                for step in range(1, self.starting_steps):
                    act = self.changing_policy.random_action()

                    new_obs, rwd, done, _ = self.env.step(np.array([self.action_space[act]]))  # perform action on environment

                    if self.stab:
                        new_obs = np.delete(new_obs, 2)

                    self.memory.add(obs, act, rwd, new_obs, done)

                    obs = new_obs

                    self.env.render()
                    if done:
                        print('Done')
                        break
        return None

    def play_with_policy(self, episode):
        for episode in range(episode, self.test_episodes):
            print('episode: ' + str(episode))                           # Print Episode
            obs = self.env.reset()                                      # Reset environment
            if self.stab:
                obs = np.delete(obs, 2)
            rwd_sum = 0                                                 # accumulated reward
            for t in range(self.test_timesteps):
                act = self.changing_policy.best_act(torch.Tensor(obs))  # Select action by random or policy
                # print('Action with policy: ' + str(act))
                new_obs, rwd, done, _ = self.env.step(
                    np.array([self.action_space[act]]))                 # perform action on environment

                if self.stab:
                    new_obs = np.delete(new_obs, 2)

                # Perform step on environment
                self.memory.add(obs, act, rwd, new_obs, done)           # Add to memory

                rwd_sum += rwd                                          # Update accumulated reward

                obs = new_obs

                if done:                                                # If terminated
                    break                                                # stop episode
                
                if t % 3 == 0:
                    self.env.render()

            self.rew_overall += rwd_sum
            self.rwd_episodes_array = np.append(self.rwd_episodes_array, rwd_sum)
            self.plot(episode)
            '''
            if rwd_sum < rwd_sum_last:
                self.changing_policy.set_weights(self.old_weights)
                print('Reset weights.')
            else:
                print('Update weights')
                self.old_weights = self.changing_policy.get_weights()
                rwd_sum_last = rwd_sum
            '''
            if self.still_learning:
                self.lspi_main()

        return 0

    def plot(self, episode):
        self.rwd_overall_array = np.append(self.rwd_overall_array, self.rew_overall/episode)
        print('Mean reward now ' + str(self.rwd_episodes_array[-1]))
        print('Reward overall: ' + str(self.rwd_overall_array[-1]))
        t = np.arange(0, self.rwd_episodes_array.__len__(), 1)
        plt.figure(2)
        plt.plot(t, self.rwd_episodes_array, 'r--', label='Episode')
        plt.plot(t, self.rwd_overall_array, 'k', label='Overall')
        if episode == 1:
            plt.legend()
        plt.show(block=False)
        plt.pause(0.01)
        plt.savefig('2_lspi_10_005_1.png')
        self.rew_episodes = np.zeros(self.rew_episodes_len)

    def plot_distance(self, distances):
        t = np.arange(0, distances.__len__(), 1)
        plt.figure(3)
        plt.plot(t, distances)
        plt.xlabel('time')
        plt.ylabel('distance')
        plt.show(block=False)
        plt.pause(0.01)
