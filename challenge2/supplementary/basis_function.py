import torch as torch
import numpy as np
import matplotlib.pyplot as plt


class RadialBasisFunction:
    def __init__(self, beta, num_actions, obs_space_size, number_means, obs_space):
        self.obs_space = obs_space
        self.number_actions = num_actions
        self.obs_space_size = obs_space_size
        self.number_means = number_means
        self.means = torch.zeros(self.obs_space_size, pow(self.number_means, self.obs_space_size))

    def set_means(self, no_grid):
        self.no_grid = no_grid
        mesh_1, mesh_2, mesh_3, mesh_4 = torch.meshgrid(
            (no_grid[0, :], no_grid[1, :], no_grid[2, :], no_grid[3, :]))
        mesh_1 = mesh_1.contiguous().view(1, -1)
        mesh_2 = mesh_2.contiguous().view(1, -1)
        mesh_3 = mesh_3.contiguous().view(1, -1)
        mesh_4 = mesh_4.contiguous().view(1, -1)
        self.means = torch.cat((mesh_1, mesh_2, mesh_3, mesh_4), dim=0)

    def calc_by_act(self, obs, act):
        phi = torch.zeros(int(pow(self.number_means, self.obs_space_size) + 1) * self.number_actions)
        phi[int(act)*(self.means.shape[1]+1):int(act+1)*(self.means.shape[1]+1)] = torch.reshape(self.eval_obs(obs), (1, -1))
        return phi

    def size(self):
        return (self.means.shape[1] + 1) * self.number_actions

    def eval_obs(self, sample):
        t_sample = sample.view(-1, 1)
        d = torch.norm(t_sample-self.means, dim=0)
        v = torch.exp(-d/2)
        return torch.cat((torch.Tensor([1]).view(-1, 1), v.view(-1, 1)))


class RadialBasisFunctionTrained:
    def __init__(self, no_grid):
        self.number_actions = 3
        self.obs_space_size = 4
        self.number_means = 3
        self.means = torch.zeros(self.obs_space_size, pow(self.number_means, self.obs_space_size))
        self.set_means(no_grid)

    def set_means(self, no_grid):
        self.no_grid = no_grid
        mesh_1, mesh_2, mesh_3, mesh_4 = torch.meshgrid(
            (no_grid[0, :], no_grid[1, :], no_grid[2, :], no_grid[3, :]))
        mesh_1 = mesh_1.contiguous().view(1, -1)
        mesh_2 = mesh_2.contiguous().view(1, -1)
        mesh_3 = mesh_3.contiguous().view(1, -1)
        mesh_4 = mesh_4.contiguous().view(1, -1)
        self.means = torch.cat((mesh_1, mesh_2, mesh_3, mesh_4), dim=0)

    def calc_by_act(self, obs, act):
        phi = torch.zeros(int(pow(self.number_means, self.obs_space_size) + 1) * self.number_actions)
        phi[int(act)*(self.means.shape[1]+1):int(act+1)*(self.means.shape[1]+1)] = torch.reshape(self.eval_obs(obs), (1, -1))
        return phi

    def eval_obs(self, obs):

        obs = np.delete(obs, 2)
        sample = torch.Tensor(obs)
        t_sample = sample.view(-1, 1)
        d = torch.norm(t_sample-self.means, dim=0)
        v = torch.exp(-d/2)
        return torch.cat((torch.Tensor([1]).view(-1, 1), v.view(-1, 1)))





