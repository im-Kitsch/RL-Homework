import random
import torch as torch
import numpy as np
from supplementary.basis_function import RadialBasisFunction, RadialBasisFunctionTrained


class Policy:

    def __init__(self, action_space, obs_space_size, number_means, obs_space, stab=True):
        self.action_space = action_space
        self.obs_space_size = obs_space_size
        self.number_means = number_means
        self.bas_fun = RadialBasisFunction(0.99, self.action_space.shape[0], obs_space_size, self.number_means, obs_space, stab)
        self.weights = torch.empty(self.action_space.shape[0] * (pow(number_means, obs_space_size) + 1),
                                           dtype=torch.float32).uniform_(0, 1).view(-1, 1)
    def random_action(self):
        return random.randrange(0, len(self.action_space))

    def q_val(self, obs, act):
        rbf = self.bas_fun.calc_by_act(obs, act).view(1, -1)
        return torch.mm(rbf, self.weights)

    def best_act(self, obs):
        a = self.action_space.shape[0]
        qs = np.zeros(3)
        for i in range(0, a):
            qs[i] = self.q_val(obs, i)
        q_pos = np.argmax(qs)
        return q_pos

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        self.weights = new_weights


class PolicyTrained:

    def __init__(self):
        self.means = torch.load("./supplementary/means.pt")
        self.weights = torch.load("./supplementary/weights.pt")
        self.bas_fun = RadialBasisFunctionTrained(self.means)
        self.action_space = [-9., 0., 9]

    def q_val(self, obs, act):
        rbf = self.bas_fun.calc_by_act(obs, act).view(1, -1)
        return torch.mm(rbf, self.weights)

    def best_act(self, obs):
        a = self.action_space.__len__()
        qs = np.zeros(3)
        for i in range(0, a):
            qs[i] = self.q_val(obs, i)
        q_pos = np.argmax(qs)
        return np.array([self.action_space[q_pos.item()]])
