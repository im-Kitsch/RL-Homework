import numpy as np
import torch
import random as random

class DQNMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []
        self.memory_torch = torch.tensor([])
        self.position = 0

    def push(self, obs, act, rew, new_obs, done):
        new_entry = torch.cat((torch.Tensor(obs),
                          torch.tensor([act], dtype=torch.float32),
                          torch.tensor([rew], dtype=torch.float32),
                          torch.Tensor(new_obs),
                          torch.tensor([done], dtype=torch.float32)), 0)
        # print('add tensor: ' + str(tens))

        '''
        if self.memory_torch.shape[0] < self.max_size:
            if (self.memory_torch.shape[0] == 0):
                self.memory_torch = torch.cat((self.memory_torch, tens), 0).reshape(-1,1).t()
            else:
                self.memory_torch = self.memory_torch.t()
                self.memory_torch = torch.cat((self.memory_torch, tens.reshape(-1, 1)), 1)
                self.memory_torch = self.memory_torch.t()
        '''
        # new_entry = np.array([obs, act, rew, new_obs, done])

        if len(self.memory) < self.max_size:
            self.memory.append(None)
        self.memory[self.position] = new_entry
        self.position = (self.position + 1) % self.max_size

    # Get random Element from Memory
    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def get_memory(self):
        return self.memory

    def set_memory(self, memory):
        self.memory = memory
        self.position = len(self.memory)-1
        print('Memory length:' + str(len(self.memory)))

    def __len__(self):
        return len(self.memory)
