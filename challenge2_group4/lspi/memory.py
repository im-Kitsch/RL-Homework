import torch as torch
import random as random


class LSPIMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []
        self.position = 0
        return None

    # ----------------
    # 0:4 = obs
    # 5 = action
    # 6 = reward
    # 7:11 = new_obs
    # 12 = done
    def add(self, obs, act, rew, new_obs, done):
        new_entry = torch.cat((torch.Tensor(obs),
                          torch.tensor([act], dtype=torch.float32),
                          torch.tensor([rew], dtype=torch.float32),
                          torch.Tensor(new_obs),
                          torch.tensor([done], dtype=torch.float32)), 0)
        if self.memory.__len__() < self.max_size:
            self.memory.append(None)
        self.memory[self.position] = new_entry
        self.position = (self.position + 1) % self.max_size

    # Get random Element from Memory
    def sample(self, batch_size):
        return random.sample(self.memory, min(self.memory.__len__(), batch_size))

    def get_all_samples(self):
        return self.memory

    def set_memory(self, memory):
        self.memory = memory
        self.position = self.memory.__len__()-1

    def size(self):
        return self.memory.__len__()

    def clear(self):
        self.memory.clear()
        self.position = 0
