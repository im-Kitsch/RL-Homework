import numpy as np


# for the infinite case, repeat add big matrix because append or concatenate not effcient

# TODO TO improve to flexible choose saved data and type maybe panda
# TODO speed up; flexible to get data
# TODO if filled with data

class Memory:
    def __init__(self, capacity, len_compo, component):
        assert len(component) == len(len_compo)
        self.capacity = capacity
        self.len_compo = len_compo
        self.components = component
        self.memory_width = np.sum(len_compo)

        self.data_frame = np.zeros((self.capacity, self.memory_width))
        self.interval = np.cumsum(len_compo)[0:-1]
        self.counter = 0

        self.filled = False
        return

    def push_one(self, elements):
        elements = np.concatenate( [np.array(ele).flatten() for ele in elements], axis=0)
        self.data_frame[self.counter] = elements
        self.counter += 1
        if self.counter >= self.capacity:
            self.counter -= self.capacity
            self.filled = True
        return

    def sampling(self, amount="all"):
        if amount == "all":
            if self.filled:
                data = self.data_frame
            else:
                data = self.data_frame[:self.counter]
        else:
            if self.filled:
                indices = np.random.choice(self.capacity, amount, replace=False)
            else:
                indices = np.random.choice(self.counter, amount, replace=False)
            data = self.data_frame[indices]
        broken_data = np.split(data, self.interval, axis=1)
        return broken_data

# class Memory():
#     def __init__(self, capacity, len_s, len_a, len_r, infinite=False):
#         self.len_s, self.len_a, self.len_r = len_s, len_a, len_r
#         self.len_sum = len_s + len_a + len_r + len_s
#         self.capacity = capacity
#         self.infinite = infinite
#         self.memory = np.zeros((self.capacity, self.len_sum))
#         self.counter = 0  # show how many samples has been saved
#
#     def check_filled(self):
#         if self.counter >= 1000:  # TODO to improve
#             return True
#         else:
#             return False
#
#     def push(self, s, a, r, s_ne):
#         samples = np.concatenate(
#             (s.reshape(-1, self.len_s), np.reshape(a, (-1, self.len_a)), np.reshape(r, (-1, self.len_r)),
#              s_ne.reshape(-1, self.len_s)), axis=1
#         )
#
#         if not self.infinite:
#             counter = self.counter
#             counter_ne = self.counter + samples.shape[0]
#             capacity = self.capacity
#             err = counter_ne - capacity
#             if counter_ne <= capacity:
#                 self.memory[self.counter:counter_ne, :] = samples
#                 self.counter = counter_ne
#             elif err <= capacity:
#                 self.memory[:counter - err] = self.memory[err:counter]
#                 self.memory[counter - err:] = samples
#                 self.counter = self.capacity
#             else:
#                 self.memory = samples[-self.capacity:]
#                 self.counter = counter_ne
#         else:
#             counter = self.counter
#             counter_ne = self.counter + samples.shape[0]
#             mem_len = self.memory.shape[0]
#             capacity = self.capacity
#
#             if counter_ne <= mem_len:
#                 self.memory[self.counter:counter_ne] = samples
#                 self.counter = counter_ne
#             else:
#                 if counter_ne <= mem_len + capacity:
#                     add_arr = np.zeros((capacity, self.len_sum))
#                 else:
#                     add_arr = np.zeros((counter_ne - mem_len, self.len_sum))
#                 self.memory = np.concatenate((self.memory, add_arr), axis=0)
#                 self.memory[counter:counter_ne] = samples
#                 self.counter = counter_ne
#
#         return
#
#     def sampling(self, batch):
#         assert (batch <= self.counter)
#         indices = np.random.choice(self.counter, batch, replace=False)
#         samples = self.memory[indices]
#         s, a, r, s_ne = samples[:, :self.len_s], samples[:, self.len_s:self.len_s + self.len_a], \
#                         samples[:, self.len_s + self.len_a: self.len_s + self.len_a + self.len_r], \
#                         samples[:, self.len_s + self.len_a + self.len_r:]
#
#         return s, a, r, s_ne
#
#     def get_all_sample(self):
#         indices = np.arange(self.counter)
#         samples = self.memory[indices]
#         s, a, r, s_ne = samples[:, :self.len_s], samples[:, self.len_s:self.len_s + self.len_a], \
#                         samples[:, self.len_s + self.len_a: self.len_s + self.len_a + self.len_r], \
#                         samples[:, self.len_s + self.len_a + self.len_r:]
#
#         return s, a, r, s_ne
#
#     def savedata(self, path="DQN_MEMORY.npy"):
#         np.save("DQN_MEMORY.npy", self.memory[:self.counter])
#         return


if __name__ == "__main__":
    a = {}
    b = {}
    memory = Memory(60, [4, 1, 4, 1, 1, 1], ["s", "a", "s_ne", "r", "indices", "done"])
    check_point = [15, 60, 65]
    for i in range(90):
        memory.push_one([
            np.random.random(4), [2], np.random.random(4), 1, -1, False
        ])
        if i in check_point:
            a[i] = memory.sampling(1)
            b[i] = memory.sampling("all")
