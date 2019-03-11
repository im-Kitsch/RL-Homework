import numpy as np


# for the infinite case, repeat add big matrix because append or concatenate not effcient
class Memory():
    def __init__(self, capacity, len_s, len_a, len_r, infinite=False):
        self.len_s, self.len_a, self.len_r = len_s, len_a, len_r
        self.len_sum = len_s + len_a + len_r + len_s
        self.capacity = capacity
        self.infinite = infinite
        self.memory = np.zeros((self.capacity, self.len_sum))
        self.counter = 0  # show how many samples has been saved

    def check_filled(self):
        if self.counter >= 1000:  # TODO to improve
            return True
        else:
            return False

    def push(self, s, a, r, s_ne):
        samples = np.concatenate(
            (s.reshape(-1, self.len_s), np.reshape(a, (-1, self.len_a)), np.reshape(r, (-1, self.len_r)),
             s_ne.reshape(-1, self.len_s)), axis=1
        )

        if self.infinite == False:
            counter = self.counter
            counter_ne = self.counter + samples.shape[0]
            capacity = self.capacity
            err = counter_ne - capacity
            if counter_ne <= capacity:
                self.memory[self.counter:counter_ne, :] = samples
                self.counter = counter_ne
            elif err <= capacity:
                self.memory[:counter - err] = self.memory[err:counter]
                self.memory[counter - err:] = samples
                self.counter = self.capacity
            else:
                self.memory = samples[-self.capacity:]
                self.counter = counter_ne
        elif self.infinite == True:
            counter = self.counter
            counter_ne = self.counter + samples.shape[0]
            mem_len = self.memory.shape[0]
            capacity = self.capacity

            if counter_ne <= mem_len:
                self.memory[self.counter:counter_ne] = samples
                self.counter = counter_ne
            else:
                if counter_ne <= mem_len + capacity:
                    add_arr = np.zeros((capacity, self.len_sum))
                else:
                    add_arr = np.zeros((counter_ne - mem_len, self.len_sum))
                self.memory = np.concatenate((self.memory, add_arr), axis=0)
                self.memory[counter:counter_ne] = samples
                self.counter = counter_ne

        return

    def sampling(self, batch):
        assert (batch <= self.counter)
        indices = np.random.choice(self.counter, batch, replace=False)
        samples = self.memory[indices]
        s, a, r, s_ne = samples[:, :self.len_s], samples[:, self.len_s:self.len_s + self.len_a], \
                        samples[:, self.len_s + self.len_a: self.len_s + self.len_a + self.len_r], \
                        samples[:, self.len_s + self.len_a + self.len_r:]

        return s, a, r, s_ne

    def get_all_sample(self):
        indices = np.arange(self.counter)
        samples = self.memory[indices]
        s, a, r, s_ne = samples[:, :self.len_s], samples[:, self.len_s:self.len_s + self.len_a], \
                        samples[:, self.len_s + self.len_a: self.len_s + self.len_a + self.len_r], \
                        samples[:, self.len_s + self.len_a + self.len_r:]

        return s, a, r, s_ne

    def savedata(self, path="DQN_MEMORY.npy"):
        np.save("DQN_MEMORY.npy", self.memory[:self.counter])
        return