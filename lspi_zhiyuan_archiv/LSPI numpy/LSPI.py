import gym
from gym.wrappers.monitor import Monitor
from quanser_robots.cartpole import ctrl
from quanser_robots.common import Logger, GentlyTerminating

import numpy as np
from Memory import Memory
from BasisFunction import BasisRBF
import scipy.linalg
from matplotlib import pyplot as plt

# TODO means with
class LSPI:
    def __init__(self, env, **kwargs):
        # hyper parameter initialization
        self.env = env
        self.verbose = kwargs.get("gamma", False)  # output help information

        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon", 0.00)   # epsilon-Greedy parameter

        self.n_state = env.observation_space.shape[0]
        self.acts_n = kwargs.get("acts_n", 3)  # TODO to check if 24/-24 is to big
        # self.acts_list = np.linspace(env.action_space.low[0], env.action_space.high[0], self.acts_n)
        self.acts_list = np.linspace(-5., 5., self.acts_n)
        self.acts_index_list = np.arange(self.acts_n).reshape(1, -1)

        # class initialization for feature function and the memory
        low, high, r_low, r_high = self.get_state_range()
        self.state_low = low
        self.state_high = high
        self.state_discrete_list = kwargs.get("state_discrete_list", [3]*self.n_state) #TODO set as para
        self.r_low = r_low
        self.r_high = r_high

        self.basis_func = BasisRBF(rbf_gamma=5, act_n= self.acts_n, low = self.state_low, high = self.state_high,
                                   state_discrete_list=self.state_discrete_list)
        self.memory = Memory(10000, self.n_state, 1, 1, infinite=False)

        # if self.verbose:
        #     self.basis_func.feature_visualize()

        # # TODO, maybe set as random
        self.w_old = np.zeros((self.basis_func.features_sum, ))
        self.w = np.random.random(self.w_old.shape) * 0.001
        self.A = np.zeros((self.basis_func.features_sum, self.basis_func.features_sum))
        self.b = np.zeros((self.basis_func.features_sum, 1))
        #
        self.batch_size = kwargs.get("batch_size", 1)
        #
        # self.a_last = 0.
        self.rand_P_mode = 1
        return

    def policy(self, s):
        # TODO, let epsilon decrease as steps increases
        # TODO in one batch, it should be some using random policy some using greddy
        eps = np.random.uniform()

        if eps < self.epsilon:
            a = self.random_policy()
        else:
            a = self.greedy_policy(s)
        self.a_last = a
        return a

    def random_policy(self):
        if self.rand_P_mode == 1:
            # random choose actions from discretized actions
            a = np.random.choice( self.acts_list)
            a = a.reshape(-1)
        return a

    def greedy_policy(self, s, requires = "a"):
        s = s.reshape(-1, self.n_state)
        phi_sa = self.basis_func(s, np.arange(self.acts_n), require_enumerate=True)
        q_sa = phi_sa @ self.w
        a_index = np.argmax(q_sa, axis=1)
        a = self.acts_list[a_index]

        a = a.reshape(-1)
        if requires == "a":
            return a
        elif requires == "index":
            return a_index
        elif requires == "both":
            return a, a_index

    def LSTDQ(self, s, a, r, s_ne):
        # TODO, no only works for batch_size = 1, should be do modification for bigger batch

        # TODO precondition value
        # TODO, think this again

        a_indices = np.argwhere( a.reshape(-1, 1) == self.acts_list.reshape(1, -1))  # Attention for the flatten()
        a_indices = a_indices[:, 1]
        # TODO, this is assumed that only actions in the lists [-24, .., 24] are used
        a_indices = a_indices.reshape(-1, 1)
        a_indices_ne = self.greedy_policy(s_ne, requires="index")
        assert a_indices.shape == a.shape

        opt = False

        if not opt: # TODO
            self.A = np.zeros_like(self.A)
            self.b = np.zeros_like(self.b)

            np.fill_diagonal(self.A, 0.00001)  # precondition value to ensure that the matrix is full rank
            phi_sa = self.basis_func(s, a_indices)
            phi_sa_ne = self.basis_func(s_ne, a_indices_ne)

            self.A += phi_sa.T @ (phi_sa - self.gamma * phi_sa_ne)
            self.b += phi_sa.T @ r  # r is also only scalar but not array here

            rank_A = np.linalg.matrix_rank(self.A)

            if rank_A == self.basis_func.features_sum:
                w = scipy.linalg.solve(self.A, self.b)
            else:
                w = scipy.linalg.lstsq(self.A, self.b)[0]
        else:
            B = 1/0.00001 * np.eye(self.basis_func.features_sum)
            b = np.zeros((self.basis_func.features_sum, 1))
            for one_s, one_a_index, one_r, one_a_ne_index, one_s_ne in zip(s, a_indices, r, a_indices_ne, s_ne):
                phi_sa = self.basis_func(one_s, one_a_index).T
                phi_sa_ne = self.basis_func(one_s_ne, one_a_ne_index).T

                temp_err = ( phi_sa - self.gamma * phi_sa_ne).T
                B = B - B @ phi_sa @ temp_err * B / ( 1 + temp_err @ B @ phi_sa)
                b += phi_sa * one_r
            w = B @ b

        return w.reshape((-1, 1))

    def get_state_range(self, verbose = True):
        env = self.env

        temp_memory = Memory(500000, self.n_state, 1, 1, infinite=True)
        if True: #TODO, the other get means functions
            episodes = 15
            for i in range(episodes):
                done = False
                s = env.reset()
                while not done:
                    a = env.action_space.sample()
                    s_ne, reward, done, _ = env.step(a)
                    temp_memory.push(s, a, reward, s_ne)
                    s = s_ne
        s, _, r, s_ne = temp_memory.get_all_sample()

        s_low = np.min(s, axis=0).flatten()
        s_high = np.max(s, axis=0).flatten()
        low = env.observation_space.low
        high = env.observation_space.high

        r_high = np.max(r)
        r_low = np.min(r)

        # high[high == np.inf] = s_high[high == np.inf]
        # low[low == -np.inf] = s_low[low == -np.inf]

        high = np.array([0.407, 0.25, -0.96, 2, 8])
        low = np.array([-0.407, -0.25, -1, -2, -8])

        if verbose:
            print("using ", temp_memory.counter, "samples to build means for features")
            print("original high and low", env.observation_space.high, env.observation_space.low)
            print("observed high and low", s_high, s_low)
            print("manually high and low", high, low)
            print("reward high and low", r_high, r_low)

        del temp_memory, s
        return low, high, r_low, r_high

    def normalize_range01(self, state, low, high):

        low, high = low.reshape(1, -1), high.reshape(1, -1)
        state = state.reshape(-1, low.shape[1])

        normalized = (state - low)/(high - low)

        return normalized


if __name__ ==  "__main__":
    # Choose enviroment:
    # 1 = CartPoleStab
    # 2 = CartPoleSwingUp
    # 3 = FurutaPend
    env_choose = 1
    if env_choose == 1:
        # env = Logger(GentlyTerminating(gym.make('CartpoleStabShort-v0')))
        env = gym.make('CartpoleStabShort-v0')

    elif env_choose == 4:
        env = gym.make('CartpoleStabRR-v0')

    elif env_choose == 2:
        env = gym.make('CartpoleSwingShort-v0')

    else:
        env = gym.make('Qube-v0')

    # LSPI Test
    lspi = LSPI(env)

    episode = 500

    train_intervel = 5000

    # create initial memory
    i = lspi.memory.capacity
    while i > 0:

        done = False
        s = env.reset()
        while not done:
            i -= 1
            a = lspi.random_policy()
            s_ne, reward, done, _ = env.step(a)
            lspi.memory.push(s, a, reward, s_ne)
            s = s_ne

    print("created memory random memory samples")

    for i in range(episode):
        done = False

        s = env.reset()

        rewards_episodes = []
        step = 0
        rewards_accumulate = 0
        whole_steps = 0
        while not done:

            a = lspi.policy(s.reshape(1, -1))

            s_ne, r, done, _ = env.step(a)
            r = np.array(r)

            lspi.memory.push(s, a, r, s_ne)

            if step % train_intervel == 0:
                train_s, train_a, train_r, train_s_ne = lspi.memory.sampling(5000)
                train_r = lspi.normalize_range01(train_r.reshape(-1, 1), lspi.r_low, lspi.r_high)
                w = lspi.LSTDQ(train_s, train_a, train_r, train_s_ne)
                lspi.w_old = lspi.w
                lspi.w = w

            s = s_ne

            step += 1
            whole_steps += 1
            rewards_accumulate += r

        print(i, step, np.linalg.norm(lspi.w - lspi.w_old), rewards_accumulate)
    rewards_episodes.append(rewards_accumulate)
    plt.plot(rewards_episodes)







