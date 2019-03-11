import gym
from gym.wrappers.monitor import Monitor
from quanser_robots.cartpole import ctrl
from quanser_robots.common import Logger, GentlyTerminating

import numpy as np
from Memory import Memory
from BasisFunc import RBFBasis
from ShareFunc import build_means_discrete
import scipy.linalg
import math
from matplotlib import pyplot as plt

# TODO Memory and push, iteratool https://stackoverflow.com/questions/15366053/flatten-a-nested-list-of-variable-sized-sublists-into-a-scipy-array


# one part policy part
# one part LSTDQ part
class LSPI:
    def __init__(self, env, memory, rbf_basis, acts_list, **kwargs):
        # hyper parameter initialization
        self.verbose = kwargs.get("verbose", True)  # output help information

        self.env = env
        self.memory = memory
        self.rbf_basis = rbf_basis
        self.acts_list = acts_list

        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon", "greedy")  # epsilon-Greedy parameter
        self.acts_n = len(acts_list)
        self.W = np.zeros((self.rbf_basis.len_fea, 1))
        self.W = np.random.random(self.W.shape)
        return

    def LSTDQ(self, memory_package):
        states, actions, acts_indices, states_ne, rewards, dones = memory_package

        opt = False
        acts_indices = acts_indices.astype(int)
        dones = dones.astype(bool)

        if not opt:
            k = self.rbf_basis.len_fea
            A = np.zeros((k, k))
            b = np.zeros((k, 1))
            np.fill_diagonal(A, 0.00001) # precondition value to ensure that the matrix is full rank

            batch = len(states)
            steps_per_epoc = 1000
            begin_index = 0
            while begin_index < batch:
                indices = range(begin_index, begin_index + steps_per_epoc )
                begin_index += steps_per_epoc
                s, a_ind, s_ne, r, dn = states[indices], acts_indices[indices], states_ne[indices], rewards[indices], dones[indices]

                phi_sa = self.rbf_basis(s, a_ind)
                _, acts_ne_indices = self.greedy_policy(s_ne)
                acts_ne_indices = acts_ne_indices[:, None]
                phi_sa_ne = self.rbf_basis(s_ne, acts_ne_indices)
                phi_sa_ne[~dn.flatten()] = np.zeros((1, phi_sa_ne.shape[1]))

                A += phi_sa.T @ (phi_sa - self.gamma * phi_sa_ne)
                b += phi_sa.T @ r  # r is also only scalar but not array here

            rank_A = np.linalg.matrix_rank(A)

            if rank_A == k:
                W = scipy.linalg.solve(A, b)
            else:
                print("this time singular")
                W = scipy.linalg.lstsq(A, b)[0]

        else:
            print("LSTDQ-opt is deprecated")

        return W

    def LSPI_train(self):
        err = np.inf
        while err > 0.005:
            components = self.memory.sampling("all")
            W = self.LSTDQ(components)
            err = np.linalg.norm(W - self.W)
            self.W = W
            print("updated W, {}".format(err))
        return

    def policy(self, state):
        if state.ndim == 1:
            state = state[None, :]

        if self.epsilon is "greedy":
            act, act_index = self.greedy_policy(state)
        else:
            seeding = np.random.random()
            if seeding < self.epsilon:
                act, act_index = self.random_policy(state)
            else:
                act, act_index = self.greedy_policy(state)
        return act, act_index

    def greedy_policy(self, state):
        batch = state.shape[0]
        phi_s_a = self.rbf_basis(state)
        q_s_a = phi_s_a @ self.W
        act_index = np.argmax(q_s_a, axis=1).flatten()
        act = self.acts_list[act_index]
        return act, act_index

    def random_policy(self, state):
        act_index = np.random.choice(self.acts_n, state.shape[0])
        act = self.acts_list[act_index]
        return act, act_index


def drop_state(state, i):
    if state.ndim == 1:
        state = state[None, :]
    return np.delete(state, i, axis=1)


def normalization_zero_one(state, minimum, maximum):
    if state.ndim == 1:
        state = state[None, :]
    if minimum.ndim == 1:
        minimum = minimum[None, :]
    if maximum.ndim == 1:
        maximum = maximum[None, :]
    return (state - minimum)/(maximum - minimum)


# should be flexible
def state_preprocess_config(i_drop, minimum, maximum):
    minimum = minimum
    maximum = maximum
    i_drop = i_drop

    def process(state):
        state = drop_state(state, i_drop)
        state = normalization_zero_one(state, minimum, maximum)
        return state
    return process


def reward_nomalize(reward_low, reward_high):
    reward_low = reward_low
    reward_high = reward_high

    def r_normalize(rewards):
        return (rewards - reward_low) / (reward_high - reward_low)
    return r_normalize

def random_sample_feeding(env, lspi, state_preprocess):
    step_counter = lspi.memory.capacity
    while step_counter > 0:
        done = False
        s = env.reset()
        while not done:
            step_counter -= 1
            a, a_ind = lspi.random_policy(s.reshape(1, -1))
            s_ne, r, done, _ = env.step(a)
            lspi.memory.push_one([
                state_preprocess(s), a, a_ind, state_preprocess(s_ne), r, done
            ])
            s = s_ne

        done = True
    return


# def evaluate(env, lspi, state_preprocess):
#     for i in range(10):
#         done = False
#
#         s = env.reset()
#         step_counter_episode = 0
#         step_record = []
#         while not done:
#             step_counter_episode += 1
#             s_processed = state_preprocess(s)
#             a, a_ind = lspi.policy(s_processed)
#             s_ne, r, done, _ = env.step(a)
#             s = s_ne
#         step_record.append(step_counter_episode)
#
#     print("This evaluation this update, average steps,")
#
#     return

def collect_data_config(env, lspi, state_preprocess, reward_preprocess):
    episodes_max = 1000

    def collecting(steps_per_good_episode, num_good_episode=np.inf, whole_samples=np.inf):
        good_sample = []
        bad_sample = []
        good_sampling_counter = 0

        for i in range(episodes_max):
            done = False
            record = []
            s = env.reset()
            step_counter_episode = 0

            while not done:
                step_counter_episode += 1

                s_processed = state_preprocess(s)
                a, a_ind = lspi.policy(s_processed)
                s_ne, r, done, _ = env.step(a)
                s_ne_processed = state_preprocess(s_ne)

                record.append([
                    s_processed, a, a_ind, s_ne_processed, reward_preprocess(r), done
                ])

                s = s_ne
            if step_counter_episode >= steps_per_good_episode:
                good_sample.extend(record)
                good_sampling_counter += 1
            else:
                bad_sample.extend(record)

            if good_sampling_counter >= num_good_episode or len(good_sample) >= whole_samples:
                break

        if True:
            print("collect {} episode, good episode {}, ave steps {}, whole steps {} efficient {}".format(
                i+1, good_sampling_counter, len(good_sample)/good_sampling_counter, len(good_sample), good_sampling_counter/(i+1)))
        return good_sample
    return collecting


def main():
    env = gym.make('CartpoleStabShort-v0')
    env.seed(0)
    np.random.seed(0)

    act_list = np.array([-5., 0., 5.])
    low = np.array([-0.407, -np.sin(0.25), -4, -20])
    high = np.array([0.407, np.sin(0.25), 4, 20])
    discrete_list = [6, 6, 5, 5]
    record_components = ["s", "a", "a_index", "s_ne", "r", "done"]
    record_com_len = [4, 1, 1, 4, 1, 1]
    state_preprocess = state_preprocess_config(2, low, high)

    r_low, r_high = env.reward_range[0], env.reward_range[1]
    reward_preprocess = reward_nomalize(r_low, r_high)

    # means = build_means_discrete(np.zeros_like(low), np.ones_like(high), discrete_list)
    means = np.random.random((800, 4))
    rbf_gamma = 5
    rbf_basis = RBFBasis(rbf_gamma, len(act_list), means, fea_normalization=True, feature_biased=True)

    memory = Memory(10000, record_com_len, record_components)

    lspi = LSPI(env, memory, rbf_basis, act_list, epsilon=0.03)

    # build_samples
    random_sample_feeding(env, lspi, state_preprocess)
    print("created memory random memory samples")

    collect = collect_data_config(env, lspi, state_preprocess, reward_preprocess)

    # pre-train
    lspi.LSPI_train()
    components = collect(100, whole_samples=10000)
    for item in components:
        lspi.memory.push_one(item)

    lspi.LSPI_train()
    components = collect(100, whole_samples=10000)
    for item in components:
        lspi.memory.push_one(item)

    lspi.LSPI_train()
    components = collect(200, whole_samples=10000)
    for item in components:
        lspi.memory.push_one(item)

    lspi.LSPI_train()
    components = collect(200, whole_samples=10000)
    for item in components:
        lspi.memory.push_one(item)

    # episodes = 100000
    # train_interval = 10000
    # train_interval_episode = 100
    # step_counter = 0
    #
    # for i in range(episodes):
    #     done = False
    #     reward_record = 0.0
    #     s = env.reset()
    #     step_counter_episode = 0
    #     if (i % train_interval_episode) == 0:
    #         lspi.LSPI_train()
    #     while not done:
    #         # if (step_counter % train_interval) == 0:
    #         #     lspi.LSPI_train()
    #
    #         step_counter += 1
    #         step_counter_episode += 1
    #
    #         s_processed = state_preprocess(s)
    #         a, a_ind = lspi.policy(s_processed)
    #         s_ne, r, done, _ = env.step(a)
    #         s_ne_processed = state_preprocess(s_ne)
    #         reward_record += r
    #
    #         lspi.memory.push_one([
    #             s_processed, a, a_ind, s_ne_processed, reward_preprocess(r), done
    #         ])
    #
    #         s = s_ne
    #
    #     print("episode{}, step{}, reward {} ave_reward".format(
    #         i, step_counter_episode, reward_record), reward_record/step_counter_episode )

    return

if __name__== "__main__":
    main()
