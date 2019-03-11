import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from matplotlib import pyplot as plt

class BasisRBF:
    def __init__(self, rbf_gamma, act_n, low, high, state_discrete_list,
                 atan_process=False, in_normalization=True, fea_normalization=True):

        self.gamma = rbf_gamma
        self.act_n = act_n

        self.fea_normalization = fea_normalization
        self.in_normalization = in_normalization
        self.atan_process = atan_process

        self.s_high = high.reshape(1, -1)
        self.s_low = low.reshape(1, -1)
        self.s_range = self.s_high - self.s_low

        # if self.atan_process: # TODO To move to LSPI
        #     self.s_high[0, 1] = 0.25
        #     self.s_high[0, 2:4] = self.s_high[0, 3:]
        #     self.s_high = self.s_high[0, :4]
        #
        #     self.s_low[0, 1] = 0.
        #     self.s_low[0, 2:4] = self.s_low[0, 3:]
        #     self.s_low = self.s_low[0, :4]
        #
        #     self.s_range = self.s_high - self.s_low

        if not(self.in_normalization):
            self.means = self.build_means(low, high, state_discrete_list)
        else:
            self.means = self.build_means(np.zeros_like(low), np.ones_like(high), state_discrete_list)

        self.features_n = self.means.shape[0] + 1
        self.features_sum = self.features_n * act_n

        assert self.means.shape[1] == (self.s_low.shape[1] - self.atan_process)
        return

    def __call__(self, states, acts_indices, require_enumerate = False):
        """
        :param samples: sample, shape: batch * s_len
        :param acts_indices: list/array, the tuple of actions for phi(s, a)
        :return: phi(s,a), shape: batch * n_act * K, K ist (n_rbf+1) * # possible actions
        """

        # preprocess the states
        states = self.preprocessing(states)


        # rbf kernels
        phi = self.rbf(states)

        # combining

        # TODO need a clean version
        if require_enumerate == True:
            acts_indices = np.array(acts_indices).flatten()
            phi_s_a = np.zeros((len(states), len(acts_indices), self.features_sum))
            indices_a = acts_indices[:, None]
            indices_phi = np.arange(self.features_n)[None, :] + indices_a * self.features_n
            phi = phi[:, None, :]
            phi_s_a[:, indices_a, indices_phi] = phi

        else:
            phi_s_a = np.zeros((len(states),  self.features_sum))
            # indices_a = acts_indices[:, None]
            indices_phi = np.arange(self.features_n)[None, :] + acts_indices.reshape(-1, 1) * self.features_n
            # phi = phi[:, None, :]
            phi_s_a[:, indices_phi] = phi

        return phi_s_a

    def preprocessing(self, s):

        if self.atan_process:
            angles = np.arctan2(s[:, [1]], -s[:, [2]]) # consider sin and cos of (pi - theta)
            s = np.concatenate((s[:, [0]], angles, s[:, 3:]), axis=1)

        if self.in_normalization:
            s = s - self.s_low
            s /= self.s_range

        return s

    def rbf(self, states):
        phi = rbf_kernel(states, self.means, gamma = self.gamma)

        phi = np.concatenate((1.0 * np.ones((phi.shape[0], 1)), phi), axis=1)  # add element 1

        if self.fea_normalization:
            phi /= np.sum(phi, axis=1, keepdims=True)


        return phi

    def build_means(self, low, high, state_discrete_list, verbose=True):
        low = low.flatten()
        high = high.flatten()
        #TODO, maybe not proper for the function?
        means = [np.linspace(start, stop, step) for start, stop, step in zip(low, high, state_discrete_list)]
        means = np.meshgrid(*means)
        means = [i.flatten() for i in means]
        means = np.array(means)
        means = means.T

        if verbose:
            print(means.shape[0], " features for every action")
        return means

    def feature_visualize(self):

        number_point = 50
        low = np.min(self.means, axis=0)  # then don't need to normalize the input
        high = np.max(self.means, axis=0)
        s_len = self.s_low.shape[1]

        # visualization of feature function
        for i in range(s_len):

            states = np.zeros((number_point, s_len))

            plt.figure()
            states[:, i] = np.linspace(low[i], high[i], number_point)
            rbf_features = self.rbf(states)
            rbf_features = rbf_features[:, 1:]
            plt.title("feature for s" + str(i) + ", the other is fixed with 0")
            x = np.repeat(states[:, [i]], rbf_features.shape[1], axis=1)
            plt.plot(x, rbf_features)

            # plt.figure()
            # states[:, :] = high[None, :]
            # states[:, i] = np.linspace(low[i], high[i], number_point)
            # rbf_features = self.rbf(states)
            # plt.title("feature for s" + str(i) + ", the other is fixed with highest")
            # x = np.repeat(states[:, [i]], rbf_features.shape[1], axis=1)
            # plt.plot(x, rbf_features)

            # plt.figure()
            # states[:, :] = low[None, :]
            # states[:, i] = np.linspace(low[i], high[i], number_point)
            # rbf_features = lspi.basis_func.rbf(states)
            # plt.title("feature for s" + str(i) + ", the other is fixed with lowest")
            # x = np.repeat(states[:, [i]], rbf_features.shape[1], axis=1)
            # plt.plot(x, rbf_features)
            plt.show()

        return

    # Not used anymore, because it was already implemented in sklearn
    # def rbf(self, s):
    #     """
    #     :param s: sample, shape: batch * s_len
    #     :return: rbf feature, batch * (n_rbf+1), i.e. batch * k
    #     """
    #
    #     if self.atan_process:
    #         angles = np.arctan2(s[:, [1]], -s[:, [2]]) # consider sin and cos of (pi - theta)
    #         s = np.concatenate((s[:, [0]], angles, s[:, 3:]), axis=1)
    #
    #     if self.in_normalization:
    #         s = self.input_normalization(s)
    #
    #     s = s[:, None, :]
    #     means = self.means[None, :, :]
    #
    #     phi = s - means  # shape batch * (k-1) * s_len
    #     phi = np.linalg.norm(phi, axis=2) # shape: batch * (k-1)
    #     phi = np.exp(-self.gamma * phi)
    #     one_vec = np.ones((phi.shape[0], 1))
    #     phi = np.concatenate((one_vec, phi), axis=1)
    #     if self.fea_normalization:
    #         phi /= np.sum(phi, axis=1, keepdims=True)
    #     return phi

    # def input_normalization(self, samples):
    #
    #     normalized_samples = samples - self.s_low
    #     normalized_samples /= self.s_range
    #
    #     return normalized_samples




# TODO means width
if __name__== "__main__":
    print("Test THis Class")
    s_len = 5
    n_sample = 2
    n_rbf = 3
    act_n = 5

    gamma = 5

    st_list = [n_rbf] * (s_len )

    low = -np.arange(s_len - 1) - 1
    high = np.arange(s_len - 1) + 1
    # samples = np.arange(n_sample * s_len).reshape(-1, s_len)

    low = np.array([-2, -1., -1., -3., -3.])
    high = np.array([+2, +1., +1., +3., +3.])
    # samples = np.array([[0.5, 1., -1., 3, 3],
    #                     [0.5, -1, -1, 3, 3]])
    basisFunc = BasisRBF(gamma, act_n, low, high, st_list, fea_normalization=True)
    #
    # samples_processed = basisFunc.preprocessing(samples)
    # phi = basisFunc.rbf(samples_processed)
    # phi_sa = basisFunc(samples, [0, 2])

    # print("low, high")
    # print(low, high)
    # print("samples, preprocessed samples")
    # print(samples)
    # print(samples_processed)
    # print("means")
    # print(basisFunc.means)
    # print("Phi")
    # print(phi)
    # print("Phi_sa")
    # print(phi_sa)

    basisFunc.feature_visualize()
