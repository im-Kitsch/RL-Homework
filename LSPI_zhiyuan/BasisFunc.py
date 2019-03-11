import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from ShareFunc import build_means_discrete


class RBFBasis:
    def __init__(self, rbf_gamma, act_n, means, fea_normalization=True, feature_biased = True):

        self.rbf_gamma = rbf_gamma
        self.act_n = act_n
        self.means = means
        self.fea_normalization = fea_normalization
        self.act_list = np.arange(self.act_n)

        self.feature_biased = feature_biased
        self.len_mean = len(means)
        self.len_fea_short = self.len_mean + self.feature_biased
        self.len_fea = self.len_fea_short * self.act_n
        return

    def __call__(self, states, act_indices="all"):

        states = states.reshape(-1, self.means.shape[1])

        phi = self.rbf(states)

        if act_indices is "all":
            phi_s_a = np.zeros((len(states), self.act_n, self.len_fea))
            indices_a = self.act_list[:, None]
            indices_phi = np.arange(self.len_mean + 1)[None, :]
            indices_phi = indices_a * self.len_fea_short + indices_phi
            phi_s_a[:, indices_a, indices_phi] = phi[:, None, :]
        else:
            assert len(states) == len(act_indices)

            phi_s_a = np.zeros((len(states), self.len_fea))
            indices_phi = act_indices * self.len_fea_short + np.arange(self.len_fea_short)[None, :]
            phi_s_a[np.arange(len(states))[:, None], indices_phi] = phi
        return phi_s_a

    def rbf(self, states):
        rbf_bias = lambda phi: np.concatenate((np.ones((phi.shape[0],1)), phi), axis=1)

        phi = rbf_kernel(states, self.means, gamma = self.rbf_gamma)
        if self.fea_normalization:
            phi = normalize(phi, norm="l1", axis=1)
        if self.feature_biased:
            phi = rbf_bias(phi)
        return phi

    def feature_visualize(self):

        number_point = 50
        low = np.min(self.means, axis=0)  # then don't need to normalize the input
        high = np.max(self.means, axis=0)
        s_len = self.means.shape[1]

        # visualization of feature function
        # for i in range(s_len):
        for i in range(1):

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


def main():
    test_rbf_gamma = 2
    test_mean = np.array([[1, 1, 1, 1],
                          [-1, -1, -1, -1],
                          # [0.5, 1],
                          ], dtype=float)
    test_act_n = 3

    rbf_basis = RBFBasis(rbf_gamma=test_rbf_gamma, means=test_mean, act_n=test_act_n,
                         fea_normalization=True, feature_biased=True)

    test_samples = np.array([
        [1, 2, 3, 4],
        [0, 0, 3, 4],
        [0.5, 0.5, 3, 4]
    ])

    a_indices = np.array([
        [0], [1], [2]
    ])

    print(rbf_basis.rbf(test_samples))

    print("enumarate", rbf_basis(test_samples))

    print("certain act", rbf_basis(test_samples, a_indices))

    low = np.ones(4)
    high = np.zeros(4)
    dis_list = [4]*4
    test_mean = build_means_discrete(low, high, dis_list)
    rbf_basis = RBFBasis(rbf_gamma=test_rbf_gamma, means=test_mean, act_n=test_act_n,
                         fea_normalization=True, feature_biased=True)
    rbf_basis.feature_visualize()
    return


if __name__ == "__main__":
    main()



