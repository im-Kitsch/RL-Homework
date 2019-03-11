import numpy as np


def build_means_discrete(low, high, state_discrete_list):
    low = low.flatten()
    high = high.flatten()
    # TODO, maybe not proper for the function?
    means = [np.linspace(start, stop, step) for start, stop, step in zip(low, high, state_discrete_list)]
    means = np.meshgrid(*means)
    means = [i.flatten() for i in means]
    means = np.array(means)
    means = means.T
    return means