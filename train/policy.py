import numpy as np


def greedy(values):
    return np.argmax(values)


# See https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py
def epsilon_greedy(values, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.randint(0, len(values))
    else:
        return greedy(values)
