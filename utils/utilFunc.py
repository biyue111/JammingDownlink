import numpy as np


def combination(a_element_list, depth):
    if depth >= len(a_element_list)-1:
        return np.reshape(a_element_list[depth], (len(a_element_list[depth]), 1))
    else:
        com_list = combination(a_element_list, depth + 1)
        result_list = []
        for i in range(len(a_element_list[depth])):
            new_line = a_element_list[depth][i] * np.ones((len(com_list), 1))
            tmp_list = np.hstack((new_line, com_list))
            if i == 0:
                result_list = tmp_list
            else:
                result_list = np.vstack((result_list, tmp_list))
        return result_list


class OrnsteinUhlenbeckProcess(object):
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """
    def __init__(self, theta=0.15, mu=0, sigma=1.5, x0=0, dt=0.5, n_steps_annealing=1000, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + \
            sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        # x = sigma * np.random.normal(size=self.size)
        self.x0 = x
        return x
