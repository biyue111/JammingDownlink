import numpy as np

import configs as configs
from utils.utilFunc import *


class QL:
    def __init__(self, act_num, state_num, act_range, gamma=0.01, lr=0.85, temp=15.0):
        # Some parameters
        self.gamma = gamma  # discount factor
        self.alpha = lr  # learning rate
        self.T = temp  # Temperature of Boltzmann_sampling
        self.min_T = 0.05
        self.epsilon = 0.1

        self.q_table = np.ones((state_num, act_num))
        self.action_num = act_num  # number of actions
        self.state_num = state_num  # number of states

    def update_q_table(self, s_i, a_i, r, s_i_):
        """
        :param s_i: current state's index
        :param a_i: action's index
        :param r: reward
        :param s_i_: next state's index
        :return:
        """
        max_q = max(self.q_table[s_i_])
        self.q_table[s_i, a_i] = (1 - self.alpha) * (self.q_table[s_i, a_i]) \
                                 + self.alpha * (r + self.gamma * max_q)

    # def epsilon_greedy_sampling(self,s_i):

    def boltzmann_sampling(self, s_i, e):
        temp = max(self.min_T, self.T - (e / 100))
        print(temp)

        prob_list = np.exp(self.q_table[s_i] / temp) / sum(np.exp(self.q_table[s_i] / temp))
        chosen_a = np.random.choice(self.action_num, p=prob_list)
        return chosen_a

    # def epsilon_greedy_sampling(self, s_i, e):

