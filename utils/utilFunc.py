import numpy as np
import math
import configs as configs


def channel_to_raw_channel(channel):
    return configs.RAW_CHANNEL_LIST[channel]


def raw_channel_to_channel(raw_channel):
    # Used in BSAgent.get_real_action
    min_dis = configs.RAW_CHANNEL_STEP
    min_dis_idx = 0
    for i in range(len(configs.RAW_CHANNEL_LIST)):
        dis = abs(raw_channel - configs.RAW_CHANNEL_LIST[i])
        if dis <= min_dis:
            min_dis_idx = i
            min_dis = dis
    return min_dis_idx



def raw_channel_to_2raw_channels(raw_channel):
    # There are two raw channels possible
    # used in act
    tmp_element = np.array([configs.RAW_CHANNEL_LIST[0], configs.RAW_CHANNEL_LIST[0]])
    if raw_channel < configs.RAW_CHANNEL_LIST[0]:
        tmp_element = np.array([configs.RAW_CHANNEL_LIST[0], configs.RAW_CHANNEL_LIST[0]])
    elif raw_channel > configs.RAW_CHANNEL_LIST[-1]:
        tmp_element = np.array([configs.RAW_CHANNEL_LIST[-1], configs.RAW_CHANNEL_LIST[-1]])
    else:
        for j in range(configs.CHANNEL_NUM - 1):
            if configs.RAW_CHANNEL_LIST[j] <= raw_channel <= configs.RAW_CHANNEL_LIST[j + 1]:
                tmp_element = np.array([configs.RAW_CHANNEL_LIST[j], configs.RAW_CHANNEL_LIST[j + 1]])
    return tmp_element


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


def calculate_datarate(bs_power, jammer_power, user_num_in_channel, user_id):
    """
    :param bs_power:
    :param jammer_power:
    :param user_id:
    :return:
    """
    interference = jammer_power * configs.J2U_PATHLOSS[user_id] + configs.NOISE
    sinr = bs_power * configs.B2U_PATHLOSS[user_id] / interference
    data_rate = math.log2(1 + sinr) / user_num_in_channel
    return sinr, data_rate


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
