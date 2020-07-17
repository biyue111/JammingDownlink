import math as math
from Agents.abstract_agent import AbstractAgent
import configs as configs
from utils.utilFunc import *
from Agents.Q_learning.Q_learning_brain import *
import itertools


class QAgent(AbstractAgent):
    def __init__(self, act_range, act_dim, state_dim):
        # Create a ddpg network with
        # actions: [power allocation in each channel, channel chosen for each user]
        # states: [user position (x, y), channel chosen, data_rate] * number of user
        # action range: [0, 1]
        self.model_name = 'Q_learning_boltzmann'
        self.max_power = configs.BS_MAX_POWER
        self.act_range = act_range
        self.act_dim = act_dim
        self.state_dim = state_dim
        # print(self.act_dim, self.state_dim)
        self.state_id = 0
        self.action_id = 0
        self.reward = 0.0
        self.new_state_id = 0

        self.power_level_num = 13  # number of power level
        self.action_list = []
        self.generate_action_list()
        self.state_data_rate_level_num = 5  # number of data rate levels in state
        self.state_list = []
        self.generate_state_list()

        self.brain = QL(len(self.action_list), len(self.state_list), act_range)

    def generate_state_list(self):
        """
        [chosen channel, data rate level] * user_num
        data rate level = 0 if data rate < min data rate
        data rate level = 1 if min data rate <= data rate < 2 * min data rate
        data rate level = 2 if data rate >= 2 * min data rate
        ...
        """
        chosen_channel_list = []
        for chosen_channels in itertools.product(range(configs.CHANNEL_NUM), repeat=configs.USER_NUM):
            l_chosen_channels = [cc for cc in chosen_channels]
            chosen_channel_list.append(l_chosen_channels)
        data_rate_levels_list = []
        for data_rate_levels in itertools.product(range(self.state_data_rate_level_num), repeat=configs.USER_NUM):
            l_data_rates_levels = [dr for dr in data_rate_levels]
            data_rate_levels_list.append(l_data_rates_levels)

        for n in range(len(chosen_channel_list)):
            for m in range(len(data_rate_levels_list)):
                #  merge two list
                l = []
                for j in range(len(chosen_channel_list[n])):
                    l.append(chosen_channel_list[n][j])
                    l.append(data_rate_levels_list[m][j])
                self.state_list.append(l)
        print("state list: ", self.state_list)

    def generate_action_list(self):
        """
        One element of the actoin list: [power in each channel] + [channel selection of each user]
        """
        power_levels_list = []
        for accumulated_power_levels in itertools.combinations_with_replacement(
                range(self.power_level_num), configs.CHANNEL_NUM - 1):
            power_levels = [accumulated_power_levels[0]]
            for i in range(1, configs.CHANNEL_NUM- 1):
                power_levels.append(accumulated_power_levels[i] - accumulated_power_levels[i-1])
            power_levels.append(self.power_level_num - 1 - accumulated_power_levels[configs.CHANNEL_NUM - 2])
            real_power_levels = [configs.BS_MAX_POWER * power_level / (self.power_level_num-1)
                                 for power_level in power_levels]

            power_levels_list.append(real_power_levels)

        channel_selection_list = []
        for channel_selection in itertools.combinations_with_replacement(
                range(configs.CHANNEL_NUM), configs.USER_NUM):
            l_channel_selection = [e for e in channel_selection]
            channel_selection_list.append(l_channel_selection)

        # Combine two parts of action
        for p in power_levels_list:
            for c in channel_selection_list:
                self.action_list.append(p + c)

    def act(self, s, e):
        # Input: state, episode number
        # Output: action
        print("q_agent.act q_state:", self.get_q_state(s))
        a_i = self.brain.boltzmann_sampling(self.get_state_index(self.get_q_state(s)), e)
        print("q_agent.act action:", self.action_list[a_i])
        return self.action_list[a_i]

    def get_real_action(self, raw_a):
        """
        Turn the action into two lists: [power_allocation_ls, user_channel_ls]
        :param raw_a:
        :return:
        """
        print(raw_a)
        power_allocation_ls = raw_a[0:configs.CHANNEL_NUM]
        user_channel_ls = raw_a[configs.CHANNEL_NUM:configs.CHANNEL_NUM + configs.USER_NUM]
        return [power_allocation_ls, user_channel_ls]

    def update_brain(self, e):
        self.brain.update_q_table(self.state_id, self.action_id, self.reward, self.new_state_id)

    def memorize(self, state, a, r, new_state):
        """
        Memorize one sample and turn values into indexes
        :param state:
        :param a:
        :param r:
        :param new_state:
        """
        self.state_id = self.get_state_index(self.get_q_state(state))
        self.action_id = self.get_action_index(a)
        self.reward = r
        self.new_state_id = self.get_state_index(self.get_q_state(new_state))

    def get_q_state(self, s):
        """
        :param s: the state from Env [user_position_x, y, raw_channel_chosen, data_rate] * user num
        :return: [channel_chosen, data_rate_level]
        data rate level = 0 if data rate < min data rate
        data rate level = 1 if min data rate <= data rate < 2 * min data rate
        data rate level = 2 if data rate >= 2 * min data rate
        """
        q_s = []
        normalized_data_rate_threshild = configs.DATA_RATE_THRESHOLD / configs.MAX_DATARATE
        i = configs.CHANNEL_NUM  # pass jammer info
        for j in range(configs.USER_NUM):
            i += 2  # pass user position
            q_s.append(raw_channel_to_channel(s[i]))
            i += 1
            dr_level = math.floor(s[i] / normalized_data_rate_threshild)
            if dr_level > self.state_data_rate_level_num - 1:
                dr_level = self.state_data_rate_level_num - 1
            q_s.append(dr_level)
            i += 1
        return q_s

    def get_state_index(self, s):
        """
        Take a state and return the index of that state
        :param s:
        :return:
        """
        return self.state_list.index(s)

    def get_action_index(self, a):
        return self.action_list.index(a)

    def pre_train(self, states, bs_actions, rewards, a_channels):
        pass
