import configs as configs
import numpy as np
from utils.utilFunc import *


class DownlinkEnv:
    def __init__(self):
        self.user_positions = configs.USER_POSITIONS
        self.jammer_position = configs.JAMMER_POSITION
        self.position_range = configs.POSITION_RANGE
        self.channel_num = configs.CHANNEL_NUM
        self.user_num = configs.USER_NUM
        self.bs_max_power = configs.BS_MAX_POWER
        self.max_user_datarate = self.get_max_datarate()
        print("Max_user_datarate: ", self.max_user_datarate)
        self.sinr_threshold = configs.SINR_THRESHOLD  # if higher than that, this channel is occupied by jammer

    def get_init_state(self):
        ini_jammer_ch = np.zeros(self.channel_num)
        r = np.zeros(self.user_num)
        ini_ch = [(self.channel_num - 1) for _ in range(self.user_num)]
        s = self.generate_state(ini_jammer_ch, r, ini_ch)
        return s

    def get_max_datarate(self):
        max_sinr = 0
        max_datarate = 0
        for i in range(self.user_num):
            sinr, datarate = calculate_datarate(self.bs_max_power, jammer_power=0.0,
                                                user_num_in_channel=1.0, user_id=i)
            max_datarate = max(max_datarate, datarate)
            max_sinr = max(max_sinr, sinr)
        return max_datarate

    def generate_state(self, jammer_power_allocation, ch, r):
        s = np.zeros(self.channel_num + 4 * self.user_num)
        # Write jammer jammed channel
        i = 0
        for j in range(self.channel_num):
            s[i] = jammer_power_allocation[j] / configs.JAMMER_POWER
            i += 1
        # Write user's position (x, y), channel chosen and data rate to the state
        for j in range(self.user_num):
            s[i] = self.user_positions[j][0] / self.position_range
            i += 1
            s[i] = self.user_positions[j][1] / self.position_range
            i += 1
            s[i] = channel_to_raw_channel(int(ch[j]))
            i += 1
            s[i] = r[j] / self.max_user_datarate
            i += 1
        return s

    def calculate_bs_reward(self, bs_power_allocation, user_channel_ls, jammer_power_allocation):
        user_num_ls = np.zeros(self.channel_num)
        datarate_ls = np.zeros(self.user_num)
        sinr_ls = np.zeros(self.user_num)
        jammed_penalty = -5.0
        useless_power_penalty = -1.0
        min_reward = -2.0
        congested_user_reward_offset = 0.0
        # channel_availability[i][j] = 1: channel i is available for user j
        channel_availability = np.ones((self.channel_num, self.user_num))
        # calculate number of user in one channel
        for i in range(self.user_num):
            user_num_ls[user_channel_ls[i]] += 1

        # Check having congested user or not (if found a jammed user, no need to check)
        # congested_user_num = 0
        # if jammed_flag == 0:
        #     for i in range(self.user_num):
        #         congested_flag = 0
        #         if user_num_ls[user_channel_ls[i]] > 1:
        #             for j in range(self.channel_num):
        #                 if user_num_ls[j] == 0 and channel_availability[j][i] == 1:
        #                     congested_flag = 1
        #         congested_user_num += congested_flag

        # Calculate data rate (if found a a jammed user or having idle channel, no nned to calculate)
        for i in range(self.user_num):
            sinr_ls[i], datarate_ls[i] = calculate_datarate(bs_power_allocation[user_channel_ls[i]],
                                                        jammer_power_allocation[user_channel_ls[i]],
                                                        user_num_ls[user_channel_ls[i]], i)

        # Check choosing jammed channel or not
        jammed_flag = 0
        for i in range(self.user_num):
            if sinr_ls[i] < self.sinr_threshold and jammer_power_allocation[user_channel_ls[i]] > 0:
                jammed_flag += 1

        reward = 0
        if jammed_flag > 0:
            reward = jammed_penalty * jammed_flag
        # elif congested_user_num > 0:
        #     reward = congested_user_reward_offset - congested_user_num
        # useless_power = 0.0
        # for i in range(self.channel_num):
        #     channel_no_user_flag = 1
        #     for j in range(self.user_num):
        #         if user_channel_ls[j] == i:
        #             channel_no_user_flag = 0
        #     if channel_no_user_flag == 1:
        #         useless_power += bs_power_allocation[i]
        # reward += useless_power * useless_power_penalty

        reward += sum(datarate_ls)  # TODO: fairness
        reward = max(min_reward, reward)
        # print("data rate list:" + str(datarate_ls))

        return jammed_flag, reward, datarate_ls

    def bs_virtual_step(self, action_ls):
        return self.step(action_ls)

    def step(self, action_ls):
        """
        :param action_ls: [bs_action_ls, jammer_action_ls]
        bs_actions_ls: [power_allocation, user_channel]:
            the power allocation for each channel and a list of user channel chosen
        jammer_power_allocation: jammer power allocation
        :return reward: the reward for BS
        datarate_ls: the data rate list for each user
        """
        bs_action_ls = action_ls[0]
        jammer_power_allocation = action_ls[1]
        bs_power_allocation = bs_action_ls[0]
        user_channel_ls = bs_action_ls[1]

        jammed_flag, reward, datarate_ls = self.calculate_bs_reward(bs_power_allocation, user_channel_ls, jammer_power_allocation)
        s_ = self.generate_state(jammer_power_allocation, user_channel_ls, datarate_ls)

        return jammed_flag, reward, s_
