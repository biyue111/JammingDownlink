import configs as configs
import numpy as np


class DownlinkEnv:
    def __init__(self):
        self.user_positions = configs.USER_POSITIONS
        self.jammer_position = configs.JAMMER_POSITION
        self.channel_num = configs.CHANNEL_NUM
        self.user_num = configs.USER_NUM

    def generate_state(self, ch, r):
        s = np.zeros(4 * self.user_num)
        # Write user's position (x, y), channel chosen and data rate to the state
        for i in range(self. user_num):
            s[i*4] = self.user_positions[i][0]
            s[i*4 + 1] = self.user_positions[i][1]
            s[i*4 + 2] = ch[i]
            s[i*4 + 3] = r[i]
        return s

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
        power_allocation = bs_action_ls[0]
        user_channel_ls = bs_action_ls[1]

        user_num_ls = np.zeros(self.channel_num)
        datarate_ls = np.zeros(self.user_num)
        # calculate number of user in one channel
        for i in range(self.user_num):
            user_num_ls[user_channel_ls[i]] += 1
        for i in range(self.user_num):
            datarate_ls[i] = configs.calculate_datarate(power_allocation[user_channel_ls[i]],
                                                        jammer_power_allocation[user_channel_ls[i]],
                                                        user_num_ls[user_channel_ls[i]], i)  # TODO

        reward = sum(datarate_ls)  # TODO: fairness
        s_ = self.generate_state(user_channel_ls, datarate_ls)
        return reward, s_
