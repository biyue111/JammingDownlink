import math as math
from DDPG.ddpg import *
from DownlinkEnv import *
import configs as configs
from tqdm import tqdm
from utils.utilFunc import *
import matplotlib.pyplot as plt
import csv
from agents import *


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self):
        self.pick_times = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.overall_step = 0.000
        self.hundred_step = [0.000 for x in range(0, 100)]
        self.env = DownlinkEnv()

    def bs_pre_train(self, bs_agent, s):
        # Generate actions
        print("------------Begin pre-train------------")
        ch_step = 2.0 / (1.0 * configs.CHANNEL_NUM)  # The "wide" of a channel
        a_channels_bound = np.arange(-1, 1 + ch_step, ch_step)
        j = 0
        a_channels = np.zeros(configs.CHANNEL_NUM * 2)  # Channel upper and channel lower
        for i in range(len(a_channels_bound) - 1):  # One channel, two data points
            a_channels[j] = a_channels_bound[i] + ch_step / 10.0
            j += 1
            a_channels[j] = a_channels_bound[i + 1] - ch_step / 10.0
            j += 1
        print("a_channels:", a_channels)

        a_channel_list = np.array(np.meshgrid(a_channels, a_channels, a_channels)).T.reshape(-1, 3)
        jmr_a = np.zeros(configs.CHANNEL_NUM)
        bs_raw_actions = []
        rewards = []
        states = []
        for i in range(len(a_channel_list)):
            bs_raw_a = np.hstack((np.ones(configs.CHANNEL_NUM), a_channel_list[i]))
            bs_a = bs_agent.get_real_action(bs_raw_a)
            r, new_s = self.env.step([bs_a, jmr_a])
            bs_raw_actions.append(bs_raw_a)
            rewards.append(r)
            states.append(s)
        print("Pre-train input data list:")
        for i in range(len(a_channel_list)):
            print(states[i], bs_raw_actions[i], rewards[i])

        bs_agent.pre_train(np.array(states), np.array(bs_raw_actions), np.array(rewards))

    def run(self, bs_agent, jmr_agent):
        tqdm_e = tqdm(range(configs.UPDATE_NUM), desc='Score', leave=True, unit=" episodes")
        old_state = self.env.get_init_state()
        records = np.zeros(configs.UPDATE_NUM)

        jammed_flag_list = np.zeros(configs.UPDATE_NUM)
        reward_list = np.zeros(configs.UPDATE_NUM)
        state_records = np.zeros((configs.UPDATE_NUM, configs.USER_NUM * 4))
        power_allocation_records = np.zeros((configs.UPDATE_NUM, configs.CHANNEL_NUM))
        user_channel_choosing_records = np.zeros((configs.UPDATE_NUM, configs.USER_NUM))

        for e in range(configs.UPDATE_NUM):
            # BS Actor takes an action
            print("------- " + str(e) + " ---------")
            bs_raw_a_ls, bs_raw_a_ls_no_noise = bs_agent.act(old_state, e)
            bs_a_ls = bs_agent.get_real_action(bs_raw_a_ls)
            # Jammer takes an action
            jmr_a_ls = jmr_agent.act(old_state, e)

            # Retrieve new state, reward, and whether the state is terminal
            print([bs_a_ls, jmr_a_ls])
            jammed_flag, r, new_state = self.env.step([bs_a_ls, jmr_a_ls])
            jammed_flag_list[e] = jammed_flag
            reward_list[e] = r
            state_records[e] = new_state
            power_allocation_records[e] = bs_a_ls[0]
            user_channel_choosing_records[e] = bs_a_ls[1]

            # Add outputs to memory buffer
            if e > 0:
                if bs_agent.brain.buffer.count < bs_agent.brain.buffer.buffer_size or e % 30 == 0:
                    bs_agent.memorize(old_state, bs_raw_a_ls, r, new_state)

            """ Update using virtual data """
            # if e > 800:
            #     bs_virtual_raw_actions = bs_agent.get_virtual_actions(bs_raw_a_ls)
            #     v_rewards = np.zeros(len(bs_virtual_raw_actions))
            #     v_old_states = np.zeros((len(bs_virtual_raw_actions), configs.USER_NUM * 4))
            #     v_next_states = np.zeros((len(bs_virtual_raw_actions), configs.USER_NUM * 4))
            #     for k in range(len(bs_virtual_raw_actions)):
            #         v_old_states[k] = old_state
            #         v_bs_action = bs_agent.get_real_action(bs_virtual_raw_actions[k])
            #         v_jammed_flag, v_rewards[k], v_next_states[k] = self.env.bs_virtual_step([v_bs_action, jmr_a_ls])
            #     bs_agent.virtual_update_brain(v_old_states, bs_virtual_raw_actions, v_rewards, v_next_states)
            if e % 50 == 0:
                bs_agent.update_brain_channel_selection()

            # Update current state
            old_state = new_state

            # Show results
            records[e] = r
            print("Raw actions: ", bs_raw_a_ls)
            print("Actions: " + str(bs_a_ls))
            print("Reward: " + str(r))
        print("BS station buffer")
        bs_agent.brain.buffer.print_buffer()
        # ---------- Save models ----------
        bs_agent.save_brain("results/save_net.ckpt")
        # --------- Print results ---------
        # Write reward and success rate to a csv file
        csv_file = open('CHACNet Reward list.csv', 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(reward_list)
        csv_file.close()
        csv_file = open('CHACNet jammed flag list.csv', 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(jammed_flag_list)
        csv_file.close()
        csv_file = open('CHACNet state records.csv', 'w', newline='')
        writer = csv.writer(csv_file)
        for i in range(configs.USER_NUM * 4):
            state_record = [state_records[j][i] for j in range(configs.UPDATE_NUM)]
            writer.writerow(state_record)
        csv_file.close()
        csv_file = open('CHACNet power allocation records.csv', 'w', newline='')
        writer = csv.writer(csv_file)
        for i in range(configs.CHANNEL_NUM):
            power_allocation_record = [power_allocation_records[j][i] for j in range(configs.UPDATE_NUM)]
            writer.writerow(power_allocation_record)
        csv_file.close()
        csv_file = open('CHACNet channel choosing records.csv', 'w', newline='')
        writer = csv.writer(csv_file)
        for i in range(configs.USER_NUM):
            state_record = [state_records[j][i] for j in range(configs.UPDATE_NUM)]
            writer.writerow(state_record)
        csv_file.close()


# -------------------- MAIN ----------------------------
base_station_agent = BSAgent(1.0)
jammer_agent = JMRAgent()
env = Environment()
try:
    env.run(base_station_agent, jammer_agent)
finally:
    pass
