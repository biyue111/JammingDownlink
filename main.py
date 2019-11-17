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
    def __init__(self, bs_state_dim, bs_act_dim):
        self.pick_times = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.overall_step = 0.000
        self.hundred_step = [0.000 for x in range(0, 100)]
        self.env = DownlinkEnv()
        self.bs_state_dim = bs_state_dim
        self.bs_act_dim = bs_act_dim
        self.pre_train_bs_raw_actions = []
        self.pre_train_rewards = []
        self.pre_train_states = []

    def bs_pre_train(self, bs_agent, s):
        # Generate actions
        print("------------Begin pre-train------------")
        ch_step = 2.0 / (1.0 * configs.CHANNEL_NUM)  # The "wide" of a channel
        a_channels = configs.RAW_CHANNEL_LIST
        a_channel_list = np.array(np.meshgrid(a_channels, a_channels, a_channels)).T.reshape(-1, 3)
        power_step = 0.2
        a_powers = np.arange(-1.0, 1 + power_step, power_step)
        a_power_list = np.array(np.meshgrid(a_powers, a_powers, a_powers)).T.reshape(-1, 3)
        jmr_a = np.zeros(configs.CHANNEL_NUM)
        for i in range(len(a_power_list)):
            for j in range(len(a_channel_list)):
                bs_raw_a = np.hstack((a_power_list[i], a_channel_list[j]))
                bs_a = bs_agent.get_real_action(bs_raw_a)
                jamming_flag, r, new_s = self.env.step([bs_a, jmr_a])
                self.pre_train_bs_raw_actions.append(bs_raw_a)
                self.pre_train_rewards.append(r)
                self.pre_train_states.append(s)
        a_channel_test_list = [[a_channels[0], a_channels[1], a_channels[1]],
                               [a_channels[0], a_channels[2], a_channels[0]],
                               [a_channels[2], a_channels[1], a_channels[1]]]
        # bs_agent.pre_train(np.array(self.pre_train_states), np.array(self.pre_train_bs_raw_actions),
        #                    np.array(self.pre_train_rewards), np.array(a_channel_list))
        bs_agent.pre_train(np.array(self.pre_train_states), np.array(self.pre_train_bs_raw_actions),
                           np.array(self.pre_train_rewards), np.array(a_channel_test_list))

    def run(self, bs_agent, jmr_agent):
        # tqdm_e = tqdm(range(configs.UPDATE_NUM), desc='Score', leave=True, unit=" episodes")
        # print(configs.RAW_CHANNEL_LIST)
        old_state = self.env.get_init_state()
        records = np.zeros(configs.UPDATE_NUM)

        jammed_flag_list = np.zeros(configs.UPDATE_NUM)
        reward_list = np.zeros(configs.UPDATE_NUM)
        state_records = np.zeros((configs.UPDATE_NUM, bs_agent.state_dim))
        power_allocation_records = np.zeros((configs.UPDATE_NUM, configs.CHANNEL_NUM))
        user_channel_choosing_records = np.zeros((configs.UPDATE_NUM, configs.USER_NUM))

        self.bs_pre_train(bs_agent, old_state)
        for e in range(configs.UPDATE_NUM):
            # BS Actor takes an action
            print("------- " + str(e) + " ---------")
            bs_raw_a_ls, bs_raw_a_ls_no_noise = bs_agent.act(old_state, e)
            bs_a_ls = bs_agent.get_real_action(bs_raw_a_ls)
            # Jammer takes an action
            jmr_a_ls = jmr_agent.act(old_state, e)

            # Retrieve new state, reward, and whether the state is terminal
            jammed_flag, r, new_state = self.env.step([bs_a_ls, jmr_a_ls])
            jammed_flag_list[e] = jammed_flag
            reward_list[e] = r
            state_records[e] = new_state
            power_allocation_records[e] = bs_a_ls[0]
            user_channel_choosing_records[e] = bs_a_ls[1]

            # Show actions
            records[e] = r
            info_color = "0;32m"
            print("\033["+info_color+"[Info] Jammer action: \033[0m", jmr_a_ls)
            print("\033["+info_color+"[Info] BS Raw actions: \033[0m", [round(k, 4) for k in bs_raw_a_ls])
            print("\033["+info_color+"[Info] BS Actions: \033[0m" + str(bs_a_ls))
            print("\033["+info_color+"[Info] Reward: \033[0m" + str(r))
            # Add outputs to memory buffer
            if e > 0:
                if bs_agent.brain.buffer.count < bs_agent.brain.buffer.buffer_size or e % 1 == 0:
                    bs_agent.memorize(old_state, bs_raw_a_ls, r, new_state)
                bs_agent.brain.last_10_buffer.memorize(old_state, bs_raw_a_ls, r, new_state)

            if e % 5 == 0 and e > configs.BEGIN_TRAINING_EPISONDE:
                bs_agent.update_brain(e, jammed_flag_list, power_allocation_records, user_channel_choosing_records)
            """ Update using virtual data """
            # if e > 701:
            #     bs_virtual_raw_actions = bs_agent.get_virtual_actions(bs_raw_a_ls)
            #     v_rewards = np.zeros(len(bs_virtual_raw_actions))
            #     v_old_states = np.zeros((len(bs_virtual_raw_actions), bs_agent.state_dim))
            #     v_next_states = np.zeros((len(bs_virtual_raw_actions), bs_agent.state_dim))
            #     for k in range(len(bs_virtual_raw_actions)):
            #         v_old_states[k] = old_state
            #         v_bs_action = bs_agent.get_real_action(bs_virtual_raw_actions[k])
            #         v_jammed_flag, v_rewards[k], v_next_states[k] = self.env.bs_virtual_step([v_bs_action, jmr_a_ls])
            #     bs_agent.virtual_update_brain(v_old_states, bs_virtual_raw_actions, v_rewards, v_next_states)

            # Update current state
            old_state = new_state

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
        for i in range(bs_agent.state_dim):
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
bs_state_dim = configs.CHANNEL_NUM + 4 * configs.USER_NUM
bs_act_dim = configs.CHANNEL_NUM + configs.USER_NUM
env = Environment(bs_state_dim=bs_state_dim, bs_act_dim=bs_act_dim)
base_station_agent = BSAgent(act_range=1.0, state_dim=bs_state_dim, act_dim=bs_act_dim)
jammer_agent = JMRAgent()
try:
    env.run(base_station_agent, jammer_agent)
finally:
    pass
