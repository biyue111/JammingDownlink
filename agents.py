import math as math
from DDPG.ddpg import *
import configs as configs
from utils.utilFunc import *
import csv


# -------------------- BS agent ------------------------
class BSAgent:
    def __init__(self, act_range, act_dim, state_dim):
        # Create a ddpg network with
        # actions: [power allocation in each channel, channel chosen for each user]
        # states: [user position (x, y), channel chosen, data_rate] * number of user
        # action range: [0, 1]
        self.max_power = configs.BS_MAX_POWER
        self.act_range = act_range
        self.act_dim = act_dim
        self.channel_raw_action_ls = configs.RAW_CHANNEL_LIST
        self.state_dim = state_dim
        # print(self.act_dim, self.state_dim)
        self.brain = DDPG(self.act_dim, self.state_dim, act_range)
        self.noise = OrnsteinUhlenbeckProcess(size=self.act_dim, n_steps_annealing=100)
        self.virtual_action_step = configs.VIRTUAL_ACTION_STEP

    def act(self, s, t):
        fix_power_flag = 0
        fix_channel_flag = 0
        # print("The state:" + str(s))
        print("## BS station action ##")
        a_no_noise_raw = self.brain.policy_action(s)
        a_no_noise = self.get_real_action(a_no_noise_raw)
        print("Action directly from brain:" + str(a_no_noise_raw))
        print("Real action directly from brain:" + str(a_no_noise))
        a_es_raw = self.brain.actor.action(s)
        print("action estimated directly from brain:" + str(a_es_raw))
        a_es = self.get_real_action(a_es_raw)
        print("Real a estimation directly from brain:" + str(a_es))
        v = self.brain.critic.target_q(np.expand_dims(s, axis=0), np.expand_dims(a_no_noise_raw, axis=0))
        print("Value of the chosen action:", v)
        # ------- Add Noise -------
        noise = self.noise.generate(t)
        print("Noise: ", noise)
        a = np.clip(a_no_noise_raw + noise, -self.act_range, self.act_range)
        a = self.brain.get_target_discrete_action(s, a)
        # ------- Fix action -------
        if fix_power_flag == 1:
            for i in range(configs.CHANNEL_NUM):
                a[i] = 1.0
        if fix_channel_flag == 1:
            step = 2.0 / (configs.CHANNEL_NUM * 1.0 - 1.0)
            action_value = -1.0
            # for i in range(configs.USER_NUM):
            #     a[i + configs.CHANNEL_NUM] = action_value
            #     action_value += step
            #     if action_value > 1.0:
            #         action_value = -1.0
            for i in range(configs.USER_NUM):
                a[i + configs.CHANNEL_NUM] = 0
        print("####")
        return a, a_no_noise_raw

    def get_real_action(self, raw_a):
        #  return "real" action: [power_allocation_ls, user_channel_ls]
        raw_power_allocation = raw_a[0:configs.CHANNEL_NUM]
        raw_user_channel_ls = raw_a[configs.CHANNEL_NUM:configs.CHANNEL_NUM + configs.USER_NUM]
        # Normalization
        raw_power_allocation = [(raw_power_allocation[i] + self.act_range) / (2 * self.act_range)
                                for i in range(configs.CHANNEL_NUM)]
        # raw_user_channel_ls = [(raw_user_channel_ls[i] + self.act_range) / (2 * self.act_range)
        #                        for i in range(configs.USER_NUM)]
        # Calculate the meaningful action (power and channel chosen)
        if sum(raw_power_allocation) == 0:
            power_allocation_ls = [self.max_power / configs.CHANNEL_NUM for _ in range(configs.CHANNEL_NUM)]
        else:
            power_allocation_ls = [raw_power_allocation[i] * self.max_power / sum(raw_power_allocation)
                                   for i in range(configs.CHANNEL_NUM)]
        user_channel_ls = [raw_channel_to_channel(raw_user_channel_ls[i]) for i in range(configs.USER_NUM)]
        # print("[Test] raw_user_channel_ls, user_channel_ls", raw_user_channel_ls, user_channel_ls)
        for i in range(configs.USER_NUM):
            if user_channel_ls[i] >= configs.CHANNEL_NUM:
                user_channel_ls[i] = configs.CHANNEL_NUM - 1
        return [power_allocation_ls, user_channel_ls]

    def update_brain(self, episode, jammed_flag_list, power_list, channel_list):
        needed_channel_selection = self.is_needed_update_channel_selection_actor(episode, jammed_flag_list)
        needed_power_allocation = self.is_needed_update_power_allocation_actor(episode, power_list, channel_list)
        if needed_channel_selection or needed_power_allocation:
            self.brain.train()

    def is_needed_update_channel_selection_actor(self, episode, jammed_flag_list):
        needed_flag = 0
        for i in range(10):
            if jammed_flag_list[episode - i] > 0:
                needed_flag = 1
        return needed_flag

    # def update_brain_channel_selection(self, episode, jammed_flag_list):
    #     actor_needed_update_flag = self.is_needed_update_channel_selection_actor(episode, jammed_flag_list)
    #     if self.brain.buffer.count > configs.BATCH_SIZE:
    #         if actor_needed_update_flag == 1:
    #             print("\033[0;33m[Info] Need channel selection actor update. \033[0m")
    #         self.brain.train_channel_selection_transfer(actor_needed_update_flag)

    def is_needed_update_power_allocation_actor(self, episode, power_list, channel_list):
        needed_flag = 0
        for i in range(10):
            p = episode - i
            useless_power = 0.0
            for c in range(configs.CHANNEL_NUM):
                channel_used = 0
                for u in range(configs.USER_NUM):
                    if channel_list[p][u] == c:
                        channel_used = 1
                if channel_used == 0:
                    useless_power += power_list[p][c]
            if useless_power * 100.0 >= configs.BS_MAX_POWER:
                needed_flag = 1
        return needed_flag

    # def virtual_update_brain(self, states, bs_actions, rewards, next_states):
    #     self.brain.virtual_train(states, bs_actions, rewards, next_states)

    def memorize(self, old_state, a, r, new_state):
        self.brain.memorize(old_state, a, r, new_state)

    def get_virtual_actions(self, real_raw_action):
        # Generate virtual actions
        channel_chosen = real_raw_action[configs.USER_NUM:(configs.USER_NUM + configs.CHANNEL_NUM)]
        pa_step = 1.0 * self.virtual_action_step  # The "wide" of a channel
        a_pa = np.arange(-1, 1 + pa_step, pa_step)

        a_pa_ls = np.array(np.meshgrid(a_pa, a_pa, a_pa)).T.reshape(-1, 3)  # TODO generalize
        virtual_actions = np.zeros((len(a_pa_ls), configs.USER_NUM + configs.CHANNEL_NUM))
        for i in range(len(a_pa_ls)):
            virtual_actions[i] = np.hstack((a_pa_ls[i], channel_chosen))
        # print("Number of Virtual actions: ", len(virtual_actions))
        # print("Virtual actions: ", virtual_actions)
        return virtual_actions

    def critic_test(self, s, file_name):
        #  Print critic network
        csv_file = open(file_name, 'w', newline='')
        writer = csv.writer(csv_file)
        a3 = np.arange(-1, 1, 0.05)
        raw_a3 = [(a3[i] + self.act_range) / (2 * self.act_range) for i in range(len(a3))]
        real_a3 = [math.floor(raw_a3[i] * configs.CHANNEL_NUM) for i in range(len(a3))]
        writer.writerow(a3)
        writer.writerow(real_a3)

        a1_idx = [0, 0, 1, 2]
        a2_idx = [0, 1, 1, 2]
        for k in range(len(a1_idx)):
            a1 = self.brain.discrete_action_ls[a1_idx[k]]
            a2 = self.brain.discrete_action_ls[a2_idx[k]]
            v = np.zeros(len(a3))
            t_v = np.zeros(len(a3))
            for i in range(len(a3)):
                a = [1, 1, 1, a1, a2, a3[i]]
                t_v[i] = self.brain.critic.target_q(np.expand_dims(s, axis=0), np.expand_dims(a, axis=0))
                v[i] = self.brain.critic.q_value(np.expand_dims(s, axis=0), np.expand_dims(a, axis=0))
            writer.writerow(t_v)
            writer.writerow(v)
            # print("The value of pairs: ", v)

    def actor_test(self, s):
        # s = [1.0, 1.0, 1.0, 3.65596417, -1.0, 1.0, 1.0, 3.65596417]
        raw_a = self.brain.actor.action(s)
        raw_a_t = self.brain.actor.target_action(s)
        a = self.get_real_action(raw_a)
        a_t = self.get_real_action(raw_a_t)
        print("Pre-train actor test: ", raw_a, a)
        print("Pre-train actor test: ", raw_a_t, a_t)
        v = self.brain.critic.target_q(np.expand_dims(s, axis=0), np.expand_dims(raw_a, axis=0))
        print("Pre-train actor test q-value: ", v)

    def save_brain(self, path):
        self.brain.save_session(path)

    def load_brain(self, path):
        self.brain.load_session(path)


# -------------------- JAMMER AGENT --------------------
class JMRAgent:
    def __init__(self):
        self.power = configs.JAMMER_POWER

    def act(self, s, t):
        a = np.zeros(configs.CHANNEL_NUM)
        jammed_channel = t % configs.CHANNEL_NUM  # 3-phase jammer
        # jammed_channel = t % (configs.CHANNEL_NUM - 1)  # 2-phase jammer
        # jammed_channel = 0  # fixed jammer
        a[jammed_channel] = 1.0
        a = a * self.power
        # print("Jammer's action: ", a)
        return a
