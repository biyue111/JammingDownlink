import math as math
from DDPG.ddpg import *
from DownlinkEnv import *
import configs as configs
from tqdm import tqdm
from utils.utilFunc import OrnsteinUhlenbeckProcess
import matplotlib.pyplot as plt
import csv

# -------------------- BS agent ------------------------
class BSAgent:
    def __init__(self, act_range):
        # Create a ddpg network with
        # actions: [power allocation in each channel, channel chosen for each user]
        # states: [user position (x, y), channel chosen, data_rate] * number of user
        # action range: [0, 1]
        self.max_power = configs.BS_MAX_POWER
        self.act_range = act_range
        self.act_dim = configs.CHANNEL_NUM + configs.USER_NUM
        self.state_dim = 4 * configs.USER_NUM
        # print(self.act_dim, self.state_dim)
        self.brain = DDPG(self.act_dim, self.state_dim, act_range)
        self.noise = OrnsteinUhlenbeckProcess(size=self.act_dim, n_steps_annealing=1000)

    def act(self, s, t):
        fix_power_flag = 0
        fix_channel_flag = 0
        # print("The state:" + str(s))
        a = self.brain.policy_action(s)
        # Clip continuous values to be valid w.r.t. environment
        print("a directly from brain:" + str(a))
        a_es_raw = self.brain.actor.action(s)
        print("a estimation directly from brain:" + str(a_es_raw))
        a_es = self.get_real_action(a_es_raw)
        print("Real a estimation directly from brain:" + str(a_es))
        v = self.brain.critic.target_q(np.expand_dims(s, axis=0), np.expand_dims(a, axis=0))
        print("Value of the chosen action:", v)
        noise = self.noise.generate(t)
        print("Noise: ", noise)
        a = np.clip(a + noise, -self.act_range, self.act_range)
        if fix_power_flag == 1:
            for i in range(configs.CHANNEL_NUM):
                a[i] = 0.5
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

        return a

    def get_real_action(self, raw_a):
        #  return "real" action: [power_allocation_ls, user_channel_ls]
        raw_power_allocation = raw_a[0:configs.CHANNEL_NUM]
        raw_user_channel_ls = raw_a[configs.CHANNEL_NUM:configs.CHANNEL_NUM+configs.USER_NUM]
        # Normalization
        raw_power_allocation = [(raw_power_allocation[i] + self.act_range)/(2*self.act_range)
                                for i in range(configs.CHANNEL_NUM)]
        raw_user_channel_ls = [(raw_user_channel_ls[i] + self.act_range)/(2*self.act_range)
                               for i in range(configs.USER_NUM)]
        # Calculate the meaningful action (power and channel chosen)
        if sum(raw_power_allocation) == 0:
            power_allocation_ls = [self.max_power / configs.CHANNEL_NUM for _ in range(configs.CHANNEL_NUM)]
        else:
            power_allocation_ls = [raw_power_allocation[i] * self.max_power / sum(raw_power_allocation)
                                   for i in range(configs.CHANNEL_NUM)]
        user_channel_ls = [math.floor(raw_user_channel_ls[i] * configs.CHANNEL_NUM) for i in range(configs.USER_NUM)]
        for i in range(configs.USER_NUM):
            if user_channel_ls[i] >= configs.CHANNEL_NUM:
                user_channel_ls[i] = configs.CHANNEL_NUM - 1
        return [power_allocation_ls, user_channel_ls]

    def update_brain(self):
        if self.brain.buffer.count > configs.BATCH_SIZE:
            self.brain.train()

    def memorize(self, old_state, a, r, new_state):
        self.brain.memorize(old_state, a, r, new_state)

    def pre_train(self, states, bs_actions, rewards):
        # Generate pre-train data
        self.brain.pre_train(states, bs_actions, rewards, states)

    def critic_test(self, s, file_name):
        #  Print critic network
        csv_file = open(file_name, 'w', newline='')
        writer = csv.writer(csv_file)
        a3 = np.arange(-1, 1, 0.05)
        raw_a3 = [(a3[i] + self.act_range)/(2*self.act_range) for i in range(len(a3))]
        real_a3 = [math.floor(raw_a3[i] * configs.CHANNEL_NUM) for i in range(len(a3))]
        writer.writerow(a3)
        writer.writerow(real_a3)

        a1 = -1
        a2 = -1
        v = np.zeros(len(a3))
        t_v = np.zeros(len(a3))
        for i in range(len(a3)):
            a = [1, 1, 1, a1, a2, a3[i]]
            t_v[i] = self.brain.critic.target_q(np.expand_dims(s, axis=0), np.expand_dims(a, axis=0))
            v[i] = self.brain.critic.q_value(np.expand_dims(s, axis=0), np.expand_dims(a, axis=0))
        writer.writerow(t_v)
        writer.writerow(v)
        # print("The value of pairs: ", v)

        a1 = -1
        a2 = 0
        v = np.zeros(len(a3))
        for i in range(len(a3)):
            a = [1, 1, 1, a1, a2, a3[i]]
            t_v[i] = self.brain.critic.target_q(np.expand_dims(s, axis=0), np.expand_dims(a, axis=0))
            v[i] = self.brain.critic.q_value(np.expand_dims(s, axis=0), np.expand_dims(a, axis=0))
        writer.writerow(t_v)
        writer.writerow(v)
        # print("The value of pairs: ", v)
        # csv_file.close()

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


# -------------------- JAMMER AGENT --------------------
class JMRAgent:
    def __init__(self):
        self.power = configs.JAMMER_POWER

    def act(self, s, t):
        a = np.zeros(configs.CHANNEL_NUM)
        a[0] = 1.0
        a = a * self.power * 10
        print("Jammer's action: ", a)
        return a


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
            a_channels[j] = a_channels_bound[i+1] - ch_step / 10.0
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

        # self.bs_pre_train(bs_agent, old_state)
        # bs_agent.critic_test(old_state, 'critic_pre_train_test.csv')
        # bs_agent.actor_test(old_state)

        reward_list = np.zeros(configs.UPDATE_NUM)

        for e in range(configs.UPDATE_NUM):
            # print("e:", e)
            # bs_distribution = bs_agent.brain.get_distribution(s)
            # print("bs_distribution: ", bs_distribution)
            # jammer_distribution = jammer_agent. brain.get_distribution(s)
            # print("jammer_distribution: ", jammer_distribution)
            # bs_r = self.env.step([action_ls])

            # BS Actor takes an action
            print("------- " + str(e) + " ---------")
            bs_raw_a_ls = bs_agent.act(old_state, e)
            bs_a_ls = bs_agent.get_real_action(bs_raw_a_ls)
            # Jammer takes an action
            jmr_a_ls = jmr_agent.act(old_state, e)

            # Retrieve new state, reward, and whether the state is terminal
            print([bs_a_ls, jmr_a_ls])
            r, new_state = self.env.step([bs_a_ls, jmr_a_ls])
            reward_list[e] = r
            # print("New state: ", new_state)
            # Add outputs to memory buffer
            if e > 0:
                bs_agent.memorize(old_state, bs_raw_a_ls, r, new_state)
            # update the agent
            if e % 100 == 0:
                bs_agent.update_brain()

            # Update current state
            old_state = new_state

            # Test BS agent critic network
            # if e >= 500:
            #     bs_agent.critic_test(old_state)
                # bs_agent.actor_test()

            # Show results
            records[e] = r
            print("Raw actions: ", bs_raw_a_ls)
            print("Actions: " + str(bs_a_ls))
            print("Reward: " + str(r))
            # tqdm_e.set_description("Actions:" + str(bs_a_ls) + "Reward: " + str(r))
            # tqdm_e.refresh()

        bs_agent.critic_test(old_state, 'critic_test.csv')
        # Write reward to a csv file
        csv_file = open('Reword list', 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(reward_list)
        csv_file.close()


# -------------------- MAIN ----------------------------

# state_count = configs.STATE_CNT  # env.env.observation_space.shape[0]
# action_count = configs.ACTION_CNT  # env.env.action_space.n

# bs_agent = LFAQBSAgentPowerAllocation(state_count, action_count,
#                                       configs.BS_LFAQ_TEMPERATURE, configs.BS_LFAQ_MIN_TEMPERATURE,
#                                       configs.BS_LFAQ_ALPHA, configs.BS_LFAQ_BETA, configs.BS_LFAQ_GAMMA,
#                                       configs.BS_LFAQ_KAPPA, configs.BS_LFAQ_UPDATE_LOOPS)
# jammer_agent = LFAQJammerAgent(state_count, action_count,
#                                configs.JMR_LFAQ_TEMPERATURE, configs.JMR_LFAQ_MIN_TEMPERATURE,
#                                configs.JMR_LFAQ_ALPHA, configs.JMR_LFAQ_BETA, configs.JMR_LFAQ_GAMMA,
#                                configs.JMR_LFAQ_KAPPA, configs.JMR_LFAQ_UPDATE_LOOPS, configs.JMR_POWER_FACTOR)
base_station_agent = BSAgent(1.0)
jammer_agent = JMRAgent()
env = Environment()
try:
    env.run(base_station_agent, jammer_agent)
finally:
    pass
