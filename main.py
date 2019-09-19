import math as math
from DDPG.ddpg import *
from DownlinkEnv import *
import configs as configs
from tqdm import tqdm
from utils.utilFunc import OrnsteinUhlenbeckProcess
import matplotlib.pyplot as plt


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
        self.noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

    def act(self, s, t):
        # print("The state:" + str(s))
        a = self.brain.policy_action(s)
        # Clip continuous values to be valid w.r.t. environment
        print("a directly from brain:" + str(a))
        v = self.brain.critic.target_predict([np.expand_dims(s, axis=0), np.expand_dims(a, axis=0)])
        print("Value of action:", v)
        a = np.clip(a + self.noise.generate(t), -self.act_range, self.act_range)
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

    def critic_test(self, s):
        a_c = 0.7
        a = np.array([[1, 1, 1, -a_c, -a_c], [1, 1, 1, -a_c, a_c],
                      [1, 1, 1, a_c, -a_c], [1, 1, 1, a_c, a_c]])
        v = np.zeros(len(a))
        for i in range(len(a)):
            v[i] = self.brain.critic.target_predict([np.expand_dims(s, axis=0), np.expand_dims(a[i], axis=0)])
        print("The value of pairs: ", v)


# -------------------- JAMMER AGENT --------------------
class JMRAgent:
    def __init__(self):
        self.power = configs.JAMMER_POWER

    def act(self, s, t):
        a = self.power * np.zeros(configs.CHANNEL_NUM)
        return a


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self):
        self.pick_times = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.overall_step = 0.000
        self.hundred_step = [0.000 for x in range(0, 100)]
        self.env = DownlinkEnv()

    def run(self, bs_agent, jmr_agent):
        tqdm_e = tqdm(range(configs.UPDATE_NUM), desc='Score', leave=True, unit=" episodes")
        old_state = self.env.get_init_state()
        records = np.zeros(configs.UPDATE_NUM)
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
            r, new_state = self.env.step([bs_a_ls, jmr_a_ls])
            # Add outputs to memory buffer
            if e > 0:
                bs_agent.memorize(old_state, bs_raw_a_ls, r, new_state)
            # update the agent
            bs_agent.update_brain()

            # Update current state
            old_state = new_state

            # Test BS agent critic network
            if e >= 500:
                bs_agent.critic_test(old_state)

            # Show results
            records[e] = r
            print("Raw actions: ", bs_raw_a_ls)
            print("Actions: " + str(bs_a_ls))
            print("Reward: " + str(r))
            # tqdm_e.set_description("Actions:" + str(bs_a_ls) + "Reward: " + str(r))
            # tqdm_e.refresh()

        plt.plot(records)


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
