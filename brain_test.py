from agents import *
from DownlinkEnv import *

# -------------------- MAIN ----------------------------
env = DownlinkEnv()
bs_agent = BSAgent(1.0)
jammer_agent = JMRAgent()
bs_agent.load_brain("results/save_net.ckpt")
old_state = env.get_init_state()
bs_raw_a_ls, bs_raw_a_ls_no_noise = bs_agent.act(old_state, 1000)
bs_a_ls = bs_agent.get_real_action(bs_raw_a_ls)
# Jammer takes an action
jmr_a_ls = jammer_agent.act(old_state, 1)
# Retrieve new state, reward, and whether the state is terminal
jammed_flag, r, new_state = env.step([bs_a_ls, jmr_a_ls])
old_state = new_state

print("------- " + "Test" + " ---------")
bs_agent.critic_test(np.array([1.0, 1.0, 2.0, 7.380375, -1.0, 1.0, 2.0, 3.691388, 1.0, -1.0, 1.0, 7.386889]), 'critic_test.csv')
