from Agents.abstract_agent import AbstractAgent
import configs as configs
from utils.utilFunc import *


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