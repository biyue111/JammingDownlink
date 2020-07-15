import numpy as np
import math
from utils.utilFunc import *

# General configs
CHANNEL_NUM = 3
UPDATE_NUM = 1000
SINR_THRESHOLD = 0.5
DATA_RATE_THRESHOLD = math.log2(1 + SINR_THRESHOLD)

RAW_CHANNEL_STEP = 2.0 / (CHANNEL_NUM * 1.0)
RAW_CHANNEL_LIST = np.arange(-1.0 + 0.5 * RAW_CHANNEL_STEP, 1.0, RAW_CHANNEL_STEP)

# BS  and users common configs
USER_NUM = 4
USER_POSITIONS = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
                           [0.9, 0.9], [-0.9, 0.9], [0.9, -0.9], [-0.9, -0.9]])
POSITION_RANGE = 1.0
BATCH_SIZE = 200
BEGIN_TRAINING_EPISONDE = 50
BS_MAX_POWER = 10
VIRTUAL_ACTION_STEP = 0.2

# Jammer common configs
JAMMER_POWER = 30.0
JAMMER_POSITION = [0.0, 0.0]

# Channel condition configs
PATHLOSS_FACTOR = 2
B2U_PATHLOSS = np.zeros(USER_NUM)
for i in range(USER_NUM):
    distance = np.sqrt(np.square(USER_POSITIONS[i][0]) + np.square(USER_POSITIONS[i][1]))
    B2U_PATHLOSS[i] = np.power(distance, -PATHLOSS_FACTOR)
J2U_PATHLOSS = np.zeros(USER_NUM)
for i in range(USER_NUM):
    distance = np.sqrt(np.square(USER_POSITIONS[i][0] - JAMMER_POSITION[0])
                       + np.square(USER_POSITIONS[i][1] - JAMMER_POSITION[1]))
    J2U_PATHLOSS[i] = np.power(distance, -PATHLOSS_FACTOR)

print("B2U_PATHLOSS", B2U_PATHLOSS[0], B2U_PATHLOSS[1])
print("J2U_PATHLOSS", J2U_PATHLOSS[0], J2U_PATHLOSS[1])
NOISE = 0.01

MAX_SINR = 0
MAX_DATARATE = 0
for i in range(USER_NUM):
    sinr, datarate = calculate_datarate(BS_MAX_POWER, jammer_power=0.0,
                                        user_num_in_channel=1.0, user_id=i)
    MAX_DATARATE = max(MAX_DATARATE, datarate)
    MAX_SINR = max(MAX_SINR, sinr)
