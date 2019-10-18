import numpy as np
import math

# General configs
CHANNEL_NUM = 3
UPDATE_NUM = 600
SINR_THRESHOLD = 0.5

# BS  and users common configs
USER_NUM = 3
USER_POSITIONS = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
BATCH_SIZE = 499
BS_MAX_POWER = 10
VIRTUAL_ACTION_STEP = 0.2

# Jammer common configs
JAMMER_POWER = 10.0
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


def calculate_datarate(bs_power, jammer_power, user_num_in_channel, user_id):
    """
    :param bs_power:
    :param jammer_power:
    :param user_id:
    :return:
    """
    interference = jammer_power * J2U_PATHLOSS[user_id] + NOISE
    sinr = bs_power * B2U_PATHLOSS[user_id] / interference
    data_rate = math.log2(1 + sinr) / user_num_in_channel
    return sinr, data_rate
