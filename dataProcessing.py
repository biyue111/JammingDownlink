import csv
import configs as configs
from utils.utilFunc import *


def get_max_datarate():
    # Copy from DownlinkEnv.py
    max_sinr = 0
    max_datarate = 0
    for i in range(configs.USER_NUM):
        sinr, datarate = calculate_datarate(configs.BS_MAX_POWER, jammer_power=0.0,
                                            user_num_in_channel=1.0, user_id=i)
        max_datarate = max(max_datarate, datarate)
        max_sinr = max(max_sinr, sinr)
    return max_datarate


# -------------------- MAIN ----------------------------
scenario_name = 'downlink_basic'
model_name = 'Q_learning_boltzmann'
experiment_num = 10

for e in range(experiment_num):
    # calculate success_rate
    io_prefix = 'temp_results/' + scenario_name + '/' + model_name + '/' + str(e) + '_'
    f = open(io_prefix + 'jammed_flag_list.csv', 'r')
    reader = csv.reader(f)
    for row in reader:
        jammed_flags_str = row
    jammed_flags = [float(jammed_flags_str[i]) for i in range(len(jammed_flags_str))]
    f.close()

    times = 0.0
    jamming_free_times = 0.0
    success_rate = []
    for i in range(len(jammed_flags)):
        times += 1.0
        if jammed_flags[i] == 0:
            jamming_free_times += 1.0
        if times >= 20:
            success_rate.append(1.0 * jamming_free_times/(times * 1.0))
            times = 0.0
            jamming_free_times = 0.0
    csv_file = open(io_prefix + 'success_rate.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(success_rate)
    csv_file.close()

    # calculate sum_data_rate
    max_datarate = get_max_datarate()
    f_r = open(io_prefix + 'state_records.csv', 'r')
    f_w = open(io_prefix + 'sum_data_rate.csv', 'w', newline='')
    reader = csv.reader(f_r)
    writer = csv.writer(f_w)
    row_num = 0
    rows_toread = [6, 10, 14]
    data_rate = []
    for row in reader:
        # skip first three lines
        if row_num in rows_toread:
            state_rec_line = row
            data_rate.append([float(state_rec_line[i])*max_datarate for i in range(len(state_rec_line))])
        row_num += 1
    print(data_rate)
    sum_data_rate = []
    for i in range(len(data_rate[0])):
        temp = 0
        for j in range(len(data_rate)):
            temp += data_rate[j][i]
        sum_data_rate.append(temp)
    writer.writerow(sum_data_rate)

    f.close()
    f_w.close()




