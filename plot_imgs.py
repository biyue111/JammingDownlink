import csv
import numpy as np
import matplotlib.pyplot as plt
import configs as configs


def read_csv_array(file_name):
    file_str = file_name
    f = open(file_str, 'r')
    reader = csv.reader(f)

    str_data = []
    for row in reader:
        str_data.append(row)
    f.close()

    data = np.array([[float(x) for x in row] for row in str_data])
    print(data.shape)
    print(data)
    return data


def read_csv_files(scenario_name, model_name, experiment_num, file_name):
    for e in range(experiment_num):
        io_prefix = 'temp_results/' + scenario_name + '/' + model_name + '/' + str(e) + '_'
        data = read_csv_array(io_prefix + file_name)
        if e == 0:
            all_data = data
        else:
            all_data = np.vstack((all_data, data))
    print(all_data)
    print(all_data.shape)
    return all_data


# -------------------- MAIN ----------------------------
# plot success rate data
success_rates = read_csv_files('downlink_basic', 'DDPG_PNN_SEQ', 10, 'success_rate.csv')
success_rate_mean = np.mean(success_rates, axis=0)
success_rate_std = np.std(success_rates, axis=0)
x = np.arange(1, 1000, 20)
up = success_rate_mean + success_rate_std
down = success_rate_mean - success_rate_std
plt.plot(x, success_rate_mean, color='black')
plt.fill_between(x, up, down)
plt.show()

sum_data_rates = read_csv_files('downlink_basic', 'DDPG_PNN_SEQ', 10, 'sum_data_rate.csv')
sum_data_rate_mean = np.mean(sum_data_rates, axis=0)
sum_data_rate_std = np.std(sum_data_rates, axis=0)
x = range(1, configs.UPDATE_NUM+1)
up = sum_data_rate_mean + sum_data_rate_std
down = sum_data_rate_mean - sum_data_rate_std
plt.plot(x, sum_data_rate_mean, color='black')
plt.fill_between(x, up, down)
plt.show()


