import csv

# -------------------- MAIN ----------------------------
f = open('CHACNet jammed flag list.csv', 'r')
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
    if times >= 100:
        success_rate.append(1.0 * jamming_free_times/(times * 1.0))
        times = 0.0
        jamming_free_times = 0.0
csv_file = open('CHACNet success rate.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(success_rate)
csv_file.close()