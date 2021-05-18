import numpy as np
'''
def get_best_performance():
    metrics = ['Recall@5', 'MAP@5', 'Recall@10', 'MAP@10']
    fname = "log/ModelTree2_1_2021-04-26_14-00-32.txt"
    output_fname = "result/ModelTree2_1_2021-04-26_14-00-32.txt"

    validation_performances = [] #[[epoch, recall@5, map@5, recall@10, map@10], [...], ...]
    test_performances = []
    val = []
    test = []
    flag = 0
    file = open(fname, "r")

    for line in file:
        if "=======" in line:
            if val != []:
                validation_performances.append(val)
                test_performances.append(test)
            idx = int(line.strip().split("Epoch ")[1].split()[0])
            val = [idx]
            test = [idx]

        if "Validation performance" in line:
            flag = 1
        elif "Test performance" in line:
            flag = 2

        if "@" not in line:
            continue

        if flag == 1:
            val.append(float(line.strip().split(": ")[-1]))
        else:
            test.append(float(line.strip().split(": ")[-1]))
    if val != []:
        validation_performances.append(val)
        test_performances.append(test)

    validation_performances = np.array(validation_performances)
    test_performances = np.array(test_performances)

    fw = open(output_fname, "w+")
    for metric in metrics:
        fw.write("according to metric " + metric + "\n")
        best_val_idx = np.argmax(validation_performances[:,1])
        fw.write("best validation epoch: " + str(best_val_idx) + "\n")
        for i in range(len(metrics)):
            fw.write("validation: " + metrics[i] +": " + str(validation_performances[best_val_idx][i+1]) + "\n")
        for i in range(len(metrics)):
            fw.write("test: " + metrics[i] +": " + str(test_performances[best_val_idx][i+1]) + "\n")
        fw.write("\n\n")
    fw.flush()
    fw.close()
'''
def get_best_performance_1(k_list):
    metrics = []
    for k in k_list:
        metrics.append('macro_precision@%d' % k)
        metrics.append('micro_precision@%d' % k)
        metrics.append('macro_recall@%d' % k)
        metrics.append('micro_recall@%d' % k)
        metrics.append('macro_f1@%d' % k)
        metrics.append('micro_f1@%d' % k)
        metrics.append('macro_map@%d' % k)
    metrics.append('macro_auc')
    metrics.append('micro_auc')

    fname = "log/ModelTree2_1_2021-05-13_14-13-26.txt"
    output_fname = "result/ModelTree2_2021-05-14_10-37-05.txt"

    validation_performances = [] #[[epoch, recall@5, map@5, recall@10, map@10], [...], ...]
    test_performances = []
    val = []
    test = []
    flag = 0
    file = open(fname, "r")

    for line in file:
        if "=======" in line:
            if val != []:
                validation_performances.append(val)
                test_performances.append(test)
            idx = int(line.strip().split("Epoch ")[1].split()[0])
            val = [idx]
            test = [idx]

        if "valid: " in line:
            val.append(float(line.strip().split(": ")[-1]))
        elif "test: " in line:
            test.append(float(line.strip().split(": ")[-1]))

    if val != []:
        validation_performances.append(val)
        test_performances.append(test)

    validation_performances = np.array(validation_performances)
    test_performances = np.array(test_performances)

    fw = open(output_fname, "w+")
    for metric in metrics:
        fw.write("according to metric " + metric + "\n")
        best_val_idx = np.argmax(validation_performances[:,1])
        fw.write("best validation epoch: " + str(best_val_idx) + "\n")
        for i in range(len(metrics)):
            fw.write("validation: " + metrics[i] +": " + str(validation_performances[best_val_idx][i+1]) + "\n")
        for i in range(len(metrics)):
            fw.write("test: " + metrics[i] +": " + str(test_performances[best_val_idx][i+1]) + "\n")
        fw.write("\n\n")
    fw.flush()
    fw.close()

if __name__ == '__main__':
    k_list = [5, 10, 15]
    get_best_performance_1(k_list)