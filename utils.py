import os
import pickle

import datetime
import  math

import config

def load_dataset(dataset_split_type='train'):

    if dataset_split_type != 'train' and dataset_split_type != 'test' and dataset_split_type != 'valid':
        exit("Please enter train/test/valid")

    print('Checking whether train doc\'s segments file exists')
    dataset_segments_file_path = 'train_flat_dataset_segments.pickle'
    if os.path.exists(dataset_segments_file_path):
        with open(dataset_segments_file_path, 'rb') as file:
            dataset_segments = pickle.load(file)

        return dataset_segments

    print('Train Doc Segments File does not exists. Exiting...')
    exit(0)



metric_file = None

def start_prediction_eval():
    global metric_file
    targets = []
    outputs = []

    # Opening the metric file to write the scores
    os.makedirs(os.path.dirname(config.metrics_path), exist_ok=True)
    metric_file = open(config.metrics_path, 'a+')
    metric_file.write("\n"+str(datetime.datetime.now())+"\n")
    # Writing  hyperparameters to file
    line = "epochs: {}, lr: {}, weight_decay: {}, momentum: {}, k: {}, cluster_nos: {}\n".format(config.epochs, config.learning_rate, config.weight_decay, config.momentum, config.top_k, config.clusters_num)
    metric_file.write(line)


    with open(config.prediction_save_path, 'rb') as handle:
        temp_target = pickle.load(handle)
        outputs     = pickle.load(handle)


    targets = []
    for each_row in temp_target:
        each_row = [i for i, e in enumerate(each_row[0]) if e == 1]
        targets.append(each_row)


    top_k = [1,3,5]
    precision(top_k, targets, outputs)
    metric_file.close()
    print('Evaluation is finished. Evaluation results are at: ',config.metrics_path)




def precision(top_k, targets, outputs):
    total_test_nos = len(targets)

    for each_k in top_k:
        k = each_k
        precision = 0

        # Traversing the list
        for index in range(0, total_test_nos):

            each_output = outputs[index][0:k]
            count       = 0
            for each_elem in each_output:
                if each_elem in targets[index]:
                    count = count + 1

            precision = precision + count/k

        precision = precision/total_test_nos
        line      = 'Precision@{}:{}\n'.format(k, precision)
        print(line)
        metric_file.write(line)
