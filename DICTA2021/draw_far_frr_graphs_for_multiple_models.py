import os
from datetime import datetime

import matplotlib.pyplot as plt

from utils.graph_util import evaluate

folders = ["../data/benchmark_data/similarity_distance_comparison/sigmoid_CP1", "../data/benchmark_data/similarity_distance_comparison/2048_CP1","../data/benchmark_data/similarity_distance_comparison/512_CP1"]

all_datasets = ["fei_face_original", "georgia_tech", "sof_original", "fei_face_frontal",
                "youtube_faces", "lfw", "in_house_dataset"]

colors = ['b', 'c', 'g', 'm', 'r', 'y', 'darkorange', 'green']
line_styles = ['solid', 'dashed', 'dotted']
label = ["sigmoid",  "2048", "512"]


def __get_evaluation_line_details(line):
    reference, probe, label_reference, label_probe, true_label = line.split(" ")
    key = f"{reference.strip()}_{probe.strip()}_{label_reference.strip()}_{label_probe.strip()}"
    return key, int(true_label.strip())


def __get_input_evaluation_line_details(line):
    reference, probe, label_reference, label_probe = line.split(" ")
    key = f"{reference.strip()}_{probe.strip()}_{label_reference.strip()}_{label_probe.strip()}"
    return key


def get_true_labels(evaluation_file, input_evaluation_file):
    if os.path.isfile(input_evaluation_file):
        with open(input_evaluation_file, 'r', newline='') as file:
            input_evaluation_lines = file.readlines()

        if os.path.isfile(evaluation_file):
            with open(evaluation_file, 'r', newline='') as file:
                evaluation_lines = file.readlines()

            input_index = 0
            evaluation_index = 0
            input_true_labels = []

            while input_index < len(input_evaluation_lines):

                input_line = __get_input_evaluation_line_details(input_evaluation_lines[input_index])
                evaluation_line, true_label = __get_evaluation_line_details(evaluation_lines[evaluation_index])

                if input_line == evaluation_line:
                    input_true_labels.append(true_label)
                    evaluation_index += 1
                    input_index += 1
                else:
                    evaluation_index += 1

                if len(evaluation_lines) < evaluation_index:
                    print("something wrong with evaluation list.")
                    break
            return input_true_labels
        else:
            return []
    else:
        return []




i = 0

for folder in folders:
    print(folder)
    similarity_score_list = []
    true_labels_list = []
    for dataset in all_datasets:
        file_name = f"{folder}/score_{dataset}.txt"

        evaluation_file = f"..\\data\\training_pairs_data\\{dataset}\\evaluation_{dataset}.txt"
        input_evaluation_file = f"..\\data\\training_pairs_data\\{dataset}\\landmark_evaluation_input_{dataset}.txt"

        true_labels = get_true_labels(evaluation_file, input_evaluation_file)

        with open(file_name, 'r', newline='') as file:
            scores = file.readlines()
        similarity_score = []
        for score in scores:
            similarity_score.append(float(score.strip()))
        score_length = len(similarity_score)

        similarity_score_list.extend(similarity_score)
        true_labels_list.extend(true_labels[:score_length])

    [fnmr_fmr0, fnmr_fmr100, fnmr_fmr1000, eer, thresholds, FMRs, FNMRs] = evaluate(similarity_score_list,
                                                                                    true_labels_list)
    print(f"Accuracy {1 - eer}  Rank {fnmr_fmr100}")
    print(f"0: {fnmr_fmr0}  1000: {fnmr_fmr1000}")
    plt.plot(thresholds, FMRs, colors[i],  label=f"FAR-{label[i]}")
    plt.plot(thresholds, FNMRs, colors[i], linestyle='dashed',label=f"FRR-{label[i]}")
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title('FAR/FRR')
    plt.xlabel('Threshold')
    plt.ylabel('Similarity score')


    i += 1
date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
plt.savefig(f'.\\FAR_FRR_all_{date_time}', bbox_inches='tight')
plt.clf()
