import os
from datetime import datetime

import matplotlib.pyplot as plt

from utils.graph_util import evaluate_TP_FP_rates

folder = "../data/benchmark_data/similarity_distance_comparison/2048_CP1"
all_datasets = ["fei_face_original", "georgia_tech", "sof_original", "fei_face_frontal",
                "youtube_faces", "lfw", "in_house_dataset"]
dataset_labels = ["FEI face original", "Georgia Tech", "SoF original", "FEI face frontal", "YouTube Faces", "LFW",
                  "In house Dataset"]

colors = ['b', 'c', 'g', 'm', 'r', 'y', 'darkorange', 'green']


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


similarity_score_list = []
true_labels_list = []
for i, dataset in enumerate(all_datasets):
    print(dataset)
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

    true_labels_list = true_labels[:score_length]
    [thresholds, TPRs, FPRs] = evaluate_TP_FP_rates(similarity_score, true_labels_list)
    tpr_lim = 0
    fpr_lim = 0
    for j, val in enumerate(TPRs):
        if val > 0.6:
            tpr_lim = j
            break
    for j, val in enumerate(FPRs):
        if val > 0.6:
            fpr_lim = j
            break
    print(tpr_lim, fpr_lim)
    a = FPRs[tpr_lim:fpr_lim]
    b = TPRs[tpr_lim:fpr_lim]
    plt.plot(FPRs[tpr_lim:fpr_lim], TPRs[tpr_lim:fpr_lim], colors[i], label=f'{dataset_labels[i]}')

plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.title('FPR/TPR')
plt.xlabel('FPR')
plt.ylabel('TPR')

date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
plt.savefig(f'.\\FPR_TPR_all_{date_time}', bbox_inches='tight')
plt.clf()
