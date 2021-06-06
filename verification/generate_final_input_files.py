import numpy as np
from tqdm import tqdm

dataset = "our_dataset"
evaluation_file = f"./inputs/landmark_evaluation_inut_{dataset}.txt"
landmark_file = f"./inputs/landmark_list_{dataset}.txt"

DATASET_BASE_FOLDER = "D:\MaskedFaceRecognitionCompetition\dataset\evaluation_datasets\\"
base_path = f"{DATASET_BASE_FOLDER}\\{dataset}\\"


def get_evaluation_line_details(line):
    reference, probe, label_reference, label_probe = line.split(" ")

    return reference.strip(), probe.strip(), label_reference.strip(), label_probe.strip()


def get_landmark_line_details(line):
    values = line.split(" ")
    file_name = values[0]
    info = np.array(values[1:]).astype(int)

    return file_name, info


with open(evaluation_file, 'r', newline='') as file:
    lines = file.readlines()
reference_list, probe_list, label_reference_list, label_probe_list = [], [], [], []
for line in lines:
    reference, probe, label_reference, label_probe = get_evaluation_line_details(line)
    reference = f"{base_path}{reference}"
    probe = f"{base_path}{probe}"
    reference_list.append(reference)
    probe_list.append(probe)
    label_reference_list.append(label_reference)
    label_probe_list.append(label_probe)

with open(f"./inputs/final_{evaluation_file.split('/')[-1]}", 'w') as f:
    for idx, reference in enumerate(tqdm(reference_list)):
        details = [reference, probe_list[idx], label_reference_list[idx], label_probe_list[idx]]
        print(' '.join(details) + "\n")
        f.write(' '.join(details) + "\n")

with open(landmark_file, 'r', newline='') as file:
    lines = file.readlines()

landmark_details = []
for line in lines:
    file, info = get_landmark_line_details(line)
    file = f"{base_path}{file}"
    info = info.astype(str)
    data = [file]
    data.extend(info)

    landmark_details.append(data)

with open(f"./inputs/final_{landmark_file.split('/')[-1]}", 'w') as f:
    for details in tqdm(landmark_details):
        print(' '.join(details) + "\n")
        f.write(' '.join(details) + "\n")
