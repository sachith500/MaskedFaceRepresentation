import argparse
import csv
import os
from datetime import datetime

import numpy as np

from utils.evaluation_utils import evaluate
from utils.lable_utils import get_true_labels
from verification.pipeline import Pipeline


def create_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def regenerate_results(output_path, base_folder="D:\\MaskedFaceRecognitionCompetition\\dataset\\evaluation_datasets\\"):
    datasets = ["fei_face_original", "georgia_tech", "sof_original", "fei_face_frontal", "youtube_faces", "lfw",
                "in_house_dataset"]
    experiment_models = ["EX1", "CP1", "CP2", "FT1", "FT2", "FT3"]
    trained_models = {
        "EX1": "./models/EX1.pt", "CP1": "./models/CP1.pt", "CP2": "./models/CP2.pt",
        "FT1": "./models/FT1.pt", "FT2": "./models/FT2.pt", "FT3": "./models/FT3.pt"
    }
    if os.path.isdir(output_path):
        result_csv = f"{output_path}/experiment_2_results.csv"
    else:
        result_csv = output_path

    with open(result_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        data_dict = dict()
        row = [""]
        row.extend(experiment_models)
        writer.writerow(row)
        date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        base_saving_path = f"./outputs/{date_time}"
        create_folder(base_saving_path)

        for model in experiment_models:
            model_data = dict()
            for dataset in datasets:
                evaluation_file = f"./data/training_pairs_data/{dataset}/evaluation_{dataset}.txt"
                input_evaluation_file = f"./data/training_pairs_data/{dataset}/landmark_evaluation_input_{dataset}.txt"
                landmarks_file = f"./data/training_pairs_data/{dataset}/landmark_list_{dataset}.txt"

                saving_model_path = f"{base_saving_path}/{model}"
                create_folder(saving_model_path)

                output_file = f"{saving_model_path}\\score_{dataset}.txt"

                true_labels = get_true_labels(evaluation_file, input_evaluation_file)
                true_labels = np.array(true_labels).astype(int)
                row = [input_evaluation_file]

                trained_model = trained_models.get(model)
                dataset_base_folder = f"{base_folder}/{dataset}/"
                pipeline = Pipeline(input_evaluation_file, landmarks_file, output_file, trained_model,
                                    dataset_base_folder)
                similarity_score = pipeline.process()
                similarity_score_length = len(similarity_score)

                [fnmr_fmr0, fnmr_fmr100, fnmr_fmr1000, eer] = evaluate(similarity_score,
                                                                       true_labels[:similarity_score_length])
                row.extend([fnmr_fmr0, fnmr_fmr100, fnmr_fmr1000, eer])
                model_data[dataset] = eer

            data_dict[model] = model_data

        for dataset in datasets:
            row = [dataset]
            for model in experiment_models:
                row.append(data_dict.get(model).get(dataset))
            writer.writerow(row)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="Convert face to synthetic masked face",
                                         description='Convert face to synthetic masked face', )
    arg_parser.add_argument('--output', action='store', type=str, default="./")

    args = arg_parser.parse_args()

    regenerate_results(args.output)
