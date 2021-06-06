import argparse
import csv
import os

import numpy as np

from utils.evaluation_utils import evaluate
from utils.lable_utils import get_true_labels, get_similarity_scores


def regenerate_results(output_path):
    datasets = ["fei_face_original", "georgia_tech", "sof_original", "fei_face_frontal", "youtube_faces", "lfw",
                "in_house_dataset"]
    experiment_models = ["CP1", "CP2", "FT1", "FT2", "FT3"]

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

        for model in experiment_models:
            model_data = dict()
            for dataset in datasets:
                score_file = f"./data/benchmark_data/experiment_2/{model}/score_{dataset}.txt"
                evaluation_file = f"./data/training_pairs_data/{dataset}/evaluation_{dataset}.txt"
                input_evaluation_file = f"./data/training_pairs_data/{dataset}/landmark_evaluation_input_{dataset}.txt"

                true_labels = get_true_labels(evaluation_file, input_evaluation_file)
                true_labels = np.array(true_labels).astype(int)
                row = [input_evaluation_file]

                similarity_score = get_similarity_scores(score_file)
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
