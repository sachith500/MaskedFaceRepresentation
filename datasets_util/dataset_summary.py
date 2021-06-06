import os

from tqdm import tqdm

all_other_datasets = ["feifei_original", "georgia_tech_non_crop", "sof_original", "feifei_front_align",
                      "youtube_faces_categories", "lfw"]


def get_total_image_count(dataset):
    dataset_path = f"D:\\MaskedFaceRecognitionCompetition\\dataset\\evaluation_datasets\\{dataset}"
    total_images = 0
    for folder in tqdm(os.listdir(dataset_path)):
        folder_path = f"{dataset_path}\\{folder}"
        files = [f for f in os.listdir(folder_path) if os.path.isfile(f"{folder_path}\\{f}")]
        total_images += len(files)

    print(dataset, total_images)


def get_used_image_count(dataset):
    landmark_file = f"../outputs/landmark_list_{dataset}.txt"

    with open(landmark_file, 'r') as f:
        lines = f.readlines()

    identities = dict()

    for line in lines:
        file_path = line.split(" ")[0]
        folder = file_path.split("/")[2]
        identities[folder] = True

    print(dataset, len(identities.keys()))

if __name__ == "__main__":
    for dataset in all_other_datasets:
        get_used_image_count(dataset)

