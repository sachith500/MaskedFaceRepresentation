import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from utils.facemask import FaceMasked


def generate_front_align_dataset(dataset_path, new_dataset_folder_path):
    dl = FaceMasked('../assets/shape_predictor_68_face_landmarks.dat')
    if not os.path.isdir(new_dataset_folder_path):
        os.mkdir(new_dataset_folder_path)
    for file in tqdm(os.listdir(dataset_folder)):
        person = file[:-5]

        celebrity_folder = new_dataset_folder_path + f"\\{person}"
        if not os.path.isdir(celebrity_folder):
            os.mkdir(celebrity_folder)

        src_path = f"{dataset_path}\\{file}"
        new_file_path = f"{celebrity_folder}\\{file}"
        new_masked_file_path = f"{celebrity_folder}\\masked_{file}"
        #
        if os.path.isfile(src_path):
            shutil.copy(src_path, new_file_path)

        image = cv2.imread(src_path)

        image_with_mask = dl.simulateMask(np.array(image, dtype=np.uint8), mask_type="a", color=(255, 255, 255),
                                          draw_landmarks=False)
        if image_with_mask is None:
            continue
        cv2.imwrite(new_masked_file_path, image_with_mask)


def generate_original_image_dataset(dataset_path, new_dataset_folder_path):
    dl = FaceMasked('../assets/shape_predictor_68_face_landmarks.dat')
    selected_face_types = ['05', '06', '11', '12', '13', '14']
    if not os.path.isdir(new_dataset_folder_path):
        os.mkdir(new_dataset_folder_path)
    for file in tqdm(os.listdir(dataset_folder)):
        name_components = file.split("-")
        person = name_components[0]
        type = name_components[1][:-4]

        if not type in selected_face_types:
            continue


        celebrity_folder = new_dataset_folder_path + f"\\{person}"
        if not os.path.isdir(celebrity_folder):
            os.mkdir(celebrity_folder)

        src_path = f"{dataset_path}\\{file}"
        new_file_path = f"{celebrity_folder}\\{file}"
        new_masked_file_path = f"{celebrity_folder}\\masked_{file}"
        #
        if os.path.isfile(src_path):
            shutil.copy(src_path, new_file_path)

        image = cv2.imread(src_path)

        image_with_mask = dl.simulateMask(np.array(image, dtype=np.uint8), mask_type="a", color=(255, 255, 255),
                                          draw_landmarks=False)
        if image_with_mask is None:
            continue
        cv2.imwrite(new_masked_file_path, image_with_mask)

def get_total_image_count():
    dataset_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\evaluation_datasets\\feifei_front_align"
    total_images = 0
    for folder in tqdm(os.listdir(dataset_path)):
        folder_path = f"{dataset_path}\\{folder}"
        files = [f for f in os.listdir(folder_path) if os.path.isfile(f"{folder_path}\\{f}")]
        total_images += len(files)

    print(total_images)
if __name__ == "__main__":
    # dataset_folder = "D:\MaskedFaceRecognitionCompetition\dataset\evaluation_datasets\\fei_face_database\\originalimages"
    # new_dataset_folder = "D:\MaskedFaceRecognitionCompetition\dataset\evaluation_datasets\\fei_face_database\\original_images_category"
    # generate_original_image_dataset(dataset_folder, new_dataset_folder)
    get_total_image_count()
