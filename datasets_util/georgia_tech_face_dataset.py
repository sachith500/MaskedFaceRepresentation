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
        name_components = file.split("_")
        person = name_components[0]

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


if __name__ == "__main__":
    dataset_folder = "D:\MaskedFaceRecognitionCompetition\dataset\evaluation_datasets\georgia_tech_face_database\GTdb_crop\cropped_faces"
    new_dataset_folder = "D:\MaskedFaceRecognitionCompetition\dataset\evaluation_datasets\georgia_tech_face_database\\gt_db_crop_category"
    generate_front_align_dataset(dataset_folder, new_dataset_folder)