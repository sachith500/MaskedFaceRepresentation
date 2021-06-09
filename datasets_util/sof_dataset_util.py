import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from utils.masked_face_creator import MaskedFaceCreator


class SoFMaskedFaceDatasetCreator:
    def __init__(self, dataset_path, new_dataset_folder_path, mask_type="a"):
        self.dataset_path = dataset_path
        self.new_dataset_folder_path = new_dataset_folder_path
        self.mask_type = mask_type
        self.mask_color = (255, 255, 255)
        self.masked_face_creator = MaskedFaceCreator('./assets/shape_predictor_68_face_landmarks.dat')

    def generate(self):
        if not os.path.isdir(self.new_dataset_folder_path):
            os.mkdir(self.new_dataset_folder_path)
        for file in tqdm(os.listdir(self.dataset_path)):
            name_components = file.split("_")
            person = name_components[0]

            celebrity_folder = self.new_dataset_folder_path + f"\\{person}"
            if not os.path.isdir(celebrity_folder):
                os.mkdir(celebrity_folder)

            src_path = f"{self.dataset_path}\\{file}"
            new_file_path = f"{celebrity_folder}\\{file}"
            new_masked_file_path = f"{celebrity_folder}\\masked_{file}"
            #
            if os.path.isfile(src_path):
                shutil.copy(src_path, new_file_path)

            image = cv2.imread(src_path)

            image_with_mask = self.masked_face_creator.simulateMask(np.array(image, dtype=np.uint8),
                                                                    mask_type=self.mask_type, color=(255, 255, 255),
                                                                    draw_landmarks=False)
            if image_with_mask is None:
                continue
            cv2.imwrite(new_masked_file_path, image_with_mask)
