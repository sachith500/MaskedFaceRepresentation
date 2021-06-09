import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from utils.masked_face_creator import MaskedFaceCreator


class ClebAMaskedFaceDatasetCreator:
    def __init__(self, dataset_path, dataset_assert_file, new_dataset_folder_path, mask_type="a"):
        self.dataset_path = dataset_path
        self.new_dataset_folder_path = new_dataset_folder_path
        self.mask_type = mask_type
        self.mask_color = (255, 255, 255)
        self.masked_face_creator = MaskedFaceCreator('./assets/shape_predictor_68_face_landmarks.dat')
        self.dataset_assert_file = dataset_assert_file

    def get_celebrity_dataset(self):
        f = open(self.dataset_assert_file, "r")
        lines = f.readlines()
        celebrity_dict = {}
        for line in lines:
            data = line.split(" ")
            celebrity = data[1].rstrip()
            image = data[0]

            image_array = celebrity_dict.get(celebrity)
            if image_array is None:
                image_array = []
            image_array.append(image)

            celebrity_dict[celebrity] = image_array

        return celebrity_dict

    def generate(self):
        celebrity_dataset = self.get_celebrity_dataset()
        celebrity_dataset_path = self.dataset_path
        mask_dataset_path = self.new_dataset_folder_path
        if not os.path.isdir(mask_dataset_path):
            os.mkdir(mask_dataset_path)

        for celebrity in tqdm(celebrity_dataset.keys()):
            celebrity_files = celebrity_dataset.get(celebrity)
            masked_celebrity_folder = f"{mask_dataset_path}/{celebrity}"
            if not os.path.isdir(masked_celebrity_folder):
                os.mkdir(masked_celebrity_folder)
            for file in celebrity_files:
                file = file[:-4] + ".png"
                src_path = f"{celebrity_dataset_path}\\{file}"
                unmask_file_path = f"{masked_celebrity_folder}\\{file}"
                new_masked_file_path = f"{masked_celebrity_folder}\\masked_{file}"

                if os.path.isfile(src_path):
                    image = cv2.imread(src_path)
                    image_with_mask = self.masked_face_creator.simulateMask(np.array(image, dtype=np.uint8),
                                                                            mask_type=self.mask_type,
                                                                            color=self.mask_color,
                                                                            draw_landmarks=False)
                    if image_with_mask is None:
                        continue
                    shutil.copy(src_path, unmask_file_path)
                    cv2.imwrite(new_masked_file_path, image_with_mask)

    def show_celebrity_dataset(self):
        celebrity_dict = self.get_celebrity_dataset()

        for key in celebrity_dict.keys():
            images = celebrity_dict.get(key)
            for image_path in images:
                png_path = f"{image_path[:-4]}.png"
                i_path = f"{self.dataset_path}\\{png_path}"
                image = cv2.imread(i_path)
                cv2.imshow(key, image)
                cv2.waitKey()
