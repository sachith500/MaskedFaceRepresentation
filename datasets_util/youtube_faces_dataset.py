import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from utils.masked_face_creator import MaskedFaceCreator


class YoutubeMaskedFaceDatasetCreator:
    def __init__(self, dataset_path, new_dataset_folder_path, mask_type="a"):
        self.dataset_path = dataset_path
        self.new_dataset_folder_path = new_dataset_folder_path
        self.mask_type = mask_type
        self.mask_color = (255, 255, 255)
        self.masked_face_creator = MaskedFaceCreator('./assets/shape_predictor_68_face_landmarks.dat')
        self.no_of_images_from_video = 5

    def generate(self):
        if not os.path.isdir(self.new_dataset_folder_path):
            os.mkdir(self.new_dataset_folder_path)
        for person in tqdm(os.listdir(self.dataset_path)):
            src_folder = f"{self.dataset_path}\\{person}"
            celebrity_folder = f"{self.new_dataset_folder_path}\\{person}"

            if not os.path.isdir(celebrity_folder):
                os.mkdir(celebrity_folder)

            for video_folder in tqdm(os.listdir(src_folder)):
                video_folder_path = f"{src_folder}//{video_folder}"
                all_frames = [i for i in os.listdir(video_folder_path) if os.path.isfile(f"{video_folder_path}//{i}")]
                selected_frames = random.sample(all_frames, 15)

                mask_images_count = 0

                for frame_file in selected_frames:
                    src_path = f"{video_folder_path}\\{frame_file}"

                    new_file_name = f"{video_folder}_{frame_file}"
                    new_file_path = f"{celebrity_folder}\\{new_file_name}"
                    new_masked_file_path = f"{celebrity_folder}\\masked_{new_file_name}"

                    if os.path.isfile(src_path):
                        shutil.copy(src_path, new_file_path)

                    image = cv2.imread(src_path)

                    image_with_mask = self.masked_face_creator.simulateMask(np.array(image, dtype=np.uint8),
                                                                            mask_type=self.mask_type,
                                                                            color=self.mask_color,
                                                                            draw_landmarks=False)
                    if image_with_mask is None:
                        continue
                    cv2.imwrite(new_masked_file_path, image_with_mask)

                    mask_images_count += 1
                    if mask_images_count > self.no_of_images_from_video:
                        break
