import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from utils.masked_face_creator import MaskedFaceCreator


def create_masked_face(input_dir_path, output_dir_path, mask_type):
    mask_face_creator = MaskedFaceCreator('./assets/shape_predictor_68_face_landmarks.dat')
    if os.path.isdir(input_dir_path):
        if os.path.isdir(output_dir_path):
            os.mkdir(output_dir_path)

        for face_image_name in tqdm(os.listdir(input_dir_path)):
            face_image_path = f"{input_dir_path}/{face_image_name}"
            masked_face_image_path = f"{output_dir_path}/masked_{face_image_name}"
            if not os.path.isfile(face_image_path):
                continue
            image = cv2.imread(face_image_path)

            image_with_mask = mask_face_creator.simulateMask(np.array(image, dtype=np.uint8), mask_type=mask_type,
                                                             color=(255, 255, 255), draw_landmarks=False)
            if image_with_mask is None:
                print(f"Image: {face_image_path} : Couldn't find a face to apply synthetic mask")
                continue
            cv2.imwrite(masked_face_image_path, image_with_mask)
    else:
        print("Please check your input directory path")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="Convert face to synthetic masked face",
                                         description='Convert face to synthetic masked face', )
    arg_parser.add_argument('--input', action='store', type=str)
    arg_parser.add_argument('--output', action='store', type=str)
    arg_parser.add_argument('--mask', action='store', type=str, default="a")

    args = arg_parser.parse_args()

    create_masked_face(args.input, args.output, args.mask)
