import argparse
import os

import cv2
import numpy as np

from utils.masked_face_creator import MaskedFaceCreator


def create_masked_face(input_file_path, output_file_path, mask_type):
    mask_face_creator = MaskedFaceCreator('./assets/shape_predictor_68_face_landmarks.dat')
    if os.path.isfile(input_file_path):
        image = cv2.imread(input_file_path)

        image_with_mask = mask_face_creator.simulateMask(np.array(image, dtype=np.uint8), mask_type=mask_type,
                                                         color=(255, 255, 255),
                                                         draw_landmarks=False)
        if image_with_mask is None:
            print("Couldn't find a face to apply synthetic mask")
        cv2.imwrite(output_file_path, image_with_mask)
    else:
        print("Please check your input file path")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="Convert face to synthetic masked face",
                                         description='Convert face to synthetic masked face', )
    arg_parser.add_argument('--input', action='store', type=str)
    arg_parser.add_argument('--output', action='store', type=str)
    arg_parser.add_argument('--mask', action='store', type=str, default="a")

    args = arg_parser.parse_args()

    create_masked_face(args.input, args.output, args.mask)
