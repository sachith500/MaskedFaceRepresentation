import os

import cv2
import numpy as np
from tqdm import tqdm

from verification.config import *
from verification.model import Siamese


class Pipeline:

    def __init__(self, evaluation_list_file, landmarks_file, output_path, model_path=None, dataset_base_folder=None):
        self.evaluation_list_file = evaluation_list_file
        if isinstance(model_path,list):
            self.model_paths = model_path
        else:
            self.model_paths = [model_path]
        self.dataset_base_folder = dataset_base_folder
        self.bbox_dict = dict()
        self.facial_key_points_dict = dict()
        if os.path.isfile(landmarks_file):
            with open(landmarks_file, 'r', newline='') as file:
                lines = file.readlines()
                for line in lines:
                    file_name, bbox, facial_key_points = self.__get_landmark_line_details(line)
                    self.bbox_dict[file_name] = bbox
                    self.facial_key_points_dict[file_name] = facial_key_points

        else:
            print(f"Landmark file is not accessible. Please check the path {landmarks_file}.")
            exit(0)

        self.output_path = output_path

    def build_simple_models(self, model_path):
        model = Siamese(model_path)
        return model

    def build_models(self, model_path):
        model = Siamese(model_path)
        return model

    def evaluate_images(self, reference_image, reference_bbox, probe_image, probe_bbox, true_label):
        pass

    def __image_margin(self, image):
        shape = IMG_SHAPE
        background = np.zeros(shape, dtype=np.uint8)
        image_shape = image.shape

        r = int((shape[0] - image_shape[0]) / 2)
        c = int((shape[1] - image_shape[1]) / 2)

        background[r:r + image_shape[0], c:c + image_shape[1], :] = image

        return background

    def scale_and_resize_image(self, image):
        shape = IMG_SHAPE
        image_shape = image.shape

        if image_shape[0] > image_shape[1]:
            # print((image_shape), (shape[0], int(image_shape[1] * (shape[1] / image_shape[0]))))
            resized_image = cv2.resize(image, (int(image_shape[1] * (shape[1] / image_shape[0])), shape[0]))
        else:
            # print((image_shape), (int(image_shape[0] * (shape[1] / image_shape[1])), shape[1]))
            resized_image = cv2.resize(image, (shape[0], int(image_shape[0] * (shape[1] / image_shape[1]))))
        # print(image_shape, resized_image.shape)
        return self.__image_margin(resized_image)

    def __crop_image(self, image, x1, y1, x2, y2, ratio=1.0):

        y1_ = int(y1 - (y2 - y1) * ratio)
        x1_ = int(x1 - (x2 - x1) * ratio)
        y2_ = int(y2 + (y2 - y1) * ratio)
        x2_ = int(x2 + (x2 - x1) * ratio)

        if x1_ <= 0:
            x1_ = 0
        if y1_ <= 0:
            y1_ = 0
        if x2_ > image.shape[1]:
            x2_ = image.shape[1]
        if y2_ > image.shape[0]:
            y2_ = image.shape[0]

        cropped_image = image[y1_:y2_, x1_:x2_]

        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            print(image.shape, x1, x2, y1, y2, x1_, x2_, y1_, y2_)

        return self.scale_and_resize_image(cropped_image)

    def open_image(self, image_path):
        if not PRODUCTION:
            image_path = f"{self.dataset_base_folder}\\{image_path}"
        image = cv2.imread(image_path)
        if image is None:
            print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __get_processed_inference_images(self, image_path):
        images = []
        original_image = self.open_image(image_path)

        face_bbox = self.bbox_dict.get(image_path)
        if face_bbox is not None:
            for ratio in [0.6]:
                cropped_image = self.__crop_image(original_image, face_bbox[0], face_bbox[1], face_bbox[2],
                                                  face_bbox[3], ratio=ratio)
                images.append(cropped_image)

        return np.array(images)

    def process(self):
        print("Processing started.")
        reference_list, probe_list, label_reference_list, label_probe_list = self.read_evaluation_list_file()
        print("Reading evaluation file list is done.")
        print("Evaluation is started.")
        comparison_scores = None
        for model_path in self.model_paths:
            scores = []
            self.siamese_model = self.build_models(model_path)
            for idx, reference in enumerate(tqdm(reference_list)):
                probe = probe_list[idx]
                score = self.inference_images(reference, probe)

                scores.append(score)
            scores = np.array(scores)
            if comparison_scores is None:
                comparison_scores = scores
            else:
                comparison_scores += scores
        comparison_scores /= len(self.model_paths)
        print("Evaluation is completed.")
        self.write_results_to_output(comparison_scores)
        print("Writing the scores to the file is completed.")
        return comparison_scores

    def inference_images(self, reference, probe):
        reference_images = self.__get_processed_inference_images(reference)
        probe_images = self.__get_processed_inference_images(probe)

        self.siamese_model = self.siamese_model.cuda()
        predicted_score = self.siamese_model.predict([reference_images, probe_images])

        average_score = np.average(predicted_score)
        return average_score

    def __get_landmark_line_details(self, line):
        values = line.split(" ")
        file_name = values[0]
        bbox = np.array(values[1:5]).astype(int)
        facial_key_points = np.array(values[5:]).astype(int)

        bbox = bbox.tolist()
        facial_key_points = facial_key_points.tolist()

        return file_name, bbox, facial_key_points

    def __get_evaluation_line_details(self, line):
        reference, probe, label_reference, label_probe = line.split(" ")

        return reference.strip(), probe.strip(), label_reference.strip(), label_probe.strip()

    def read_evaluation_list_file(self):
        reference_list, probe_list, label_reference_list, label_probe_list = [], [], [], []
        if os.path.isfile(self.evaluation_list_file):
            with open(self.evaluation_list_file, 'r', newline='') as file:
                lines = file.readlines()
                i = 0
                for line in lines:
                    if i > 1000:
                        break
                    reference, probe, label_reference, label_probe = self.__get_evaluation_line_details(line)
                    reference_list.append(reference)
                    probe_list.append(probe)
                    label_reference_list.append(label_reference)
                    label_probe_list.append(label_probe)
                    i += 1

        else:
            print(f"Evaluation list is not accessible. Please check the path {self.evaluation_list_file}.")

        return reference_list, probe_list, label_reference_list, label_probe_list

    def write_results_to_output(self, comparison_score_list):
        output_file_name = self.output_path
        if os.path.isdir(self.output_path):
            output_file_name = f"{output_file_name}\\scores.txt"

        with open(output_file_name, 'w', newline='') as file:
            for comparison_score in comparison_score_list:
                comparison_score_str = str(comparison_score) + "\n"
                file.write(comparison_score_str)
