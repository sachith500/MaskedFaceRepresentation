import argparse
import os

from datasets_util.celeb_dataset_utils import ClebAMaskedFaceDatasetCreator
from datasets_util.fei_face_dataset_util import FeiFaceFrontAlignMaskedFaceDatasetCreator, \
    FeiFaceOriginalMaskedFaceDatasetCreator
from datasets_util.georgia_tech_face_dataset import GeorgiaTechMaskedFaceDatasetCreator
from datasets_util.lfw_dataset_util import LFWMaskedFaceDatasetCreator
from datasets_util.sof_dataset_util import SoFMaskedFaceDatasetCreator
from datasets_util.youtube_faces_dataset import YoutubeMaskedFaceDatasetCreator


def get_dataset_creator(base_folder, new_database_folder, dataset_name):
    dataset_creator = None
    if dataset_name == "celeba":
        dataset_path = f"{base_folder}/{dataset_name}/images"
        dataset_assert_file = f"{base_folder}/{dataset_name}/identity_CelebA.txt"
        mask_unmask_dataset_path = f"{new_database_folder}/{dataset_name}"
        dataset_creator = ClebAMaskedFaceDatasetCreator(dataset_path, dataset_assert_file, mask_unmask_dataset_path)
    elif dataset_name == "fei_face_frontal":
        dataset_path = f"{base_folder}/{dataset_name}"
        mask_unmask_dataset_path = f"{new_database_folder}/{dataset_name}"
        dataset_creator = FeiFaceFrontAlignMaskedFaceDatasetCreator(dataset_path, mask_unmask_dataset_path)
    elif dataset_name == "fei_face_original":
        dataset_path = f"{base_folder}/{dataset_name}"
        mask_unmask_dataset_path = f"{new_database_folder}/{dataset_name}"
        dataset_creator = FeiFaceOriginalMaskedFaceDatasetCreator(dataset_path, mask_unmask_dataset_path)
    elif dataset_name == "georgia_tech":
        dataset_path = f"{base_folder}/{dataset_name}"
        mask_unmask_dataset_path = f"{new_database_folder}/{dataset_name}"
        dataset_creator = GeorgiaTechMaskedFaceDatasetCreator(dataset_path, mask_unmask_dataset_path)
    elif dataset_name == "sof_original":
        dataset_path = f"{base_folder}/{dataset_name}"
        mask_unmask_dataset_path = f"{new_database_folder}/{dataset_name}"
        dataset_creator = SoFMaskedFaceDatasetCreator(dataset_path, mask_unmask_dataset_path)
    elif dataset_name == "youtube_faces":
        dataset_path = f"{base_folder}/{dataset_name}"
        mask_unmask_dataset_path = f"{new_database_folder}/{dataset_name}"
        dataset_creator = YoutubeMaskedFaceDatasetCreator(dataset_path, mask_unmask_dataset_path)
    elif dataset_name == "lfw":
        dataset_path = f"{base_folder}/{dataset_name}"
        mask_unmask_dataset_path = f"{new_database_folder}/{dataset_name}"
        dataset_creator = LFWMaskedFaceDatasetCreator(dataset_path, mask_unmask_dataset_path)

    return dataset_creator


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="Generate masked face dataset",
                                         description='Generate masked face dataset.', )
    arg_parser.add_argument("--base_folder", type=str, default="../dataset")
    arg_parser.add_argument("--new_database_folder", type=str, default="../new_dataset")

    args = arg_parser.parse_args()
    if not os.path.isdir(args.new_database_folder):
        os.mkdir(args.new_database_folder)

    datasets = ["celeba", "fei_face_original", "georgia_tech", "sof_original", "fei_face_frontal", "lfw",
                "youtube_faces"]

    for dataset in datasets:
        mask_dataset_creator = get_dataset_creator(args.base_folder, args.new_database_folder, dataset)
        mask_dataset_creator.generate()
