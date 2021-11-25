import argparse
import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from utils.masked_face_creator import MaskedFaceCreator

random.seed(2021)


def get_age(file_name):
    age = file_name.split("_")[0]
    return age


def get_gender(file_name):
    gender = file_name.split("_")[1]
    return gender


def get_race(file_name):
    race = file_name.split("_")[2]
    return race


def generate_folders(dataset_folder, new_dataset_folder, age=False, gender=False, race=False):
    dl = MaskedFaceCreator('../assets/shape_predictor_68_face_landmarks.dat')
    if age:
        generate_dataset_folder = f"{new_dataset_folder}\\age"
        label_function = get_age
    elif gender:
        generate_dataset_folder = f"{new_dataset_folder}\\gender"
        label_function = get_gender
    elif race:
        generate_dataset_folder = f"{new_dataset_folder}\\race"
        label_function = get_race
    else:
        return

    if not os.path.isdir(new_dataset_folder):
        os.mkdir(new_dataset_folder)
    if not os.path.isdir(generate_dataset_folder):
        os.mkdir(generate_dataset_folder)
    for file in tqdm(os.listdir(dataset_folder)):
        file_path = f"{dataset_folder}\\{file}"
        classification = label_function(file)
        classification_folder = f"{generate_dataset_folder}\\{classification}"
        new_file_path = f"{classification_folder}\\{file}"
        if not os.path.isdir(classification_folder):
            os.mkdir(classification_folder)

        image = cv2.imread(file_path)
        image_with_mask = dl.simulateMask(np.array(image, dtype=np.uint8), mask_type="a", color=(255, 255, 255),
                                          draw_landmarks=False)
        if image_with_mask is None:
            continue
        cv2.imwrite(new_file_path, image_with_mask)


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def copy_files(src, dest, file_list):
    for file in file_list:
        src_path = f"{src}\\{file}"
        dest_path = f"{dest}\\{file}"
        shutil.copyfile(src_path, dest_path)


def separate_dataset_into_train_val_test(dataset_folder, new_folder, train_split, val_split, test_split):
    make_dir(new_folder)

    train_folder = f"{new_folder}\\train"
    val_folder = f"{new_folder}\\val"
    test_folder = f"{new_folder}\\test"

    make_dir(train_folder)
    make_dir(val_folder)
    make_dir(test_folder)

    for dir in os.listdir(dataset_folder):
        train_folder_dir = f"{train_folder}\\{dir}"
        val_folder_dir = f"{val_folder}\\{dir}"
        test_folder_dir = f"{test_folder}\\{dir}"

        make_dir(train_folder_dir)
        make_dir(val_folder_dir)
        make_dir(test_folder_dir)

        current_dir = f"{dataset_folder}\\{dir}"

        files = [f for f in os.listdir(current_dir) if os.path.isfile(f"{current_dir}\\{f}")]
        random.shuffle(files)
        train_split_files_len = int(len(files) * train_split)
        val_split_files_len = int(len(files) * val_split)
        test_split_files_len = int(len(files) - train_split_files_len - val_split_files_len)

        print(
            f"For {dir} train files {train_split_files_len} val files {val_split_files_len} test files {test_split_files_len}")

        train_files = files[:train_split_files_len]
        val_files = files[train_split_files_len:train_split_files_len + val_split_files_len]
        test_files = files[train_split_files_len + val_split_files_len:]

        copy_files(current_dir, train_folder_dir, train_files)
        copy_files(current_dir, val_folder_dir, val_files)
        copy_files(current_dir, test_folder_dir, test_files)


def generate_dataset(dataset_path, new_dataset_path, training, validation, test):
    split_size = 10
    create_folder(new_dataset_path)

    gender_dataset_folder = f"{dataset_path}\\gender"
    data_dictionary = dict()
    for gender in os.listdir(gender_dataset_folder):
        gender_folder_path = f"{gender_dataset_folder}\\{gender}"
        gender_files = os.listdir(gender_folder_path)

        processed_gender_files = 0
        for image in gender_files:

            age = get_age(image)
            race = get_race(image)

            try:
                int(race)
            except:
                continue
            processed_gender_files += 1
            gender_data = data_dictionary.get(gender)
            if gender_data is None:
                gender_data = dict()

            race_data = gender_data.get(race)

            if race_data is None:
                race_data = dict()

            age_data = race_data.get(age)

            if age_data is None:
                age_data = []
            age_data.append(image)

            race_data[age] = age_data
            gender_data[race] = race_data
            data_dictionary[gender] = gender_data

    total_count = 0

    for gender in data_dictionary.keys():
        race_dict = data_dictionary.get(gender)
        gender_count = 0
        distribution_bucket = []
        for race in race_dict.keys():
            age_dict = race_dict.get(race)
            race_count = 0

            age_keys = [int(f) for f in age_dict.keys()]

            age_keys.sort()

            for age in age_keys:
                age_files = age_dict.get(str(age))
                distribution_bucket.extend(age_files)

                # print(gender, race, age, len(age_files))
                value = len(age_files)
                total_count += value
                gender_count += value
                race_count += value

                while len(distribution_bucket) > split_size:
                    images_to_split = distribution_bucket[:split_size]
                    images_to_progress = distribution_bucket[split_size:]
                    distribution_bucket = images_to_progress
                    copy_images_to_respective_classes(dataset_path, new_dataset_path, gender, images_to_split, training,
                                                      validation, test)

        copy_images_to_respective_classes(dataset_path, new_dataset_path, gender, distribution_bucket, training,
                                          validation, test)
        # print(gender,race, gender_count, race_count)
        print("============================================")

    print(total_count)


def copy_images_to_respective_classes(dataset_path, new_dataset_path, gender, images, train, val, test):
    no_of_images = len(images)
    training_split = int(no_of_images * train)
    validation_split = int(training_split + no_of_images * val)
    test_split = int(validation_split + no_of_images * test)

    training_set = images[:training_split]
    validation_set = images[training_split:validation_split]
    test_set = images[validation_split:test_split]
    spare_set = images[test_split:]

    training_set.extend(spare_set)

    for image in training_set:
        age = get_age(image)
        race = get_race(image)
        write_images(dataset_path, new_dataset_path, "training", gender, race, age, image)

    for image in validation_set:
        age = get_age(image)
        race = get_race(image)
        write_images(dataset_path, new_dataset_path, "validation", gender, race, age, image)

    for image in test_set:
        age = get_age(image)
        race = get_race(image)
        write_images(dataset_path, new_dataset_path, "testing", gender, race, age, image)


def write_images(dataset_path, new_dataset_path, section, gender, race, age, image):
    gender_dataset_folder = f"{dataset_path}\\gender"
    gender_folder_path = f"{gender_dataset_folder}\\{gender}"
    src_path = f"{gender_folder_path}\\{image}"

    create_folder(new_dataset_path)

    destination_section = f"{new_dataset_path}\\{section}"
    create_folder(destination_section)

    age_folder = f"{destination_section}\\age"
    race_folder = f"{destination_section}\\race"
    gender_folder = f"{destination_section}\\gender"

    create_folder(age_folder)
    create_folder(race_folder)
    create_folder(gender_folder)

    age_path = f"{age_folder}\\{age}"
    race_path = f"{race_folder}\\{race}"
    gender_path = f"{gender_folder}\\{gender}"

    copy_to_destination(src_path, age_path, image)
    copy_to_destination(src_path, race_path, image)
    copy_to_destination(src_path, gender_path, image)


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def copy_to_destination(src_path, folder, image):
    create_folder(folder)
    destination_path = f"{folder}\\{image}"
    shutil.copy(src_path, destination_path)


def summary(folder_path):
    for section in ["training", "validation", "testing"]:
        validate_section(folder_path, section)


def validate_section(folder_path, section):
    section_folder = f"{folder_path}\\{section}"

    data = dict()
    types = os.listdir(section_folder)
    for type in types:
        type_folder = f"{section_folder}\\{type}"
        class_names = [f for f in os.listdir(type_folder)]
        files_list = []
        for class_name in class_names:
            files_list.extend(os.listdir(f"{type_folder}\\{class_name}"))
        data[type] = files_list

    comparison_type_images = data[types[0]]
    print(len(comparison_type_images))
    for class_name in types[1:]:
        file_names = data[class_name]
        print(len(file_names))
        for file in file_names:
            if file not in comparison_type_images:
                print(file)
                break


def get_folder_count(folder_path, section):
    return len(f"{folder_path}\\{section}")


def create_empty_folders(folder_path):
    class_names = ["age", "gender", "race"]
    splits = ["training", "validation", "testing"]

    for class_name in class_names:
        class_path = f"{folder_path}\\{class_name}"
        for split in splits:
            split_path = f"{class_path}\\{split}"
            folder_list = [int(f) for f in os.listdir(split_path)]
            for i in range(1, max(folder_list)):
                if i not in folder_list:
                    print(split_path, i)
                    create_folder(f"{split_path}\\{i}")


def distribute_age_to_bins(age_dataset_folder):
    bin_split_dict = {}

    for i in range(117):
        print(i)
        if i in range(0, 3 + 1):
            bin_split_dict[i] = 0
        elif i in range(4, 12 + 1):
            bin_split_dict[i] = 1
        elif i in range(13, 19 + 1):
            bin_split_dict[i] = 2
        elif i in range(20, 30 + 1):
            bin_split_dict[i] = 3
        elif i in range(31, 45 + 1):
            bin_split_dict[i] = 4
        elif i in range(46, 60 + 1):
            bin_split_dict[i] = 5
        elif i in range(61, 120 + 1):
            bin_split_dict[i] = 6

    splits = ["age"]
    # splits = ["training", "validation", "testing"]
    new_age_dataset = f"{age_dataset_folder}\\..\\new_stat_age"
    create_folder(new_age_dataset)

    for section in splits:
        section_path = f"{age_dataset_folder}\\{section}"
        destination_section_path = f"{new_age_dataset}\\{section}"
        create_folder(destination_section_path)

        age_keys = [int(f) for f in os.listdir(section_path)]

        age_keys.sort()
        for age in age_keys:
            # print(section, age)
            age_bin = bin_split_dict.get(age)
            src_age_path = f"{section_path}\\{age}"
            new_age_bin_path = f"{destination_section_path}\\{age_bin}"
            create_folder(new_age_bin_path)

            for file in os.listdir(src_age_path):
                src_file_path = f"{src_age_path}\\{file}"
                destination_file_path = f"{new_age_bin_path}\\{file}"
                # print(src_file_path, destination_file_path)
                shutil.copy(src_file_path, destination_file_path)

            # print(age, bin_split_dict.get(age))

            pass


def build_age_dataset(dataset_folder, output_folder):
    generate_folders(dataset_folder, output_folder, age=True)


def build_race_dataset(dataset_folder, output_folder):
    generate_folders(dataset_folder, output_folder, race=True)


def build_sex_dataset(dataset_folder, output_folder):
    generate_folders(dataset_folder, output_folder, gender=True)


def build_age_classification_dataset(dataset_folder, output_folder):
    generate_folders(dataset_folder, output_folder, age=True)
    distribute_age_to_bins(output_folder)


def build_only_age_classification_dataset(age_dataset_folder):
    distribute_age_to_bins(age_dataset_folder)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="Calculate age, race and gender accuracy",
                                         description='Calculate age, race and gender accuracy', )

    arg_parser.add_argument('--dataset', action='store', type=str,
                            default=".\\UTKface_inthewild-20210331T075050Z-001\\UTKface_inthewild\\cropped")
    arg_parser.add_argument('--output', action='store', type=str,
                            default='.\\UTKface_inthewild-20210331T075050Z-001\\UTKface_inthewild\\masked_utk_faces')
    arg_parser.add_argument('--type', action='store', type=str, default='age')  # age, race, sex, age_classification

    args = arg_parser.parse_args()

    if args.type == "age":
        build_age_dataset(args.dataset, args.output)
    elif args.type == "race":
        build_race_dataset(args.dataset, args.output)
    elif args.type == "sex":
        build_sex_dataset(args.dataset, args.output)
    elif args.type == "age_classification":
        build_age_classification_dataset(args.dataset, args.output)
    else:
        print("Type is not define: Please select following types age, race, sex, age_classification")
