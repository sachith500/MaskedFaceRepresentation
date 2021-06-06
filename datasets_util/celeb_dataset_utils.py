import csv
import os
import shutil

import cv2
from tqdm import tqdm


def get_celebrity_dataset():
    f = open('../assets/identity_CelebA.txt', "r")
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


def generate_csv_for_celebrity_dataset():
    celebrity_dict = get_celebrity_dataset()

    celebrity_dataset_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\CelebA\\Img\\img_align_celeba_png"
    mask_dataset_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\CelebA\\Img\\img_align_celeba_png_masked_old"

    with open('../assets/celebrity_details_with_mask.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        for celebrity in tqdm(celebrity_dict.keys()):
            row = [celebrity]
            celebrity_files = celebrity_dict.get(celebrity)
            celebrity_images = []
            for file in celebrity_files:
                file = file[:-4] + ".png"
                src_path = f"{celebrity_dataset_path}\\{file}"
                mask_src_path = f"{mask_dataset_path}\\masked_{file}"

                if os.path.isfile(src_path):
                    celebrity_images.append(file)

                if os.path.isfile(mask_src_path):
                    celebrity_images.append(f"masked_{file}")

            row.extend(celebrity_images)
            writer.writerow(row)


def generate_celeba_dataset_with_masks():
    celebrity_dataset = get_celebrity_dataset()
    celebrity_dataset_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\CelebA\\Img\\img_align_celeba_png"
    mask_dataset_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\CelebA\\Img\\img_align_celeba_png_masked_old"
    new_dataset_folder_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\CelebA\\Img\\img_align_celeba_with_masks"
    if not os.path.isdir(new_dataset_folder_path):
        os.mkdir(new_dataset_folder_path)
    for celebrity in tqdm(celebrity_dataset.keys()):
        celebrity_files = celebrity_dataset.get(celebrity)
        celebrity_folder = new_dataset_folder_path + f"\\{celebrity}"
        if not os.path.isdir(celebrity_folder):
            os.mkdir(celebrity_folder)
        for file in celebrity_files:
            file = file[:-4] + ".png"
            src_path = f"{celebrity_dataset_path}\\{file}"
            mask_src_path = f"{mask_dataset_path}\\masked_{file}"
            new_file_path = f"{celebrity_folder}\\{file}"
            new_masked_file_path = f"{celebrity_folder}\\masked_{file}"

            if os.path.isfile(src_path):
                shutil.copy(src_path, new_file_path)

            if os.path.isfile(mask_src_path):
                shutil.copy(mask_src_path, new_masked_file_path)


def show_celebrity_dataset():
    celebrity_dict = get_celebrity_dataset()
    path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\CelebA\\Img\\img_align_celeba_png"
    mask_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\CelebA\\Img\\img_align_celeba_png_masked_old"
    new_dataset_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\CelebA\\Img\\img_align_celeba_with_masks"

    for key in celebrity_dict.keys():
        images = celebrity_dict.get(key)
        for image_path in images:
            png_path = f"{image_path[:-4]}.png"
            i_path = f"{path}\\{png_path}"
            image = cv2.imread(i_path)
            cv2.imshow(key, image)
            cv2.waitKey()


def get_total_image_count():
    dataset_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\CelebA\\Img\\img_align_celeba_with_masks"
    total_images = 0
    for folder in tqdm(os.listdir(dataset_path)):
        folder_path = f"{dataset_path}\\{folder}"
        files = [f for f in os.listdir(folder_path) if os.path.isfile(f"{folder_path}\\{f}")]
        total_images += len(files)

    print(total_images)


def get_used_identities_and_images():
    landmark_file = f"../inputs/celeb_a_1_list.txt"

    with open(landmark_file, 'r') as f:
        lines = f.readlines()

    identities = dict()
    images = dict()

    for line in tqdm(lines):
        reference = line.split(" ")[0]
        probe = line.split(" ")[1]
        reference_folder = reference.split("/")[1]
        probe_folder = probe.split("/")[1]
        identities[reference_folder] = True
        identities[probe_folder] = True

        images[reference] = True
        images[probe] = True

    print("images", len(images.keys()), "identities", len(identities.keys()))


if __name__ == '__main__':
    # show_celebrity_dataset()
    # generate_celeba_dataset_with_masks()
    # generate_csv_for_celebrity_dataset()
    get_used_identities_and_images()
