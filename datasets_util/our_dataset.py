import os


def generate_evaluation_list(dataset_folder, out_put_file_name):
    mask_dict = dict()
    no_mask_dict = dict()

    mask_directory = f"{dataset_folder}\\mask\\"
    no_mask_directory = f"{dataset_folder}\\no_mask\\"

    mask_images = os.listdir(mask_directory)
    no_mask_images = os.listdir(no_mask_directory)

    for image in mask_images:
        user_name = get_user_name(image)
        image_path = f"./mask\\{image}"
        mask_dict[user_name] = image_path

    for image in no_mask_images:
        user_name = get_user_name(image)
        image_path = f"./no_mask\\{image}"
        no_mask_dict[user_name] = image_path

    with open(out_put_file_name, 'w') as f:
        odd = False
        for key in mask_dict.keys():
            row = []

            if odd:
                row.append(mask_dict[key])
                row.append(no_mask_dict[key])
                row.append("1")
                row.append("0")
            else:
                row.append(no_mask_dict[key])
                row.append(mask_dict[key])
                row.append("0")
                row.append("1")

            print(' '.join(row) + "\n")
            f.write(' '.join(row) + "\n")
            odd = not odd


def rename_our_dataset(dataset_folder):
    mask_directory = f"{dataset_folder}\\mask\\"
    no_mask_directory = f"{dataset_folder}\\no_mask\\"

    mask_images = os.listdir(mask_directory)
    no_mask_images = os.listdir(no_mask_directory)

    for image in mask_images:
        image_path = f"{mask_directory}\\{image}"
        new_name = image.replace(" ", "_")
        new_image_path = f"{mask_directory}\\{new_name}"
        os.rename(image_path, new_image_path)

    for image in no_mask_images:
        image_path = f"{no_mask_directory}\\{image}"
        new_name = image.replace(" ", "_")
        new_image_path = f"{no_mask_directory}\\{new_name}"
        os.rename(image_path, new_image_path)


def get_user_name(image_file_name):
    return image_file_name.split("-")[-1].strip()


if __name__ == "__main__":
    dataset_folder = "D:\MaskedFaceRecognitionCompetition\dataset\evaluation_datasets\our_dataset"
    out_put_file_name = "../outputs/evaluation_inut_our_dataset.txt"
    generate_evaluation_list(dataset_folder, out_put_file_name)
    # rename_our_dataset(dataset_folder)
