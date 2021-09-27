import os
import random

base_path = "./"
base_folder = "D:\\MaskedFaceRecognitionCompetition\\dataset\\evaluation_datasets"
#
datasets = ["RMFD"]


def get_files(folder_path):
    files = []
    for file in os.listdir(folder_path):
        files.append(file)
    return files


for dataset in datasets:
    dataset_path = f"{base_folder}\\{dataset}"
    out_put_file_name = f"data/evaluation_{dataset}.txt"


    def get_random_user_image(current_user, person_list):
        random_person = random.choice(person_list)
        if random_person != current_user:
            random_person_path = f"{dataset_path}/{random_person}"
            images = get_files(random_person_path)
            if len(images) == 0:
                return get_random_user_image(current_user, person_list)
            else:
                random_image = random.choice(images)
                return f"{base_path}/{random_person}/{random_image}"
        else:
            return get_random_user_image(current_user, person_list)


    def get_same_person_different_image(image, image_list):
        new_list = image_list.copy()
        new_list.remove(image)
        random_image = random.choice(new_list)
        if image != random_image:
            return random_image
        else:
            return get_same_person_different_image(image, image_list)


    filtered_person_directory_names = []
    filtered_single_person_directory_names = []
    data_row = []
    input_data_row = []

    filtered_person = []
    for person in os.listdir(dataset_path):
        person_folder_path = f"{dataset_path}//{person}"
        files = os.listdir(person_folder_path)

        if len(files) > 1:
            filtered_person.append(person)

    for person in filtered_person:
        person_folder_path = f"{dataset_path}//{person}"

        files = get_files(person_folder_path)

        if len(files) > 1:
            filtered_person_directory_names.append(person_folder_path)
            for image in files:
                row_image = f"{base_path}/{person}/{image}"

                different_image = get_same_person_different_image(image, files)
                different_image = f"{base_path}/{person}/{different_image}"

                random_user_image = get_random_user_image(person, filtered_person)

                data_row.append([row_image, different_image, '1', '1', '1'])  # Similar = 1
                data_row.append([row_image, random_user_image, '1', '1', '0'])

        else:
            pass
            # filtered_single_person_directory_names.append(person_folder_path)
            # for image in files:
            #     row_image = f"{base_path}/{person}/{image}"
            #     random_user_image = get_random_user_image(person, filtered_person)
            #
            #     data_row.append([row_image, random_user_image, '1', '1', '0'])  # Similar = 1

    print(len(filtered_person_directory_names))
    print(len(filtered_single_person_directory_names))

    with open(out_put_file_name, 'w', encoding='utf8') as f:
        for row in data_row:
            print('::'.join(row) + "\n")
            data_row = '::'.join(row) + "\n"
            f.write(data_row)

# Celebrity dataset
