import os

dataset_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\evaluation_datasets\\RMFD"


def get_files(folder_path):
    files = []
    for file in os.listdir(folder_path):
        files.append(file)
    return files


for person in os.listdir(dataset_path):
    person_folder_path = f"{dataset_path}//{person}"

    files = get_files(person_folder_path)

    for file in files:
        file_path = f"{person_folder_path}\\{file}"
        # new_file_name = re.sub('[^0-9a-zA-Z]+', '_', file)
        new_file_name = file[:-4] + "." + file[-3:]
        new_file_path = f"{person_folder_path}\\{new_file_name}"
        # print(file_path, new_file_path)
        os.rename(file_path, new_file_path)
