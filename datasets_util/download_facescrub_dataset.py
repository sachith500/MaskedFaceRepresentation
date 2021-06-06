import os

import cv2
import requests
from tqdm import tqdm


def read_facescrub_data(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    celebrity_dict = {}
    for line in lines:
        data = line.split("\t")
        name = data[0]
        image_id = data[1]
        url = data[3]
        if url == 'url':
            continue
        data_array = celebrity_dict.get(name)
        if data_array is None:
            data_array = []
        data_array.append((image_id, url))
        celebrity_dict[name] = data_array

    return celebrity_dict


skip_list = ['Al Pacino', 'Adrien Brody', 'Aaron Eckhart', 'Adam Brody', 'Adam McKay', 'Adam Sandler']


def download_images(dataset_file, new_dataset_path):
    celebrity_dict = read_facescrub_data(dataset_file)
    if not os.path.isdir(new_dataset_path):
        os.mkdir(new_dataset_path)

    for celebrity in tqdm(celebrity_dict.keys()):
        if celebrity in skip_list:
            continue
        folder_name = celebrity.replace(" ", "_").replace(".", "_")
        folder_path = f"{new_dataset_path}\\{folder_name}"
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        data = celebrity_dict[celebrity]
        for pair in tqdm(data):
            image_id = pair[0]
            image_path = f"{folder_path}\\{image_id}.jpg"
            if os.path.isfile(image_path):
                try:
                    i = cv2.imread(image_path)
                    if i is not None:
                        continue
                except Exception as e:
                    print(e)
            url = pair[1]
            try:
                with open(image_path, 'wb') as f:
                    f.write(requests.get(url=f"{url}").content)
            except Exception as e:
                print(url)
                print(e)


if __name__ == '__main__':
    dataset_file = '../inputs/facescrub_actresses'
    dataset_path = "D:\\MaskedFaceRecognitionCompetition\\dataset\\facescrub\\"

    download_images(dataset_file, dataset_path)
