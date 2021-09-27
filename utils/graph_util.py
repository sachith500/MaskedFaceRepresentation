import os

import cv2
import numpy as np


def err(FARs, FRRs):
    # find eer
    min_abs_diff = 5
    min_abs_diff_id = -1
    for i in range(0, FARs.__len__()):
        abs_diff = np.abs(FARs[i] - FRRs[i])
        if abs_diff < min_abs_diff:
            min_abs_diff = abs_diff
            min_abs_diff_id = i
    # print(min_abs_diff_id, min_abs_diff)
    err = (FARs[min_abs_diff_id] + FRRs[min_abs_diff_id]) / 2.0
    return (err)


def get_dist(references, probes):
    return np.linalg.norm(references - probes, axis=1)


def min_max_normalize(distance_matrix):
    temp = distance_matrix - np.min(distance_matrix)
    return temp / np.max(distance_matrix)


# call with two arrays one containing the reference features and the other containing the corresponding probe features
def get_similarity_scores(references, probes):
    if len(references) != len(probes):
        print("references and probes do not match in size")
        return

    distances = get_dist(references, probes)
    similarities = 1 / (1 + distances)
    return similarities


def save_reference_and_probe(dataset_base_folder, image_1_path, image_2_path, reference_label, probel_label,
                             match_status, sim, thresh, test_image_saving_folder):
    threshold_folder = f"{test_image_saving_folder}/{str(thresh)[:5]}"
    if not os.path.isdir(test_image_saving_folder):
        os.mkdir(test_image_saving_folder)
    if not os.path.isdir(threshold_folder):
        os.mkdir(threshold_folder)

    status_color = (0, 0, 255)

    image_1 = cv2.imread(f"{dataset_base_folder}/{image_1_path}")
    image_2 = cv2.imread(f"{dataset_base_folder}/{image_2_path}")

    name = image_1_path.split("/")[-1][:-4]
    name += "_"
    name += image_2_path.split("/")[-1]

    name = name.replace("\\", "_")

    new_image_path = f"{threshold_folder}/{name}"

    if image_1.shape != image_2.shape:
        r = image_1.shape[0]
        c = image_1.shape[1]

        if r < image_2.shape[0]:
            r = image_2.shape[0]

        if c < image_2.shape[1]:
            c = image_2.shape[1]

        image_1 = cv2.resize(image_1, (r, c))
        image_2 = cv2.resize(image_2, (r, c))

    image = np.concatenate((image_1, image_2), axis=1)
    image = cv2.putText(image, match_status, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 1,
                        cv2.LINE_AA)
    image = cv2.putText(image,
                        f"reference {reference_label} probe : {probel_label} similarity : {sim} threshold : {thresh}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(new_image_path, image)


# similarity_score (for all test pairs): range [0,1], 0 - dissimilar, 1 - similar
# actual label: 0 negative pair, 1 positive pair
def evaluate(similarity_score, actual_label):
    if len(similarity_score) != len(actual_label):
        print("similarity score and actual labels do not match in size")
        return

    thresholds = np.linspace(1, 0, num=500)

    FRRs = []
    FARs = []
    thresholds_list = []
    thresholds_false_images_dict = dict()
    for thresh in thresholds:
        false_non_match = 0.0
        true_non_match = 0.0
        false_match = 0.0
        true_match = 0.0

        data_row = []
        for idx, sim in enumerate(similarity_score):
            if sim > thresh:  # decision - same user
                if actual_label[idx] == 1:  # actual - same user
                    true_match += 1
                else:  # actual - different users
                    false_match += 1

            if sim <= thresh:  # decision - different users
                if actual_label[idx] == 1:  # actual - same user
                    false_non_match += 1
                else:  # actual - different users
                    true_non_match += 1

        thresholds_false_images_dict[thresh] = data_row
        FRR = false_non_match / (false_non_match + true_match)  # divide by all correct samples
        FAR = false_match / (true_non_match + false_match)  # divide by all wrong samples
        # when thresh == 0, FRR = 0
        # when thresh == 1, FAR = 0

        FRRs.append(FRR)
        FARs.append(FAR)
        thresholds_list.append(thresh)

    fnmr_fmr100 = 1.0
    fnmr_fmr1000 = 1.0
    fnmr_fmr0 = 1.0

    for idx, fmr in enumerate(FARs):
        if fmr < 0.01:  # FMR100
            if fnmr_fmr100 > FRRs[idx]:
                fnmr_fmr100 = FRRs[idx]

        if fmr < 0.001:  # FMR1000
            if fnmr_fmr1000 > FRRs[idx]:
                fnmr_fmr1000 = FRRs[idx]
        if fmr == 0:  # FMR0
            if fnmr_fmr0 > FRRs[idx]:
                fnmr_fmr0 = FRRs[idx]

    # Calculate EER
    eer = err(FARs, FRRs)

    return [fnmr_fmr0, fnmr_fmr100, fnmr_fmr1000, eer, thresholds, FARs, FRRs]


def evaluate_GAR(similarity_score, actual_label):
    if len(similarity_score) != len(actual_label):
        print("similarity score and actual labels do not match in size")
        return

    thresholds = np.linspace(1, 0, num=500)

    GARs = []
    FMRs = []
    thresholds_list = []
    thresholds_false_images_dict = dict()
    for thresh in thresholds:
        false_non_match = 0.0
        true_non_match = 0.0
        false_match = 0.0
        true_match = 0.0

        data_row = []
        for idx, sim in enumerate(similarity_score):
            if sim > thresh:  # decision - same user
                if actual_label[idx] == 1:  # actual - same user
                    true_match += 1
                else:  # actual - different users
                    false_match += 1

            if sim <= thresh:  # decision - different users
                if actual_label[idx] == 1:  # actual - same user
                    false_non_match += 1
                else:  # actual - different users
                    true_non_match += 1

        thresholds_false_images_dict[thresh] = data_row
        GAR = true_match / (false_non_match + true_match)  # divide by all correct samples
        FMR = false_match / (true_non_match + false_match)  # divide by all wrong samples
        # when thresh == 0, FNMR = 0
        # when thresh == 1, FMR = 0

        GARs.append(GAR)
        FMRs.append(FMR)
        thresholds_list.append(thresh)

    return [thresholds, FMRs, GARs]


def evaluate_TP_FP_rates(similarity_score, actual_label):
    if len(similarity_score) != len(actual_label):
        print("similarity score and actual labels do not match in size")
        return

    thresholds = np.linspace(1, 0, num=500)
    TPRs = []
    FPRs = []
    thresholds_list = []
    thresholds_false_images_dict = dict()
    for thresh in thresholds:
        false_non_match = 0.0
        true_non_match = 0.0
        false_match = 0.0
        true_match = 0.0

        data_row = []
        for idx, sim in enumerate(similarity_score):
            if sim > thresh:  # decision - same user
                if actual_label[idx] == 1:  # actual - same user
                    true_match += 1
                else:  # actual - different users
                    false_match += 1

            if sim <= thresh:  # decision - different users
                if actual_label[idx] == 1:  # actual - same user
                    false_non_match += 1
                else:  # actual - different users
                    true_non_match += 1

        thresholds_false_images_dict[thresh] = data_row
        TPR = true_match / (false_non_match + true_match)  # divide by all correct samples
        FPR = false_match / (true_non_match + false_match)  # divide by all wrong samples
        # when thresh == 0, FNMR = 0
        # when thresh == 1, FMR = 0

        TPRs.append(TPR)
        FPRs.append(FPR)
        thresholds_list.append(thresh)

    return [thresholds, TPRs, FPRs]
