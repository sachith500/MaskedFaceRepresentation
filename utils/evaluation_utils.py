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


# similarity_score (for all test pairs): range [0,1], 0 - dissimilar, 1 - similar
# actual label: 0 negative pair, 1 positive pair
def evaluate(similarity_score, actual_label):
    if len(similarity_score) != len(actual_label):
        print("similarity score and actual labels do not match in size")
        return

    thresholds = np.linspace(1, 0, num=50)

    FNMRs = []
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
        FNMR = false_non_match / (false_non_match + true_match)  # divide by all correct samples
        FMR = false_match / (true_non_match + false_match)  # divide by all wrong samples
        # when thresh == 0, FNMR = 0
        # when thresh == 1, FMR = 0

        FNMRs.append(FNMR)
        FMRs.append(FMR)
        thresholds_list.append(thresh)

    fnmr_fmr100 = 1.0
    fnmr_fmr1000 = 1.0
    fnmr_fmr0 = 1.0

    for idx, fmr in enumerate(FMRs):
        if fmr < 0.01:  # FMR100
            if fnmr_fmr100 > FNMRs[idx]:
                fnmr_fmr100 = FNMRs[idx]

        if fmr < 0.001:  # FMR1000
            if fnmr_fmr1000 > FNMRs[idx]:
                fnmr_fmr1000 = FNMRs[idx]
        if fmr == 0:  # FMR0
            if fnmr_fmr0 > FNMRs[idx]:
                fnmr_fmr0 = FNMRs[idx]

    # Calculate EER

    eer = err(FMRs, FNMRs)

    return [fnmr_fmr0, fnmr_fmr100, fnmr_fmr1000, eer]
