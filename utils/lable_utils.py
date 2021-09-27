import os


def __get_evaluation_line_details(line):
    reference, probe, label_reference, label_probe, true_label = line.split("::")
    key = f"{reference.strip()}_{probe.strip()}_{label_reference.strip()}_{label_probe.strip()}"
    return key, true_label.strip()


def get_true_labels(evaluation_file):
    if os.path.isfile(evaluation_file):
        with open(evaluation_file, 'r', newline='') as file:
            evaluation_lines = file.readlines()

        input_true_labels = []

        for line in evaluation_lines:
            evaluation_line, true_label = __get_evaluation_line_details(line)
            input_true_labels.append(true_label)

        return input_true_labels

    return []


def get_similarity_scores(score_file):
    similarity_scores = []
    if os.path.isfile(score_file):
        with open(score_file, 'r', newline='') as file:
            scores = file.readlines()

            for score in scores:
                score = float(score)
                similarity_scores.append(score)

    return similarity_scores
