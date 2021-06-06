import os


def __get_evaluation_line_details(line):
    reference, probe, label_reference, label_probe, true_label = line.split(" ")
    key = f"{reference.strip()}_{probe.strip()}_{label_reference.strip()}_{label_probe.strip()}"
    return key, true_label.strip()


def __get_input_evaluation_line_details(line):
    reference, probe, label_reference, label_probe = line.split(" ")
    key = f"{reference.strip()}_{probe.strip()}_{label_reference.strip()}_{label_probe.strip()}"
    return key


def get_true_labels(evaluation_file, input_evaluation_file):
    if os.path.isfile(input_evaluation_file):
        with open(input_evaluation_file, 'r', newline='') as file:
            input_evaluation_lines = file.readlines()

        if os.path.isfile(evaluation_file):
            with open(evaluation_file, 'r', newline='') as file:
                evaluation_lines = file.readlines()

            input_index = 0
            evaluation_index = 0
            input_true_labels = []

            while input_index < len(input_evaluation_lines):

                input_line = __get_input_evaluation_line_details(input_evaluation_lines[input_index])
                evaluation_line, true_label = __get_evaluation_line_details(evaluation_lines[evaluation_index])

                if input_line == evaluation_line:
                    input_true_labels.append(true_label)
                    evaluation_index += 1
                    input_index += 1
                else:
                    evaluation_index += 1

                if len(evaluation_lines) < evaluation_index:
                    print("something wrong with evaluation list.")
                    break
            return input_true_labels
        else:
            return []
    else:
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
