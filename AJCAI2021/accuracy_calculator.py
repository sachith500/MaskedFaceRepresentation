import numpy as np


class MAECalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate(output_results):
        total_correct = 0
        total_correctsq = 0
        total_result = 0
        for result in output_results:
            correct = (abs(result[0] - result[1])).sum()
            correct_sq = (abs(result[0] - result[1]) ** 2).sum()

            total_correct += correct
            total_correctsq += correct_sq
            total_result += len(result[0])

        print("MAE = " + str(total_correct / total_result))
        print("RMSE = " + str((total_correctsq / total_result) ** 0.5))
        return [["MAE", str(total_correct / total_result)], ["RMSE", str((total_correctsq / total_result) ** 0.5)]]


class PercentageCalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate(output_results):
        total_result_count = 0
        total_correct_count = 0

        for result in output_results:
            result = np.array(result)
            total_result_count += len(result[0])
            correct_count = np.sum(result[0] == result[1])
            total_correct_count += correct_count

        print("Accuracy = " + str(total_correct_count) + "/" + str(total_result_count))
        print(" ===  " + str(total_correct_count / total_result_count) + "  ===")

        return [["Accuracy", str(100 * total_correct_count / total_result_count) + "%"],
                ["Accuracy", str(total_correct_count / total_result_count)]]
