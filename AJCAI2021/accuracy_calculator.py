import numpy as np


class MAECalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate(output_results):
        correct_total = 0
        total_result = 0

        total_correct = 0
        total_correctsq = 0

        for result in output_results:
            correct = (abs(result[0] - result[1])).float().sum()
            correct_sq = (abs(result[0] - result[1]) ** 2).float().sum()

            total_correct += correct
            total_correctsq += correct_sq

        print(correct_total, total_result)
        print("MAE = " + str(correct_total / total_result))
        print("RMSE = " + str((correct_total / total_result) ** 0.5))


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
            correct_count = (result[0] == result[1]).sum()
            total_correct_count += correct_count

        print("Accuracy = " + str(total_correct_count) + "/" + str(total_result_count))
        print(" ===  " + str(total_correct_count / total_result_count) + "  ===")
