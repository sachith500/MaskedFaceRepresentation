import argparse

from product.pipeline import Pipeline

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="Face similarity verification application",
                                         description='Verify the similarity of the evaluation files.', )
    arg_parser.add_argument('input',
                            action='store',
                            nargs='*')

    args = arg_parser.parse_args()

    if len(args.input) > 3:
        print("Please input evaluation_list.txt, landmarks.txt, and output_path only.")
    elif len(args.input) < 3:
        print("One or more of the evaluation_list.txt, landmarks.txt, and output_path are missing.")
    else:
        evaluation_list = args.input[0].strip()
        landmarks = args.input[1].strip()
        output_path = args.input[2].strip()

        pipeline = Pipeline(evaluation_list, landmarks, output_path)
        pipeline.process()
