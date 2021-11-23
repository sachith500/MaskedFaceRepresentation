import argparse

from AJCAI2021.pipeline import Pipeline

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="Calculate age, race and gender accuracy",
                                         description='Calculate age, race and gender accuracy', )
    arg_parser.add_argument('--type', action='store', type=str, default='age_classification')

    args = arg_parser.parse_args()

    pipeline = Pipeline(args.type)
    pipeline.process()
