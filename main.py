import argparse

from utils.GeneralUtils import *
from ExtraSensoryModules.ExtraSensoryManager import ExtraSensoryManager
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

LOG_PATH = r".\log\classifier.log"


def get_arguments():
    parser = argparse.ArgumentParser(description='This program preprocess, train and evaluate model and data base on '
                                                 'the Recognizing Detailed Human Context article.')
    parser.add_argument('-p', '--preprocess', type=bool, help='flag for preprocess the data', default=False)
    parser.add_argument('-t', '--train', type=bool, help='flag for training models', default=False)
    parser.add_argument('-e', '--eval', type=bool, help='flag for evaluate and test models', default=True)
    parser.add_argument('-l', '--learn_params', type=bool, help='flag for learning parameters for the a model', default=False)
    parser.add_argument('-a', '--all_data', type=str, help='flag for performing grid search on all data', default=False)
    args = parser.parse_args()
    return args


def main():
    arguments = get_arguments()
    setup_custom_logger(LOG_PATH, 'classifier')
    extra_sensory = ExtraSensoryManager()
    extra_sensory.run(arguments)


if __name__ == '__main__':
    main()
