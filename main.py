import argparse

from utils.GeneralUtils import *
from extra_sensory import ExtraSensory

LOG_PATH = r".\log\classifier.log"


def get_arguments():
    parser = argparse.ArgumentParser(description='This program preprocess, train and evaluate model and data base on '
                                                 'the Recognizing Detailed Human Context article.')
    parser.add_argument('-p', '--preprocess', type=bool, help='flag for preprocess the data', default=False)
    parser.add_argument('-t', '--train', type=bool, help='flag for training models', default=True)
    parser.add_argument('-e', '--eval', type=bool, help='flag for evaluate and test models', default=False)
    parser.add_argument('-l', '--learn_params', type=bool, help='flag for learning parameters for the a model', default=True)
    parser.add_argument('-e', '--estimator', type=str, help='the name of the sklearn estimator', default='early_fusion')
    parser.add_argument('-m', '--model', type=str, help='the name of the article model', default='logistic_regression')
    args = parser.parse_args()
    return args


def main():
    arguments = get_arguments()
    setup_custom_logger(LOG_PATH, 'classifier')
    extra_sensory = ExtraSensory()
    extra_sensory.run(arguments)


if __name__ == '__main__':
    main()
