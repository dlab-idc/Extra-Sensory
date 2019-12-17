import argparse

from utils.GeneralUtils import *
from ExtraSensoryModels.Models import early_fusion, late_fusion_averaging, late_fusion_learning, single_sensor
from extra_sensory import ExtraSensory
from preprocessing.PreProcessing import PreProcess
from ExtraSensoryModels.ClasifaierMaker import ClassifierMaker
from sklearn.linear_model import LogisticRegression

LOG_PATH = r".\log\classifier.log"


def get_arguments():
    parser = argparse.ArgumentParser(description='This program preprocess, train and evaluate model and data base on '
                                                 'the Recognizing Detailed Human Context article.')
    parser.add_argument('-p', '--preprocess', type=bool, help='flag for preprocess the data', default=False)
    parser.add_argument('-t', '--train', type=bool, help='flag for training models', default=True)
    parser.add_argument('-e', '--eval', type=bool, help='flag for evaluate and test models', default=False)
    parser.add_argument('-l', '--learn', type=bool, help='flag for learning parameters for the a model', default=False)
    args = parser.parse_args()
    return args


def main():
    arguments = get_arguments()
    setup_custom_logger(LOG_PATH, 'classifier')
    extra_sensory = ExtraSensory()
    extra_sensory.run(arguments)


if __name__ == '__main__':
    main()
