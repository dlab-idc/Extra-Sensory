import pandas as pd

from utils.GeneralUtils import *
from preprocessing.PreProcessing import PreProcess
from ExtraSensoryModels.HyperParameterLearner import HyperParameterLearner
from ExtraSensoryModels.ClassifierTrainer import ClassifierTrainer
from ExtraSensoryModels.Models import early_fusion, late_fusion_averaging, late_fusion_learning, single_sensor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from logging import getLogger
from utils.ReadingTheDataUtils import get_dataframe


class ExtraSensory:
    def __init__(self):
        self.config = ConfigManager.get_config('extra_sensory')
        general_config = ConfigManager.get_config('General')
        self.logger = getLogger('classifier')
        self.names = self.config['models']['names']
        self.params = self.config['models']['params']
        self.directories_dict = general_config['directories']
        self.format_dict = general_config['formats']
        #self.learning_params = self.config['models']['learning_params']
        self.fold_number = general_config['folds']['fold_number']
        self.is_fold = general_config['folds']['is_fold']
        self.hyper_parameter_learner = HyperParameterLearner()
        self.preprocess = PreProcess()
        self.classifier_trainer = ClassifierTrainer()

    def run(self, arguments):
        if arguments.preprocess:
            self.preprocess.create_data_set()
        if arguments.train:
            if self.is_fold:
                self.create_model_per_fold(arguments)
            else:
                for name in self.names:
                    self.create_model(arguments, name)
        # if arguments.eval:
        #     pass

    def create_model_per_fold(self, arguments):
        for model_number in range(self.fold_number):
            for name in self.names:
                self.create_model(arguments, name, model_number)

    def create_model(self, arguments, model_name, model_number=None):
        train_fold, test_fold = self.get_folds_files_names(model_number)
        self.logger.info(f"Reading train {train_fold}")
        train_df = get_dataframe(train_fold)
        if self.hyper_parameter_learner.is_all_set:
            self.logger.info(f"Training on all data set. Reading test {test_fold}")
            test_df = get_dataframe(test_fold)
            train_df = pd.concat([train_df, test_df])
        model, params = self.get_extra_sensory_model(model_name)
        if arguments.learn_params:
            params = self.hyper_parameter_learner.async_grid_search(train_df.copy(), model)
        model.set_params(**params)
        self.classifier_trainer.model_name = f'{model_name}_{model_number}' if model_number else model_name
        self.classifier_trainer.train_model(train_df, model)

    def get_folds_files_names(self, fold_number):
        if fold_number:
            train_fold = self.format_dict['fold_file'].format("train", fold_number)
            test_fold = self.format_dict['fold_file'].format("test", fold_number)
        else:
            train_fold = os.path.join(self.directories_dict['fold'], 'train.csv')
            test_fold = os.path.join(self.directories_dict['fold'], 'test.csv')
        return train_fold, test_fold

    def get_extra_sensory_model(self, name):
        params = self.params[name]
        estimator_name = params['model_params'].pop('estimator')
        estimator = self.get_sklearn_model(estimator_name)
        self.hyper_parameter_learner.set_estimator(estimator_name)
        model = None
        if name in 'early_fusion':
            model = early_fusion.EarlyFusion(estimator)
        elif name in 'late_fusion_averaging':
            model = late_fusion_averaging.LateFusionAveraging(estimator)
        elif name in 'late_fusion_learning':
            model = late_fusion_learning.LateFusionLearning(estimator)
        elif name in 'single_sensor':
            model = single_sensor.SingleSensor(estimator)
        return model, params

    @staticmethod
    def get_sklearn_model(name):
        estimator = None
        if name in 'logistic_regression':
            estimator = LogisticRegression()
        elif name in 'random_forest':
            estimator = RandomForestClassifier()
        else:
            raise Exception("Invalid sklearn model name")
        return estimator
