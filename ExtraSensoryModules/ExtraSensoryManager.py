import pandas as pd
import numpy as np

from utils.GeneralUtils import *
from ExtraSensoryModules.PreProcessing import PreProcess
from ExtraSensoryModules.HyperParameterLearner import HyperParameterLearner
from ExtraSensoryModules.ClassifierTrainer import ClassifierTrainer
from ExtraSensoryModels.Models import early_fusion  # , late_fusion_averaging, late_fusion_learning, single_sensor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from logging import getLogger
from utils.ReadingTheDataUtils import get_dataframe
from ExtraSensoryModules.Evaluator import Evaluator

NUM_OF_LABELS = 6
CONFUSION_MATRIX_LABELS = 4

# Extra sensory models names
EARLY_FUSION = 'early-fusion'

# Sklearn estimators names
LOGISTIC_REGRESSION = 'logistic-regression'
RANDOM_FOREST = 'random-forest'


class ExtraSensoryManager:

    # region Constrictor
    def __init__(self):
        self.config = ConfigManager.get_config('General')
        self.logger = getLogger('classifier')
        self.models_types = self.config['models']['names']
        self.params = self.config['models']['params']
        self.directories_dict = self.config['directories']
        self.format_dict = self.config['formats']
        self.fold_number = self.config['folds']['fold_number']
        self.is_fold = self.config['folds']['is_fold']
        self.feature_selection_percent = self.config['preprocessing']['feature_selection_percent']
        self.hyper_parameter_learner = HyperParameterLearner(self.config['folds']['grid_search_method'])
        self.preprocess = PreProcess()
        self.classifier_trainer = ClassifierTrainer()
        self.evaluator = Evaluator(NUM_OF_LABELS)

    # endregion

    # region Methods
    def run(self, arguments):
        self.create_directories()
        if arguments.preprocess:
            self.preprocess.create_data_set()
        if arguments.train:
            if self.is_fold:
                self.logger.info("Training models per fold")
                self.create_model_per_fold(arguments)
            else:
                self.logger.info("Training one model")
                for name in self.models_types:
                    self.logger.info(f"Training {name}")
                    self.create_model(arguments, name)
        if arguments.eval:
            if self.is_fold:
                self.eval_model_per_fold()
            else:
                for name in self.models_types:
                    self.eval_model(name)

    def get_folds_files_path(self, fold_number):
        if fold_number is None:
            train_fold = os.path.join(self.directories_dict['fold'], 'train_3.csv')
            test_fold = os.path.join(self.directories_dict['fold'], 'test_3.csv')
        else:
            train_file_name = self.format_dict['fold_file'].format("train", self.feature_selection_percent, fold_number)
            test_file_name = self.format_dict['fold_file'].format("test", self.feature_selection_percent, fold_number)
            fold_subdir = os.path.join(self.directories_dict['fold'], str(self.feature_selection_percent))
            train_fold = os.path.join(fold_subdir, train_file_name)
            test_fold = os.path.join(fold_subdir, test_file_name)
        return train_fold, test_fold

    def get_model_name(self, model_name, estimator_name, model_number):
        """
        Set the name of the model to be {model_name}_{estimator_name}_{feature_selection_percent}_{grid_search_method}
        _{model_number}
        :param model_number:
        :param estimator_name:
        :param model_name:
        :return:
        """
        grid_search_method = self.config['folds']['grid_search_method']
        model_name = f"{model_name}_{estimator_name}_{self.feature_selection_percent}_{grid_search_method}"
        model_name = model_name if model_number is None else f'{model_name}_{model_number}'
        return model_name

    def create_directories(self):
        for directory in self.directories_dict.values():
            if not os.path.exists(directory):
                os.mkdir(directory)

    # endregion

    # region Create Models
    def create_model_per_fold(self, arguments):
        for model_number in range(self.fold_number):
            for name in self.models_types:
                self.create_model(arguments, name, model_number)

    def create_model(self, arguments, model_name, model_number=None):
        train_df = self.get_train_data(arguments, model_number)
        model, model_name, params = self.get_model_and_params(model_name, model_number)
        if arguments.learn_params:
            params = self.hyper_parameter_learner.async_grid_search(train_df.copy(), model, model_name)
        model.set_params(**params)
        self.classifier_trainer.train_model(train_df, model)

    def get_model_and_params(self, model_name, model_number):
        model, params, estimator_name = self.get_extra_sensory_model(model_name)
        self.logger.info(f"Training {model_name}_{estimator_name}")
        self.classifier_trainer.model_name = self.get_model_name(model_name, estimator_name, model_number)
        return model, self.classifier_trainer.model_name, params

    def get_train_data(self, arguments, model_number):
        train_fold, test_fold = self.get_folds_files_path(model_number)
        self.logger.info(f"Reading train {train_fold}")
        train_df = get_dataframe(train_fold)
        if self.hyper_parameter_learner.is_all_set and arguments.learn_params:
            self.logger.info(f"Training on all data set. Reading test {test_fold}")
            test_df = get_dataframe(test_fold)
            train_df = pd.concat([train_df, test_df])
        return train_df

    def get_extra_sensory_model(self, name):
        params = self.params[name].copy()
        params['model_params'] = params['model_params'].copy()
        estimator_name = params['model_params'].pop('estimator')
        estimator = self.get_sklearn_model(estimator_name)
        self.hyper_parameter_learner.set_estimator(estimator_name)
        model = None
        if name in EARLY_FUSION:
            model = early_fusion.EarlyFusion(estimator)
        # elif name in 'late_fusion_averaging':
        #     model = late_fusion_averaging.LateFusionAveraging(estimator)
        # elif name in 'late_fusion_learning':
        #     model = late_fusion_learning.LateFusionLearning(estimator)
        # elif name in 'single_sensor':
        #     model = single_sensor.SingleSensor(estimator)
        return model, params, estimator_name

    @staticmethod
    def get_sklearn_model(name):
        if name in LOGISTIC_REGRESSION:
            estimator = LogisticRegression()
        elif name in RANDOM_FOREST:
            estimator = RandomForestClassifier()
        else:
            raise Exception("Invalid sklearn model name")
        return estimator

    # endregion

    # region Eval Models

    def eval_model_per_fold(self):
        for model_name in self.models_types:
            model_accumulating_state = self.get_states_arrays(model_name)
            test_class_weights = np.zeros(NUM_OF_LABELS)
            estimator_name = self.params[model_name]['model_params']['estimator']
            for model_number in range(self.fold_number):
                self.evaluator.model_name = self.get_model_name(model_name, estimator_name, model_number)
                last_state, last_class_weights = self.eval_model(model_number)
                model_accumulating_state += last_state
                test_class_weights += last_class_weights
                test_class_weights = (last_class_weights / last_class_weights.sum())
                self.create_results(last_state, self.evaluator.model_name, test_class_weights)
            test_class_weights = (test_class_weights / test_class_weights.sum())
            self.create_results(model_accumulating_state, self.evaluator.model_name[:-2], test_class_weights,
                                is_fold=False)

    def eval_model(self, model_number=None):
        train_fold, test_fold = self.get_folds_files_path(model_number)
        self.logger.info(f"Reading test {test_fold}")
        test_df = get_dataframe(test_fold)
        state_matrix, test_class_weights = self.evaluator.eval(test_df)
        return state_matrix, test_class_weights

    def create_results(self, model_accumulating_state, model_name, test_class_weights, is_fold=True):
        macro_df = self.create_macro_results(model_accumulating_state, model_name)
        micro_df = self.create_micro_results(model_accumulating_state, model_name)
        weighted_df = self.create_wheighted_results(model_accumulating_state, model_name, test_class_weights)
        results = pd.concat([macro_df, micro_df, weighted_df])
        results = results.set_index('Test type')
        results_path = self.get_model_subdir(model_name, is_fold=is_fold)
        results.to_csv(results_path)

    def get_model_subdir(self, model_name, is_fold=True):
        attributes = model_name.split('_')[:-1] if is_fold else model_name.split('_')
        file_name = f"state_results_{self.format_dict['model_file'].format(model_name)}.csv"
        attributes.append(file_name)
        path = f"{os.path.sep}".join(attributes)
        results_path = os.path.join(self.directories_dict['results'], path)
        if not os.path.exists(os.path.dirname(results_path)):
            os.makedirs(os.path.dirname(results_path))
        return results_path

    def create_micro_results(self, model_accumulating_state, model_name):
        scores_array = model_accumulating_state.sum(axis=1)
        TP, TN, FP, FN = scores_array[0], scores_array[1], scores_array[2], scores_array[3]
        scores_array = self.get_evaluations_metric_scores(TP, TN, FP, FN)
        return self.create_result_dict(scores_array, model_name, 'micro')

    def create_macro_results(self, model_accumulating_state, model_name):
        scores_array = np.zeros((NUM_OF_LABELS,), dtype='int')
        for labels_states in range(NUM_OF_LABELS):
            labels_states = model_accumulating_state[:, labels_states]
            TP, TN, FP, FN = labels_states[0], labels_states[1], labels_states[2], labels_states[3]
            scores_array = scores_array + self.get_evaluations_metric_scores(TP, TN, FP, FN)
        scores_array = scores_array / NUM_OF_LABELS
        return self.create_result_dict(scores_array, model_name, 'macro')

    def create_wheighted_results(self, model_accumulating_state, model_name, test_class_weights):
        scores_array = []
        for labels_states in range(NUM_OF_LABELS):
            labels_states = model_accumulating_state[:, labels_states]
            TP, TN, FP, FN = labels_states[0], labels_states[1], labels_states[2], labels_states[3]
            scores_array.append(self.get_evaluations_metric_scores(TP, TN, FP, FN))
        scores_array = np.dot(test_class_weights, np.array(scores_array))
        return self.create_result_dict(scores_array, model_name, 'weighted')

    @staticmethod
    def get_states_arrays(name):
        states_shape = (CONFUSION_MATRIX_LABELS, NUM_OF_LABELS)
        model_states = np.zeros(states_shape, dtype='int')
        if name in 'single sensor':
            # feature_names = get_feature_names(i_standard_X_test,
            #                                   ['label'])  # In this case we using the data with our label!
            # sensor_names = get_sensor_names(feature_names)
            # single_sensors_states_dict = dict()
            pass
        # for sensor_name in sensor_names:
        #     single_sensors_states_dict.setdefault(sensor_name, np.zeros(states_shape))
        return model_states

    @staticmethod
    def get_evaluations_metric_scores(TP, TN, FP, FN):
        """
        Compute the performance evaluation of a classifier
        according to the metrics that were presented in the article

        :param TP: true positive
        :param FN: false negative
        :param TN: true negative
        :param FP: false positive
        :return: python tuple containing:
                TPR: true positive rate, also known as sensitivity or recall.
                        proportion of positive examples that correctly classified as positive
                TNR: true negative rate, also known as sensitivity.
                        proportion of negative examples that correctly classified as negative
                accuracy: proportion of correctly classified examples out of all examples
                precision: proportion of correctly classified examples out of the positive declared examples
                BA: balanced accuracy. measure that consider both TPR and TNR
                F1: the harmonic mean of recall(TPR) and precision

        """
        TPR = TP / (TP + FN)  # sensitivity/recall
        TNR = TN / (TN + FP)  # specifisity
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP)
        BA = (TPR + TNR) / 2
        F1 = (2 * TPR * precision) / (TPR + precision)

        return TPR, TNR, accuracy, precision, BA, F1

    @staticmethod
    def create_result_dict(scores_array, model_name, test_type):
        results_dict = {
            "Test type": test_type,
            "Classifier": [model_name],
            "Accuracy": [scores_array[2]],
            "Sensitivity": [scores_array[0]],
            "Specifisity": [scores_array[1]],
            "BA": [scores_array[4]],
            "Precision": [scores_array[3]],
            "F1": [scores_array[5]]
        }
        results = pd.DataFrame.from_dict(results_dict)
        return results

    # endregion
