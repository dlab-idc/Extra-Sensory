import logging
import os

import pandas as pd
import itertools

from utils.GeneralUtils import ConfigManager
from utils.TransformerUtils import get_X_y
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from ExtraSensoryModels.Interfaces.ExtraSensoryAbstractModel import ExtraSensoryAbstractModel

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class HyperParameterLearner:
    def __init__(self, grid_search_groups_method):
        self.config = ConfigManager.get_config('hyper_parameters_learner')
        self.directories = ConfigManager.get_config('General')['directories']
        self.logger = logging.getLogger('classifier')
        self.param_grid = None
        self.cross_validation_folds_number = None
        self.scoring_function = None
        self.grid_search_groups_method = grid_search_groups_method
        self.is_all_set = self.config['groups']['is_all_data']

    def async_grid_search(self, train, model: ExtraSensoryAbstractModel, model_name):
        """
        This function finds the best params for an estimator using greed search.
        Cross validation process is done by group K folds by uuid of the person.
        :param train: array like
        :param model: an Extra sensory model
        :return: dictionary, train model's best params
        """
        X, y = get_X_y(train, model.get_pipe())
        groups = self.get_uuid_groups(train)
        cv = self.get_grid_search_cv()
        best_estimator = GridSearchCV(estimator=model,
                                      param_grid=self.param_grid,
                                      cv=cv,
                                      refit=False,
                                      scoring=self.scoring_function,
                                      n_jobs=-1,
                                      verbose=10
                                      )
        best_estimator.fit(X, y, groups=groups)
        self.save_results(best_estimator, model_name)
        return best_estimator.best_params_

    def save_results(self, best_estimator, model_name):
        attributes = model_name.split('_')[:-1]
        attributes.append(f"hyper_parameters_results_{model_name}.csv")
        path = f"{os.path.sep}".join(attributes)
        results_path = os.path.join(self.directories['results'], path)
        if not os.path.exists(os.path.dirname(results_path)):
            os.makedirs(os.path.dirname(results_path))
        pd.DataFrame(best_estimator.cv_results_).to_csv(results_path)

    def get_grid_search_cv(self):
        cv = None
        if self.grid_search_groups_method in 'GroupKFold':
            cv = GroupKFold(n_splits=self.cross_validation_folds_number)
        elif self.grid_search_groups_method in 'StratifiedKFold':
            cv = StratifiedKFold(n_splits=3, shuffle=True)
        return cv

    def get_uuid_groups(self, train):
        """
        Map every uuid to unique numeric group number
        :param train: data frame
        :return: np array where every index is the group number of the index in the train data frame
        """
        if self.is_all_set:
            mapping = self.create_uuid_mapping()
            groups = pd.Series(train.index.map(mapping))
        else:
            groups = pd.Series(train.index).astype('category').cat.codes
        return groups

    def create_uuid_mapping(self):
        mapping = {}
        for group_number, uuid_list in self.config['groups']['groups_dict'].items():
            for uuid in uuid_list:
                mapping[uuid] = group_number
        return mapping

    def get_param_grid(self, params_list):
        grid_params_list = []
        params = {}
        possible_params = params_list.values()
        permutations = list(itertools.product(*possible_params))
        for permutation in permutations:
            for i, key in enumerate(list(params_list.keys())):
                params[key] = permutation[i]
            grid_params_list.append(params)
            params = {}
        return {"model_params": grid_params_list}

    def set_estimator(self, estimator):
        self.param_grid = self.get_param_grid(self.config[estimator]['param_grid'])
        self.cross_validation_folds_number = self.config[estimator]['cross_validation_folds_number']
        self.scoring_function = self.config[estimator]['scoring_function']




