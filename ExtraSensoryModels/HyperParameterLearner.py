import logging
import pandas as pd
import itertools

from utils.GeneralUtils import ConfigManager
from utils.TransformerUtils import get_X_y
from sklearn.model_selection import GridSearchCV, GroupKFold
from ExtraSensoryModels.Interfaces.ExtraSensoryAbstractModel import ExtraSensoryAbstractModel


class HyperParameterLearner:
    def __init__(self):
        self.config = ConfigManager.get_config('hyper_parameters_learner')
        self.logger = logging.getLogger('classifier')
        self.param_grid = None
        self.cross_validation_folds_number = None
        self.scoring_function = None

    def async_grid_search(self, train, model : ExtraSensoryAbstractModel):
        """
        This function finds the best params for an estimator using greed search.
        Cross validation process is done by group K folds by uuid of the person.
        :param train: array like
        :param model: an Extra sensory model
        :return: dictionary, train model's best params
        """
        X, y = get_X_y(train, model.get_pipe())
        groups = self.get_uuid_groups(train)
        k_folds_groups = GroupKFold(n_splits=self.cross_validation_folds_number)
        best_estimator = GridSearchCV(estimator=model,
                                      param_grid=self.param_grid,
                                      cv=k_folds_groups,
                                      refit=False,
                                      scoring=self.scoring_function,
                                      n_jobs=-1,
                                      verbose=10
                                      )
        best_estimator.fit(X, y, groups=groups)
        self.logger.info(f"CV results {best_estimator.cv_results_}")
        self.logger.info(f"Best params are {best_estimator.best_params_}")
        return best_estimator.best_params_

    @staticmethod
    def get_uuid_groups(train):
        """
        Map every uuid to unique numeric group number
        :param train: data frame
        :return: np array where every index is the group number of the index in the train data frame
        """
        groups = pd.Series(train.index).astype('category').cat.codes
        return groups

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




