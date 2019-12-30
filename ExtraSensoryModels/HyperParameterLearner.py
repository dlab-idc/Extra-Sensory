import logging
import pandas as pd

from utils.GeneralUtils import ConfigManager
from utils.TransformerUtils import get_X_y
from sklearn.model_selection import GridSearchCV, GroupKFold


class HyperParameterLearner:
    def __init__(self):
        self.config = ConfigManager.get_config('hyper_parameters_learner')
        self.logger = logging.getLogger('classifier')
        self.param_grid = self.config['logistic_regression']['param_grid']
        self.cross_validation_folds_number = self.config['logistic_regression']['cross_validation_folds_number']
        self.scoring_function = self.config['logistic_regression']['scoring_function']

    def async_grid_search(self, train, estimator):
        """
        This function finds the best params for an estimator using greed search.
        Cross validation process is done by group K folds by uuid of the person.
        :param train: array like
         and the value is a list of param optional values
        :param estimator: an sklearn estimator
        :return: dictionary, train model's best params
        """
        X, y = get_X_y(train)
        groups = self.get_uuid_groups(train)
        k_folds_groups = GroupKFold(n_splits=self.cross_validation_folds_number)
        best_estimator = GridSearchCV(estimator=estimator,
                                      param_grid=self.param_grid,
                                      cv=k_folds_groups,
                                      refit=False,
                                      scoring=self.scoring_function,
                                      n_jobs=-1
                                      )
        best_estimator.fit(X, y, groups=groups)
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
