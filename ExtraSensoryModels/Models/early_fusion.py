from ExtraSensoryModels.Interfaces.ExtraSensoryAbstractModel import ExtraSensoryAbstractModel
from sklearn.base import BaseEstimator


class EarlyFusion(ExtraSensoryAbstractModel):
    def __init__(self, estimator: BaseEstimator, model_params: dict = None):
        """

        :param estimator: sklearn estimator constructor
        :param model_params: sklearn estimator params
        """
        self.estimator = estimator
        self.model_params = model_params

    def fit(self, X, y):
        self.model_ = self.estimator.set_params(**self.model_params)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


