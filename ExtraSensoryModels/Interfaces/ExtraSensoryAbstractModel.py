from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin


class ExtraSensoryAbstractModel(ABC, BaseEstimator, ClassifierMixin):
    pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
