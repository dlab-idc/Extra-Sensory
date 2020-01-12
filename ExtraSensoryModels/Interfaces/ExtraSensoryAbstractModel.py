from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin


class ExtraSensoryAbstractModel(ABC, BaseEstimator, ClassifierMixin):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_pipe(self):
        pass
