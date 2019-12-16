from ExtraSensoryModels.Interfaces.ExtraSensoryAbstractModel import ExtraSensoryAbstractModel


class EarlyFusion(ExtraSensoryAbstractModel):
    def __init__(self, constructor, model_params: dict):
        """

        :param constructor:
        :param model_params:
        """
        self.constrictor = constructor
        self.model_params = model_params

    def fit(self, X, y):
        self.model_ = self.constrictor(**self.model_params)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


