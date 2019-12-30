import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from typing import Dict


class GaussianAndBernoulliNB(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 cat_cols_indices, num_cols_indices,
                 gaussian_params: Dict=None,
                 bernoulli_params: Dict=None):
        self.gaussian_params = gaussian_params
        self.bernoulli_params = bernoulli_params
        self.cat_cols_indices = cat_cols_indices
        self.num_cols_indices = num_cols_indices

        if gaussian_params is not None:
            self.gaussian_model = GaussianNB(**self.gaussian_params)
        else:
            self.gaussian_model = GaussianNB()

        if bernoulli_params is not None:
            self.bernoulli_model = BernoulliNB(**self.bernoulli_params)
        else:
            self.bernoulli_model = BernoulliNB()

    def __split_to_gaussian_bernoulli(self, X):
        X_bernoulli = X[:, self.cat_cols_indices]
        X_gaussian = X[:, self.num_cols_indices]

        return X_bernoulli, X_gaussian

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        X_bernoulli, X_gaussian = self.__split_to_gaussian_bernoulli(X)
        self.bernoulli_model_ = self.bernoulli_model.fit(X_bernoulli, y)
        self.gaussian_model_ = self.gaussian_model.fit(X_gaussian, y)

        self.class_log_prior_ = self.bernoulli_model.class_log_prior_


        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        #         check_is_fitted(self,'deprecated')

        # Input validation
        X = check_array(X)
        X_bernoulli, X_gaussian = self.__split_to_gaussian_bernoulli(X)

        bernoulli_log_proba = self.bernoulli_model_.predict_log_proba(X_bernoulli)
        gaussian_log_proba = self.gaussian_model_.predict_log_proba(X_gaussian)
        self.model_log_proba_ = bernoulli_log_proba + gaussian_log_proba - self.class_log_prior_

        return np.argmax(self.model_log_proba_, axis=1)





