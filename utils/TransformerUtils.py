from tempfile import mkdtemp

import pandas as pd
import numpy as np

from enum import Enum

from sklearn.feature_selection import SelectPercentile, mutual_info_regression, mutual_info_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer


class ColumnTypeEnum(Enum):
    CATEGORY = 'category'
    NUMERIC = 'numeric'


class MixedColumnTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformers):
        self.transformers = transformers
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        self.categorial_columns_, self.numeric_columns_ = split_cat_num_cols(X)

        X_category = X[self.categorial_columns_]
        X_numeric = X[self.numeric_columns_]

        for transformer_tuple in self.transformers:
            _, transformer, column_dtype = transformer_tuple

            self.validate_input(column_dtype, transformer)

            if column_dtype == ColumnTypeEnum.CATEGORY:
                X_category = transformer.fit_transform(X_category, y)
            elif column_dtype == ColumnTypeEnum.NUMERIC:
                X_numeric = transformer.fit_transform(X_numeric, y)
            else:
                raise ValueError(f"{column_dtype} is not a supported column data type")

        self.set_transformed_data_indices(X_category, X_numeric)
        self.is_fitted = True

        return self

    def set_transformed_data_indices(self, X_category, X_numeric):
        """
        Set the indeces of the categorical and numeric data
        :param X_category:
        :param X_numeric:
        :return:
        """
        self.cat_cols_indices_ = np.arange(0, X_category.shape[1])
        self.num_cols_indices_ = np.arange(X_category.shape[1], X_numeric.shape[1])

    def validate_input(self, column_dtype, transformer):
        """
        validate the transformer is an sklearn transformer and the column d-type is ColumnTypeEnum
        """
        assert isinstance(transformer, TransformerMixin), f"{transformer} is not an instance of TransformerMixin"
        assert isinstance(column_dtype, ColumnTypeEnum), f"{column_dtype} is not an instance of ColumnTypeEnum"

    def transform(self, X: pd.DataFrame):
        """
        Transform the given data
        """
        if not self.is_fitted:
            raise Exception("you must call fit first!")

        X_category = X[self.categorial_columns_]
        X_numeric = X[self.numeric_columns_]

        for transformer_tuple in self.transformers:
            _, transformer, column_dtype = transformer_tuple

            self.validate_input(column_dtype, transformer)

            if column_dtype == ColumnTypeEnum.CATEGORY:
                X_category = transformer.transform(X_category)
            elif column_dtype == ColumnTypeEnum.NUMERIC:
                X_numeric = transformer.transform(X_numeric)
            else:
                raise ValueError(f"{column_dtype} is not a supported column data type")

        X_transformed = np.concatenate((X_category, X_numeric), axis=1)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        """
        Main function, fit the transformers
        :param X: data-frame
        :param y: None (require by sklearn api)
        :param fit_params: (require by sklearn api)
        :return: numpy array, the transformed data according to the transformer
        """
        self.fit(X, y)
        X_transformed = self.transform(X)

        return X_transformed


def split_cat_num_cols(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["category"]).columns
    num_cols = X.select_dtypes(exclude=["category"]).columns

    return cat_cols, num_cols


def get_single_pre_pipe():
    transormers = [
        ('ohe', OneHotEncoder(sparse=False), ColumnTypeEnum.CATEGORY),
        ('scaler', StandardScaler(), ColumnTypeEnum.NUMERIC),
        ('SimpleImputerCat', SimpleImputer(strategy="most_frequent"), ColumnTypeEnum.CATEGORY),
        ('SimpleImputerNum', SimpleImputer(strategy='mean'), ColumnTypeEnum.NUMERIC)
    ]

    pipe = Pipeline([
        ("mixed_column_transformer", MixedColumnTransformer(transormers)),
        ("variance_threshold", VarianceThreshold(threshold=0.2))
    ], verbose=True, memory=mkdtemp())

    return pipe


def get_X_y(data, pipe, is_fitted=False):
    y = np.array(data['label']).astype('uint8')
    data.drop(['label'], axis=1, inplace=True)
    X = pipe.transform(data) if is_fitted else pipe.fit_transform(data, y)
    return X, y
