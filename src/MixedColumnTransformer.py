from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from enum import Enum


class ColumnTypeEnum(Enum):
    CATEGORY = 'category'
    NUMERIC = 'numeric'


class MixedColumnTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformers):
        self.transformers = transformers
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        self.categorial_columns_, self.numeric_columns_ =\
            self.split_cat_num_cols(X)

        X_category = X[self.categorial_columns_]
        X_numeric = X[self.numeric_columns_]

        for transformer_tuple in self.transformers:
            _, transformer, column_dtype = transformer_tuple

            assert isinstance(transformer, TransformerMixin),\
                f"{transformer} is not an instance of TransformerMixin"
            assert isinstance(column_dtype, ColumnTypeEnum),\
                f"{column_dtype} is not an instance of ColumnTypeEnum"

            if column_dtype == ColumnTypeEnum.CATEGORY:
                X_category = transformer.fit_transform(X_category)
            elif column_dtype == ColumnTypeEnum.NUMERIC:
                X_numeric = transformer.fit_transform(X_numeric)
            else:
                raise ValueError(f"{column_dtype} is not a supported column data type")

        self.cat_cols_indices_ = np.arange(0, X_category.shape[1])
        self.num_cols_indices_ = np.arange(X_category.shape[1], X_numeric.shape[1])
        self.is_fitted = True

        return self

    def transform(self, X: pd.DataFrame):
        if not self.is_fitted:
            raise Exception("you must call fit first!")

        X_category = X[self.categorial_columns_]
        X_numeric = X[self.numeric_columns_]

        for transformer_tuple in self.transformers:
            _, transformer, column_dtype = transformer_tuple

            assert isinstance(transformer, TransformerMixin),\
                f"{transformer} is not an instance of TransformerMixin"
            assert isinstance(column_dtype, ColumnTypeEnum),\
                f"{column_dtype} is not an instance of ColumnTypeEnum"

            if column_dtype == ColumnTypeEnum.CATEGORY:
                X_category = transformer.transform(X_category)
            elif column_dtype == ColumnTypeEnum.NUMERIC:
                X_numeric = transformer.transform(X_numeric)
            else:
                raise ValueError(f"{column_dtype} is not a supported column data type")

        X_transformed = np.concatenate((X_category, X_numeric), axis=1)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        self.fit(X)
        X_transformed = self.transform(X)

        return X_transformed

    @staticmethod
    def split_cat_num_cols(X: pd.DataFrame):
        cat_cols = X.select_dtypes(include=["category"]).columns
        num_cols = X.select_dtypes(exclude=["category"]).columns

        return cat_cols, num_cols
