import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def get_single_pre_pipe(cat_colds_indices, num_cols_indices):
    column_transformer_scalar = ColumnTransformer(
        [
            ('ohe', OneHotEncoder(sparse=False), cat_colds_indices),
            ('scalar', StandardScaler(), num_cols_indices)
        ]
    )
    column_transformer_null_handler = ColumnTransformer(
        [
            ('SimpleImputerCat', SimpleImputer(strategy="most_frequent"), cat_colds_indices),
            ('SimpleImputerNum', SimpleImputer(strategy='mean'), num_cols_indices)
        ]
    )
    pipe = Pipeline(
        [
            ('column_transformer_scalar', column_transformer_scalar),
            ('column_transformer_null_handler', column_transformer_null_handler)

        ], verbose=True
    )

    return pipe


def split_cat_num_cols(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["category"]).columns
    cat_colds_indices = [X.columns.get_loc(c) for c in cat_cols if c in X]
    num_cols = X.select_dtypes(exclude=["category"]).columns
    num_cols_indices = [X.columns.get_loc(c) for c in num_cols if c in X]

    return cat_colds_indices, num_cols_indices
