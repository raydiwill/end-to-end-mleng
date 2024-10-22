import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import joblib
import os


def build_preprocessor() -> ColumnTransformer:
    """ Build a preprocessor for the dataset

    Returns:
        ColumnTransformer: preprocessor for the dataset
    """

    binary_cols = ["cb_person_default_on_file"]
    ordinal_cols = ["loan_grade"]
    ohe_col = ["person_home_ownership", "loan_intent"]
    null_cols = ["loan_int_rate", "person_emp_length"]

    binary_transformer = Pipeline([("binary", OrdinalEncoder())])

    ordinal_transformer = Pipeline(
        [("ordinal", OrdinalEncoder(categories=[["A", "B", "C", "D", "E", "F", "G"]]))]
    )

    onehot_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    null_transformer = Pipeline([("impute", SimpleImputer(strategy="mean"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("null", null_transformer, null_cols),
            ("binary", binary_transformer, binary_cols),
            ("ordinal", ordinal_transformer, ordinal_cols),
            ("onehot", onehot_transformer, ohe_col),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return preprocessor
