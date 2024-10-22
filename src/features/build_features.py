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

    print("Preprocessor built!")
    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, path: str = "models/") -> None:
    """ Save the preprocessor to disk

    Args:
        preprocessor (ColumnTransformer): preprocessor to save
        path (str, optional): path to save the preprocessor. Defaults to "models/".
    """

    os.makedirs(path, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(path, "preprocessor.pkl"))
    print(f"Preprocessor saved at {path}")


def load_preprocessor(path: str = "models/") -> ColumnTransformer:
    """ Load the preprocessor from disk

    Args:
        path (str, optional): path to load the preprocessor. Defaults to "models/".

    Returns:
        ColumnTransformer: preprocessor loaded from disk
    """

    preprocessor = joblib.load(os.path.join(path, "preprocessor.pkl"))
    print(f"Preprocessor loaded from {path}")
    return preprocessor


def preprocess_data(data: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """ Preprocess the data using the preprocessor

    Args:
        data (pd.DataFrame): data to preprocess
        preprocessor (ColumnTransformer): preprocessor to use

    Returns:
        pd.DataFrame: preprocessed data
    """

    preprocessed_data = preprocessor.fit_transform(data)
    print("Data preprocessed!")
    return preprocessed_data
