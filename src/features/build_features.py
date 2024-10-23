import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import joblib
import os


def build_preprocessor() -> ColumnTransformer:
    """Build a preprocessor for the dataset

    Returns:
        ColumnTransformer: preprocessor for the dataset
    """
    print("Building preprocessor...")

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


def save_preprocessor(
    preprocessor: ColumnTransformer,
    data: pd.DataFrame,
    path: str = "/Users/raydi/Desktop/Code/portfolio/end-to-end-mleng/models",
) -> None:
    """Fit the preprocessor using the provided data and save it to disk

    Args:
        preprocessor (ColumnTransformer): preprocessor to fit and save
        data (pd.DataFrame): data to fit the preprocessor
        path (str, optional): path to save the preprocessor. Defaults to "/Users/raydi/Desktop/Code/portfolio/end-to-end-mleng/models".
    """
    print(f"Fitting preprocessor to data...")
    preprocessor.fit(data)
    print(f"Preprocessor fitted!")

    print(f"Saving preprocessor to {path}...")
    os.makedirs(path, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(path, "preprocessor.pkl"))
    print(f"Preprocessor saved at {path}")


def load_preprocessor(path: str = "/Users/raydi/Desktop/Code/portfolio/end-to-end-mleng/models") -> ColumnTransformer:
    """Load the preprocessor from disk

    Args:
        path (str, optional): path to load the preprocessor. Defaults to "/Users/raydi/Desktop/Code/portfolio/end-to-end-mleng/models".

    Returns:
        ColumnTransformer: preprocessor loaded from disk
    """
    print(f"Loading preprocessor from {path}...")
    preprocessor = joblib.load(os.path.join(path, "preprocessor.pkl"))
    print(f"Preprocessor loaded from {path}")
    return preprocessor


def preprocess_data(
    data: pd.DataFrame, preprocessor: ColumnTransformer
) -> pd.DataFrame:
    """Preprocess the data using the preprocessor

    Args:
        data (pd.DataFrame): data to preprocess
        preprocessor (ColumnTransformer): preprocessor to use

    Returns:
        pd.DataFrame: preprocessed data
    """
    print("Preprocessing data...")
    preprocessed_data = preprocessor.transform(data)
    print("Data preprocessed!")
    return preprocessed_data


def main():
    print("Reading data from 'data/raw/train.csv'...")
    data = pd.read_csv("/Users/raydi/Desktop/Code/portfolio/end-to-end-mleng/data/raw/train.csv")
    print("Data read successfully!")

    preprocessor = build_preprocessor()
    save_preprocessor(preprocessor, data)

    transform_processor = load_preprocessor()
    preprocessed_data = preprocess_data(data, transform_processor)

    preprocessed_data = pd.DataFrame(preprocessed_data)
    print("Saving preprocessed data to 'data/interim/preprocessed_data.csv'...")
    preprocessed_data.to_csv("data/interim/preprocessed_data.csv", index=False)
    print("Preprocessed data saved!")


if __name__ == "__main__":
    main()
