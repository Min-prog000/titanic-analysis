"""Preprocess use case"""

from logging import Logger

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from titanic_analysis.application.constants import (
    ADDITIONAL_ENCODING_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SELECTED_FEATURES,
)

__all__ = ["preprocess_load_data"]


def preprocess_load_data(
    logger: Logger,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess data to use for train model

    Args:
        logger (Logger): Logger
        train_data (pd.DataFrame): Train data
        test_data (pd.DataFrame): Test data

    Returns:
        tuple[np.ndarray, np.ndarray]: Preprocessed train and test data
    """
    train_data_cleaned = clean_data(logger, train_data, SELECTED_FEATURES)
    test_data_cleaned = clean_data(logger, test_data, SELECTED_FEATURES)

    preprocessor = generate_preprocessor(
        MinMaxScaler(),
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor)])

    train_data_preprocessed = pipeline.fit_transform(
        train_data_cleaned,
    )
    test_data_preprocessed = pipeline.transform(test_data_cleaned)

    logger.debug("Train data shape: %s", train_data_preprocessed.shape)
    logger.debug("Test data shape: %s", test_data_preprocessed.shape)

    logger.debug("\nColumn names: %s", preprocessor.get_feature_names_out())

    return train_data_preprocessed, test_data_preprocessed


def clean_data(
    logger: Logger,
    data_loaded: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    # Choose column
    data_filtered = data_loaded.loc[:, selected_features]
    logger.debug(data_filtered.columns)

    # Fill numeric column with mean
    mean_series = data_filtered.mean(numeric_only=True)
    mean_round = round(mean_series)
    data_cleaned = data_filtered.fillna(mean_round)

    # Fill "Embarked" column with mode
    # "S" is the most numerous category
    dataframe_groupby_embarked = data_cleaned.groupby(
        ADDITIONAL_ENCODING_COLUMN,
        dropna=False,
    )
    size_groupby_embarked = dataframe_groupby_embarked.size()
    mode_embarked_index = size_groupby_embarked.idxmax()
    data_cleaned[ADDITIONAL_ENCODING_COLUMN] = data_cleaned[
        ADDITIONAL_ENCODING_COLUMN
    ].fillna(
        mode_embarked_index,
    )

    return data_cleaned


def generate_preprocessor(
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_encoder: OneHotEncoder | None = None,
) -> ColumnTransformer:
    """Generate transformer for one-hot encoding

    Args:
        scaler (StandardScaler | MinMaxScaler | RobustScaler): Scaler
        numeric_features (list[str]): Numeric features of dataset
        categorical_features (list[str]): Categorical features of dataset
        categorical_encoder (OneHotEncoder | None, optional):
            Encoder of categorical features. Defaults to None.

    Returns:
        ColumnTransformer: Transformer including scaler and encoder for categorical data
    """
    if categorical_encoder is None:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")

    transformers = []
    if numeric_features:
        transformers.append(("numeric", scaler, numeric_features))

    if categorical_features:
        transformers.append(("categorical", categorical_encoder, categorical_features))

    return ColumnTransformer(transformers=transformers)
