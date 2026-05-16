"""Utility module for model training."""

import sys
from logging import Logger
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from titanic_analysis.application.constants import (
    CASE_ID_PATH,
    CONCAT_WITH_COLUMN,
    ID_COLUMN,
    TARGET_COLUMN,
)
from titanic_analysis.application.preprocess import preprocess_load_data
from titanic_analysis.infrastructure.io.constants import (
    CONFIG_FILE_EXTENSION,
    CONFIG_FILE_PREFIX_XGBOOST,
    CONFIG_FOLDER_PREFIX,
    SAVE_MODEL_FILE_EXTENSION_XGBOOST,
    SAVE_MODEL_FILE_PARENT_XGBOOST,
    SAVE_MODEL_FILE_PREFIX_XGBOOST,
    SAVE_MODEL_ROOT_XGBOOST,
    SAVE_TREE_FILE_INDEX_PREFIX,
    SAVE_TREE_FILE_PREFIX,
    XGBOOST,
)
from titanic_analysis.infrastructure.logic.build.utils import load_case_id

__all__ = [
    "generate_config_path",
    "generate_model_save_path",
    "generate_next_case_id",
    "generate_tree_save_path",
    "get_case_id",
    "save_case_id",
]


# =======
# Utility
# =======
def create_dataset(
    logger: Logger,
    train_dataset_path: str,
    test_dataset_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    # Data file loading
    df_train, df_test = load_data_from_csv(train_dataset_path, test_dataset_path)

    # Log dataframe head
    log_df_head(logger, df_train)
    log_df_head(logger, df_test)

    # Preprocess
    # Training data
    x_train, x_test = preprocess_load_data(
        logger,
        df_train,
        df_test,
    )

    # Log dataset head
    log_array_head(logger, x_train)
    log_array_head(logger, x_test)

    # Training label
    y_train = extract_target_column(df_train)

    # Submission file column
    passenger_ids = extract_id_column(df_test, ID_COLUMN)

    # データセットのサイズが等しいことの確認
    # TODO: Revise to be able to compare preprocessed data columns
    validate_data_shapes(logger, x_train, x_test)

    return x_train, y_train, x_test, passenger_ids


def generate_output_path(folder_path: Path, file_path: Path) -> Path:
    return folder_path.joinpath(file_path)


def generate_submission_dataframe(
    passenger_ids: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    y_pred_df = pd.DataFrame(y_pred, columns=[TARGET_COLUMN])
    return pd.concat([passenger_ids, y_pred_df], axis=CONCAT_WITH_COLUMN)


def load_data_from_csv(
    train_dataset_path: str,
    test_dataset_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)

    return df_train, df_test


def extract_target_column(df_train: pd.DataFrame) -> np.ndarray:
    target_column_series = df_train.loc[:, TARGET_COLUMN]

    return series_to_array(target_column_series)


def extract_id_column(df_test: pd.DataFrame, id_column_name: str) -> pd.Series:
    return df_test[id_column_name]


def validate_data_shapes(
    logger: Logger,
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> None:
    if not (x_train.shape and x_test.shape):
        logger.error("Not match datasets shape.")
        sys.exit()


def series_to_array(target_column: pd.Series) -> np.ndarray:
    return np.array(target_column)


def log_array_head(logger: Logger, array: np.ndarray, row_index: int = 5) -> None:
    logger.info("\n%s", array[:row_index, :])


def log_df_head(logger: Logger, dataframe: pd.DataFrame) -> None:
    logger.info("\n%s", dataframe.head())


def get_pipeline_model(
    pipeline: Pipeline,
    model_key_prefix: str,
) -> xgb.XGBClassifier:
    return pipeline.named_steps[model_key_prefix]


# ============
# Save case id
# ============


def save_case_id(case_id: int, file_name_path: str = CASE_ID_PATH) -> None:
    """Save case id.

    Args:
        case_id (int): Current case id.
        file_name_path (str, optional):
            File path saving case id. Defaults to CASE_ID_PATH.
    """
    joblib.dump(generate_next_case_id(case_id), Path(file_name_path))


def generate_next_case_id(case_id: int) -> int:
    """Generate next case id.

    Args:
        case_id (int): Current case id.

    Returns:
        int: Next case id.
    """
    return case_id + 1


def get_case_id(case_id_path_str: str) -> int:
    """Get current case id.

    Args:
        case_id_path_str (str): File path saved current case id.

    Returns:
        int: Current case id.
    """
    # Get case id
    case_id_path = Path(case_id_path_str)

    return load_case_id(case_id_path)


# ===============
# Save tree graph
# ===============
def generate_tree_save_path(
    tree_folder_path_str: str,
    case_id: int,
    save_tree_index: int,
) -> tuple[Path, Path]:
    """Generate file path saving tree data.

    Args:
        tree_folder_path_str (str): tree folder path as `str`.
        case_id (int): case id.
        save_tree_index (int): tree number saving tree data.

    Returns:
        tuple[Path, Path]: folder path and file path.
    """
    folder_path = get_tree_folder_path(tree_folder_path_str)
    file_name_path = get_tree_file_path(case_id, save_tree_index)

    file_path = generate_output_path(folder_path, file_name_path)

    return folder_path, file_path


# Folder path
def get_tree_folder_path(tree_folder_path: str) -> Path:
    return Path(tree_folder_path)


# File path
def get_tree_file_path(case_id: int, save_tree_index: int) -> Path:
    return Path(generate_tree_file_name(case_id, save_tree_index))


def generate_tree_file_name(case_id: int, save_tree_index: int) -> str:
    file_prefix = SAVE_TREE_FILE_PREFIX
    index_prefix = SAVE_TREE_FILE_INDEX_PREFIX

    return f"{file_prefix}{case_id}{index_prefix}{save_tree_index}"


# ================
# Save config file
# ================
def generate_config_path(case_id: int) -> tuple[Path, Path]:
    """Generate path output config file.

    Args:
        case_id (int): case id.

    Returns:
        tuple[Path, Path]: output folder path and file path.
    """
    folder_path = get_config_folder_path(case_id)
    file_name_path = get_config_file_name_path(case_id)

    file_path = generate_output_path(folder_path, file_name_path)

    return folder_path, file_path


# Folder path
def get_config_folder_path(case_id: int) -> Path:
    return Path(generate_config_folder_name(case_id))


def generate_config_folder_name(
    case_id: int,
    prefix: str = CONFIG_FOLDER_PREFIX,
) -> str:
    return f"{prefix}{case_id}"


# File path
def get_config_file_name_path(case_id: int) -> Path:
    return Path(generate_config_file_name(case_id))


def generate_config_file_name(
    case_id: int,
    prefix: str = CONFIG_FILE_PREFIX_XGBOOST,
    extension: str = CONFIG_FILE_EXTENSION,
) -> str:
    return f"{prefix}{case_id}{extension}"


# ==========
# Save model
# ==========
def generate_model_save_path(case_id: int) -> tuple[Path, Path]:
    """Generate path output model definition.

    Args:
        case_id (int): case id.

    Returns:
        tuple[Path, Path]: output folder path and file path.
    """
    save_folder_path = get_model_folder_path(case_id)
    save_file_name_path = get_model_file_name_path(case_id)

    save_file_path = generate_output_path(save_folder_path, save_file_name_path)

    return save_folder_path, save_file_path


# Folder path
def get_model_folder_path(case_id: int) -> Path:
    return Path(generate_model_folder_name(case_id))


def generate_model_folder_name(
    case_id: int,
    root: str = SAVE_MODEL_ROOT_XGBOOST,
    file_parent: str = SAVE_MODEL_FILE_PARENT_XGBOOST,
) -> str:
    return f"{root}{XGBOOST}{file_parent}{case_id}"


# File path
def get_model_file_name_path(case_id: int) -> Path:
    return Path(generate_model_file_name(case_id))


def generate_model_file_name(
    case_id: int,
    prefix: str = SAVE_MODEL_FILE_PREFIX_XGBOOST,
    extension: str = SAVE_MODEL_FILE_EXTENSION_XGBOOST,
) -> str:
    return f"{prefix}{case_id}{extension}"
