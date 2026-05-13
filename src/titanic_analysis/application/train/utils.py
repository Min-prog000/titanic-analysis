"""Utility module for model training."""

from pathlib import Path

from titanic_analysis.application.constants import XGBOOST_TREE_PATH
from titanic_analysis.infrastructure.io.constants import (
    CONFIG_FILE_EXTENSION,
    CONFIG_FILE_PREFIX_XGBOOST,
    CONFIG_FOLDER_PREFIX,
    SAVE_MODEL_FILE_EXTENSION_XGBOOST,
    SAVE_MODEL_FILE_PARENT_XGBOOST,
    SAVE_MODEL_FILE_PREFIX_XGBOOST,
    SAVE_MODEL_ROOT_XGBOOST,
    SAVE_TREE_FILE_INDEX_PREFIX_XGBOOST,
    SAVE_TREE_FILE_PREFIX_XGBOOST,
    XGBOOST,
)
from titanic_analysis.infrastructure.logic.build.utils import load_case_id

__all__ = [
    "generate_config_path",
    "generate_model_save_path",
    "generate_next_case_id",
    "generate_tree_save_path",
    "get_case_id",
]


# =======
# Utility
# =======
def generate_output_path(folder_path: Path, file_path: Path) -> Path:
    return folder_path.joinpath(file_path)


# ============
# Save case id
# ============
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
def generate_tree_save_path(case_id: int, save_tree_index: int) -> tuple[Path, Path]:
    """Generate file path saving tree data.

    Args:
        case_id (int): case id.
        save_tree_index (int): tree number saving tree data.

    Returns:
        tuple[Path, Path]: folder path and file path.
    """
    folder_path = get_tree_folder_path()
    file_name_path = get_tree_file_path(case_id, save_tree_index)

    file_path = generate_output_path(folder_path, file_name_path)

    return folder_path, file_path


# Folder path
def get_tree_folder_path(xgboost_tree_path: str = XGBOOST_TREE_PATH) -> Path:
    return Path(xgboost_tree_path)


# File path
def get_tree_file_path(case_id: int, save_tree_index: int) -> Path:
    return Path(generate_tree_file_name(case_id, save_tree_index))


def generate_tree_file_name(case_id: int, save_tree_index: int) -> str:
    file_prefix = SAVE_TREE_FILE_PREFIX_XGBOOST
    index_prefix = SAVE_TREE_FILE_INDEX_PREFIX_XGBOOST

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
