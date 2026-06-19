"""分析用DTO定義モジュール"""

from pathlib import Path

import yaml

from titanic_analysis.infrastructure.io.analysis.dto import AnalysisDTO
from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    PytorchConfigDTO,
)

__all__ = [
    "load_analysis_config",
    "load_lightgbm_config",
    "load_pytorch_config",
    "load_xgboost_config",
]


def load_analysis_config(config_path: Path) -> AnalysisDTO:
    """configファイル(*.yaml)を読み込む

    Args:
        config_path (Path): configファイルのパス

    Returns:
        AnalysisDTO: configファイルから読み込んだ情報のDTO
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)

    return AnalysisDTO(**config["option"]["display"])


def load_xgboost_config(config_path: Path) -> dict:
    """Load config file for training using xgboost.

    Args:
        config_path (Path): Config file path

    Returns:
        dict: Config data
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)

    return dict(**config["model"])


def load_lightgbm_config(config_path: Path) -> dict:
    """Load config file for training using lightgbm.

    Args:
        config_path (Path): Config file path

    Returns:
        dict: Config data
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)

    return dict(**config["model"])


def load_pytorch_config(config_path: Path) -> PytorchConfigDTO:
    """Load config file for training using pytorch

    Args:
        config_path (Path): Config file path

    Returns:
        TrainingPipelineDTO: DTO for config file
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)

    return PytorchConfigDTO(**config["model"])
