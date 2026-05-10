"""分析用DTO定義モジュール"""

from pathlib import Path

import yaml

from titanic_analysis.infrastructure.io.analysis.dto import AnalysisDTO
from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    GradientBoostingClassifierConfigDTO,
    LogisticRegressionConfigDTO,
    PytorchConfigDTO,
)


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


def load_logistic_regression_config(
    config_path: Path,
) -> LogisticRegressionConfigDTO:
    """Load config file for training using logistic regression

    Args:
        config_path (Path): Config file path

    Returns:
        LogisticRegressionConfigDTO: DTO for config file
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)

    return LogisticRegressionConfigDTO(**config["model"])


def load_gradient_boosting_classifier_config(
    config_path: Path,
) -> GradientBoostingClassifierConfigDTO:
    """Load config file for training using gradient boosting classifier

    Args:
        config_path (Path): Config file path

    Returns:
        GradientBoostingClassifierConfigDTO: DTO for config file
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)

    return GradientBoostingClassifierConfigDTO(**config["model"])


def load_xgboost_config(config_path: Path) -> dict:
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
