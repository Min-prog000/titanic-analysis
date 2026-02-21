"""分析用DTO定義モジュール"""

from pathlib import Path

import yaml

from titanic_analysis.infrastructure.io.analysis.dto import AnalysisDTO
from titanic_analysis.infrastructure.io.training_pipeline.dto import TrainingPipelineDTO


def load_analysis_config(config_path: Path) -> AnalysisDTO:
    """configファイル（*.yaml）を読み込む

    Args:
        config_path (Path): configファイルのパス

    Returns:
        AnalysisDTO: configファイルから読み込んだ情報のDTO
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)

    return AnalysisDTO(**config["option"]["display"])


def load_training_config(config_path: Path) -> TrainingPipelineDTO:
    with config_path.open() as file:
        config = yaml.safe_load(file)

    return TrainingPipelineDTO(**config["model"])
