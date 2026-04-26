"""分析用DTO定義モジュール"""

from pathlib import Path

import yaml

from titanic_analysis.infrastructure.io.training_pipeline.dto import PytorchConfigDTO


def load_config(config_path: Path) -> PytorchConfigDTO:
    """configファイル（*.yaml）を読み込む

    Args:
        config_path (Path): configファイルのパス

    Returns:
        AnalysisDTO: configファイルから読み込んだ情報のDTO
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)

    return PytorchConfigDTO(**config["preprocess"])
