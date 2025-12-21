"""分析用DTO定義モジュール"""

from pathlib import Path

import yaml

from titanic_analysis.framework.log.generator import generate_logger
from titanic_analysis.infrastructure.io.analysis.dto import AnalysisDTO


def load_config(config_path: Path) -> AnalysisDTO:
    """configファイル（*.yaml）を読み込む

    Args:
        config_path (Path): configファイルのパス

    Returns:
        AnalysisDTO: configファイルから読み込んだ情報のDTO
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)

    logger = generate_logger(__name__)
    logger.debug(config)

    return AnalysisDTO(**config["option"]["display"])
