from pathlib import Path

from titanic_analysis.domain.dataset.dataset import Dataset
from titanic_analysis.infrastructure.io.analysis.config_loader import load_config
from titanic_analysis.infrastructure.logic.analysis.analyzer import DatasetAnalyzer
from titanic_analysis.infrastructure.logic.analysis.utils import set_display_options


def prepare_display(config_path: Path) -> None:
    """表示の準備をする

    Args:
        config_path (Path): configファイルの文字列パス
    """
    config_dto = load_config(config_path)

    set_display_options(
        max_rows=config_dto.max_rows,
        max_columns=config_dto.max_columns,
    )


def describe_dataset(dataset: Dataset) -> None:
    """データセットの概要を表示する

    Args:
        dataset (Dataset): データセットのインスタンス
    """
    DatasetAnalyzer.display_summary(dataset)
    DatasetAnalyzer.display_categorized_columns(dataset.x)
