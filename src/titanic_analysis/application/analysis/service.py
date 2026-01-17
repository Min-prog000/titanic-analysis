"""データセットの分析を行うモジュール"""

from pathlib import Path

from titanic_analysis.application.analysis.analyzer import (
    DatasetAnalyzer,
)
from titanic_analysis.application.analysis.utils import set_display_options
from titanic_analysis.domain.dataset.dataset import Dataset, TestDataset, TrainDataset
from titanic_analysis.infrastructure.io.analysis.config_loader import load_config
from titanic_analysis.infrastructure.io.analysis.constants import (
    CONFIG_PATH,
    PATH_TEST,
    PATH_TRAIN,
)

__all__ = ["analyze"]


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


def analyze(
    config_file_name: Path = CONFIG_PATH,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    r"""データセットを解析する

    Args:
        config_file_name (Path):
            configファイルのパス
            デフォルトは'titanic_analysis\\infrastructure\\io\\analysis\\base.yaml'
        train_dataset_path (str):
            訓練用データのパス
            デフォルトは'data\\titanic\\train.csv'
        test_dataset_path (str):
            テスト用データのパス
            デフォルトは'data\\titanic\\test.csv'
    """
    prepare_display(config_file_name)

    dataset_list = [TrainDataset(train_dataset_path), TestDataset(test_dataset_path)]

    for dataset in dataset_list:
        describe_dataset(dataset)
