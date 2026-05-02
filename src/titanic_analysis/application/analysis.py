"""Analysis use case"""

from logging import Logger
from pathlib import Path

from titanic_analysis.domain.dataset.sklearn_dataset import TestDataset, TrainDataset
from titanic_analysis.infrastructure.io.analysis.constants import CONFIG_PATH
from titanic_analysis.infrastructure.io.constants import PATH_TEST, PATH_TRAIN
from titanic_analysis.infrastructure.logic.analysis.display import (
    describe_dataset,
    prepare_display,
)


def analyze(
    logger: Logger,
    config_file_name: Path = CONFIG_PATH,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    r"""データセットを解析する

    Args:
        logger (Logger):
            ロガー
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
        describe_dataset(dataset, logger)
