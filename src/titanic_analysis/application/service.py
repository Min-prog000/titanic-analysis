"""データセットの分析を行うモジュール"""

from pathlib import Path

from titanic_analysis.domain.dataset.dataset import TestDataset, TrainDataset
from titanic_analysis.infrastructure.io.analysis.constants import (
    CONFIG_PATH as ANALYSIS_CONFIG_PATH,
)
from titanic_analysis.infrastructure.io.analysis.constants import (
    PATH_TEST,
    PATH_TRAIN,
)
from titanic_analysis.infrastructure.io.training_pipeline.config_loader import (
    load_config,
)
from titanic_analysis.infrastructure.io.training_pipeline.constants import (
    CONFIG_PATH as TRAINING_CONFIG_PATH,
)
from titanic_analysis.infrastructure.logic.analysis.display import (
    describe_dataset,
    prepare_display,
)
from titanic_analysis.infrastructure.logic.preprocess import preprocessor
from titanic_analysis.interface.log.logger import TitanicLogger

__all__ = ["analyze"]


def analyze(
    config_file_name: Path = ANALYSIS_CONFIG_PATH,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    r"""データセットを解析する

    Args:
        config_file_name (Path):
            configファイルのパス
            デフォルトは'titanic_analysis\\infrastructure\\io\\training_pipeline\\base.yaml'
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


def run_training_pipeline(
    config_file_name: Path = TRAINING_CONFIG_PATH,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    config_dto = load_config(config_file_name)

    train_dataset = TrainDataset(train_dataset_path)
    test_dataset = TestDataset(test_dataset_path)

    # 抽出後の列名（共通）
    selected_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    # 共通
    encode_columns = ["Pclass", "Sex", "Embarked"]

    titanic_logger = TitanicLogger(__name__)
    logger = titanic_logger.logger

    train_dataset_preprocessed = preprocessor.DatasetPreprocessor.preprocess_dataset(
        dataset=train_dataset.x,
        selected_columns=selected_columns,
        encode_columns=encode_columns,
        logger=logger,
    )

    test_dataset_preprocessed = preprocessor.DatasetPreprocessor.preprocess_dataset(
        dataset=test_dataset.x,
        selected_columns=selected_columns,
        encode_columns=encode_columns,
        logger=logger,
    )

    logger.info(train_dataset_preprocessed)
    logger.info(test_dataset_preprocessed)
