"""データセットの分析を行うモジュール"""

from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from titanic_analysis.application.exception.exception import FalseComponentError
from titanic_analysis.domain.dataset.dataset import TestDataset, TrainDataset
from titanic_analysis.infrastructure.io.analysis.constants import (
    CONFIG_PATH as ANALYSIS_CONFIG_PATH,
)
from titanic_analysis.infrastructure.io.constants import (
    LOGISTIC_REGRESSION,
    PATH_TEST,
    PATH_TRAIN,
)
from titanic_analysis.infrastructure.io.training_pipeline.config_loader import (
    load_config,
)
from titanic_analysis.infrastructure.io.training_pipeline.constants import (
    CONFIG_PATH as TRAINING_CONFIG_PATH,
)
from titanic_analysis.infrastructure.io.utils import CsvUtility
from titanic_analysis.infrastructure.logic.analysis.display import (
    describe_dataset,
    prepare_display,
)
from titanic_analysis.infrastructure.logic.preprocess import preprocessor

__all__ = ["analyze", "run_training_pipeline"]


def analyze(
    config_file_name: Path = ANALYSIS_CONFIG_PATH,
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


def run_training_pipeline(
    logger: Logger,
    config_file_name: Path = TRAINING_CONFIG_PATH,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """The function preprocess, training, and generate submission csv

    Args:
        logger (Logger): Logger generated in `main`.
        config_file_name (Path, optional): Config file name with absolute path. Defaults to TRAINING_CONFIG_PATH.
        train_dataset_path (str, optional): Dataset path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional): Dataset path. Defaults to PATH_TEST.

    Raises:
        FalseComponentError: Raise when missing columns.
    """
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

    # 訓練データ
    x_train = train_dataset_preprocessed
    y_train = train_dataset.y

    # テストデータ
    x_test = test_dataset_preprocessed

    # 列名が過不足なく等しいことの確認
    if x_train.columns.to_numpy().all() and x_test.columns.to_numpy().all():
        columns_names = x_train.columns
    else:
        msg = "NotMatchSizeError: either array has one or more false components."
        raise FalseComponentError(msg)

    # 正規化
    scaler = MinMaxScaler()

    # モデル生成
    logreg = LogisticRegression(random_state=0)
    # パイプライン生成
    pipe = make_pipeline(scaler, logreg)

    # max_iterの範囲生成
    max_iter_scope = [np.int16(max_iter) for max_iter in np.linspace(100, 1000, num=10)]

    # ハイパーパラメータ設定
    params_logreg = {
        "logisticregression__C": np.logspace(-3, 3, num=7),
        "logisticregression__max_iter": max_iter_scope,
    }

    # グリッドサーチ
    search = GridSearchCV(pipe, params_logreg, n_jobs=2)
    search.fit(x_train, y_train)

    # グリッドサーチ結果の表示
    result_search = search.cv_results_
    result_search_df = pd.DataFrame(result_search).iloc[:, 4:]
    result_search_df_rounded = result_search_df.round(3)
    logger.info(result_search_df_rounded)

    # グリッドサーチのベストスコア表示
    best_score = search.best_score_
    logger.info("Grid search best score: %s", best_score)

    # 最高精度のモデルによる推論
    model: LogisticRegression = search.best_estimator_.named_steps["logisticregression"]
    y_pred = model.predict(np.array(x_test))

    # 提出用データの作成
    y_pred_df = pd.DataFrame(y_pred, columns=["Survived"])
    y_pred_df_submission = pd.concat(
        [test_dataset.x["PassengerId"], y_pred_df],
        axis=1,
    )
    CsvUtility.output_csv(y_pred_df_submission, LOGISTIC_REGRESSION)

    # 提出用データの表示
    logger.info(y_pred_df_submission)
