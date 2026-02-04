"""データセットの分析を行うモジュール"""

from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from titanic_analysis.application.exception.exception import FalseComponentError
from titanic_analysis.domain.dataset.sklearn_dataset import TestDataset, TrainDataset
from titanic_analysis.domain.dataset.torch_dataset import TitanicTorchDataset
from titanic_analysis.domain.model.torch import NeuralNetwork
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
from titanic_analysis.infrastructure.logic.preprocess.preprocessor import (
    DatasetPreprocessor,
)

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

    train_dataset_preprocessed = DatasetPreprocessor.preprocess_dataset(
        dataset=train_dataset.x,
        selected_columns=selected_columns,
        encode_columns=encode_columns,
        logger=logger,
    )

    test_dataset_preprocessed = DatasetPreprocessor.preprocess_dataset(
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

    # 列名の数と名前が等しいことの確認
    if x_train.columns.to_numpy().all() and x_test.columns.to_numpy().all():
        columns_names = x_train.columns
    else:
        msg = "NotMatchSizeError: either array has one or more false components."
        raise FalseComponentError(msg)

    # 正規化
    scaler = MinMaxScaler()

    # モデル生成
    weight = {0: 1.0, 1: 1.5}
    logreg = LogisticRegression(random_state=0, class_weight=weight)
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
    logger.info("Grid search best score: %s", search.best_score_)
    logger.info("Hyper parameters: %s", search.best_params_)

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


def train_loop(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.BCEWithLogitsLoss | nn.BCELoss,
    optimizer: optim.Adam | optim.SGD,
    epochs: int,
    epoch: int,
) -> NeuralNetwork:
    epoch_accuracy = 0
    epoch_loss = 0
    total_count = 0

    model.train()
    batch_size = len(dataloader)

    with tqdm(dataloader) as pbar:
        pbar.set_description(f"[Epoch {epoch + 1}/{epochs}]")
        for batch in pbar:
            data, labels = get_data_with_type_annotation(batch)
            batch_size = labels.shape[0]
            # 予測と損失の計算
            proba: Tensor = model(data)
            # threshold = 0.5
            # print((proba >= threshold).long())
            # print(type((proba >= threshold).long()))
            # pred = (proba >= threshold).float()

            # pred_tensor_class = model(x)
            # pred = torch.argmax(pred_tensor_class, dim=1).unsqueeze(dim=1)

            loss: Tensor = loss_fn(proba, labels)

            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(proba)
            # print(labels)

            threshold = 0.5
            pred = (proba >= threshold).float()

            correct = (pred == labels).sum().item()
            # print(correct)
            accuracy = correct / batch_size
            # print(accuracy)

            total_count += batch_size

            pbar.set_postfix({"accuracy": accuracy, "loss": loss.item()})

    return model


def get_data_with_type_annotation(batch: list) -> tuple[Tensor, Tensor]:
    data: Tensor = batch[0]
    labels: Tensor = batch[1]
    return data, labels


@torch.no_grad()
def test_loop(
    dataset: Tensor,
    model: NeuralNetwork,
) -> list[int]:
    pred_list = []

    with tqdm(dataset) as pbar:
        for x in pbar:
            proba = model(x)

            # BCEWithLogitsLoss
            threshold = 0.5
            pred = int(proba >= threshold)

            # BCELoss
            # pred = int(torch.argmax(proba))

            pred_list.append(pred)
        print(len(pred_list))

    return pred_list


def run_torch_training_pipeline(
    logger: Logger,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
):
    prepare_display(ANALYSIS_CONFIG_PATH)

    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

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

    train_data_filtered = train_data.loc[:, selected_columns]
    logger.debug(train_data_filtered.columns)
    train_data_mean = train_data_filtered.mean(numeric_only=True)
    train_fill_values_round = round(train_data_mean)
    train_data_preprocessed = train_data_filtered.fillna(train_fill_values_round)
    logger.debug(train_data_preprocessed.columns)

    test_data_filtered = test_data.loc[:, selected_columns]
    logger.debug(test_data_filtered.columns)
    test_data_mean = test_data_filtered.mean(numeric_only=True)
    test_fill_values_round = round(test_data_mean)
    test_data_preprocessed = test_data_filtered.fillna(test_fill_values_round)
    logger.debug(test_data_preprocessed.columns)

    numeric_features = ["Age", "Fare", "SibSp", "Parch"]
    categorical_columns = ["Pclass", "Sex", "Embarked"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ],
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor)])

    train_data_preprocessed = pipeline.fit_transform(train_data_preprocessed)
    test_data_preprocessed = pipeline.transform(test_data_preprocessed)

    logger.debug(train_data_preprocessed.shape)
    logger.debug(test_data_preprocessed.shape)

    # データセット
    logger.debug(type(train_data["Survived"][0]))
    train_labels = np.array(train_data["Survived"])
    train_dataset = TitanicTorchDataset(
        train_data_preprocessed,
        train_labels,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )

    feature_size = train_data_preprocessed.shape[1]
    model = NeuralNetwork(feature_size)

    # 1出力
    loss_fn = nn.BCEWithLogitsLoss()

    # 2出力
    # weight = torch.tensor([0.5, 1.0])
    # loss_fn = nn.BCELoss(weight=weight)

    optimizer = optim.Adam(model.parameters())
    epochs = 100
    for epoch in range(epochs):
        model = train_loop(train_dataloader, model, loss_fn, optimizer, epochs, epoch)

    test_dataset = torch.tensor(test_data_preprocessed, dtype=torch.float32)

    pred_list = test_loop(test_dataset, model)

    logger.debug(len(test_data["PassengerId"].to_numpy()))
    logger.debug(len(pred_list))

    pred_df = pd.DataFrame(
        {
            "PassengerId": test_data["PassengerId"].to_numpy(),
            "Survived": pred_list,
        },
    )
    CsvUtility.output_csv(pred_df, "torch_neuralnetwork")
