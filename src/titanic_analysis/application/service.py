"""データセットの分析を行うモジュール"""

import logging
from logging import Logger
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from titanic_analysis.application.constants import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    LOGGING_LEVEL_LITERALS,
    NUMERIC_FEATURES,
    SELECTED_FEATURES,
    TARGET_COLUMN,
)
from titanic_analysis.application.exception.exception import FalseComponentError
from titanic_analysis.domain.dataset.sklearn_dataset import TestDataset, TrainDataset
from titanic_analysis.domain.dataset.torch_dataset import TitanicTorchDataset
from titanic_analysis.domain.model.torch import NeuralNetwork
from titanic_analysis.infrastructure.io.analysis.config_loader import (
    load_training_config,
)
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
from titanic_analysis.infrastructure.logic.build.test import test_loop
from titanic_analysis.infrastructure.logic.build.train import train_loop
from titanic_analysis.infrastructure.logic.preprocess.preprocessor import (
    DatasetPreprocessor,
)

__all__ = [
    "analyze",
    "predict",
    "run_training_pipeline_pytorch",
    "run_training_pipeline_sklearn",
]


def analyze(
    logger: Logger,
    config_file_name: Path = ANALYSIS_CONFIG_PATH,
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


def run_training_pipeline_sklearn(
    logger: Logger,
    config_file_name: Path = TRAINING_CONFIG_PATH,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """The function preprocess, training, and generate submission csv

    Args:
        logger (Logger): Logger generated in `main`.
        config_file_name (Path, optional):
            Config file name with absolute path. Defaults to TRAINING_CONFIG_PATH.
        train_dataset_path (str, optional): Dataset path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional): Dataset path. Defaults to PATH_TEST.

    Raises:
        FalseComponentError: Raise when missing columns.
    """
    config_dto = load_config(config_file_name)

    train_dataset = TrainDataset(train_dataset_path)
    test_dataset = TestDataset(test_dataset_path)

    train_dataset_preprocessed = DatasetPreprocessor.preprocess_dataset(
        dataset=train_dataset.x,
        selected_features=SELECTED_FEATURES,
        encode_columns=CATEGORICAL_FEATURES,
        logger=logger,
    )

    test_dataset_preprocessed = DatasetPreprocessor.preprocess_dataset(
        dataset=test_dataset.x,
        selected_features=SELECTED_FEATURES,
        encode_columns=CATEGORICAL_FEATURES,
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
        _ = x_train.columns
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
    y_pred_df = pd.DataFrame(y_pred, columns=[TARGET_COLUMN])
    y_pred_df_submission = pd.concat(
        [test_dataset.x[ID_COLUMN], y_pred_df],
        axis=1,
    )
    CsvUtility.output_csv(y_pred_df_submission, LOGISTIC_REGRESSION)

    # 提出用データの表示
    logger.info(y_pred_df_submission)


def run_training_pipeline_pytorch(
    logger: Logger,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
):
    prepare_display(ANALYSIS_CONFIG_PATH)

    config_path = Path("config/model/base.yaml")
    config = load_training_config(config_path)

    train_valid_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    train_data_filtered = train_valid_data.loc[:, SELECTED_FEATURES]
    logger.debug(train_data_filtered.columns)

    train_data_mean = train_data_filtered.mean(numeric_only=True)
    train_fill_values_round = round(train_data_mean)
    train_data_preprocessed = train_data_filtered.fillna(train_fill_values_round)

    embarked_groupby = train_data_preprocessed.groupby("Embarked", dropna=False)
    # "S" is the most numerous category
    embarked_groupby_size = embarked_groupby.size()
    mode_embarked_index = embarked_groupby_size.idxmax()
    train_data_preprocessed["Embarked"] = train_data_preprocessed["Embarked"].fillna(
        mode_embarked_index,
    )
    embarked_groupby_after = train_data_preprocessed.groupby(
        "Embarked",
        dropna=False,
    ).sum()
    logger.debug(embarked_groupby_after)
    logger.debug(train_data_preprocessed.columns)

    test_data_filtered = test_data.loc[:, SELECTED_FEATURES]
    logger.debug(test_data_filtered.columns)
    test_data_mean = test_data_filtered.mean(numeric_only=True)
    test_fill_values_round = round(test_data_mean)
    test_data_preprocessed = test_data_filtered.fillna(test_fill_values_round)
    logger.debug(test_data_preprocessed.columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor)])

    train_data_preprocessed = pipeline.fit_transform(
        train_data_preprocessed,
    )
    test_data_preprocessed = pipeline.transform(test_data_preprocessed)

    logger.debug("Train data shape: %s", train_data_preprocessed.shape)
    logger.debug("Test data shape: %s", test_data_preprocessed.shape)

    logger.debug("Column names: %s", preprocessor.get_feature_names_out())

    # データセット
    train_labels = np.array(train_valid_data.loc[:, TARGET_COLUMN])
    train_dataset = TitanicTorchDataset(
        train_data_preprocessed,
        train_labels,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    feature_size = train_data_preprocessed.shape[1]
    model = NeuralNetwork(feature_size)

    logger.info(summary(model, (train_data_preprocessed.shape[1],)))

    # 1出力
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()

    # 2出力
    # weight = torch.tensor([0.9, 1.0])
    # loss_fn = nn.CrossEntropyLoss(weight=weight)

    train_accuracy_list = []
    train_loss_list = []
    train_correct_list = []

    optimizer = optim.Adam(model.parameters(), config.learning_rate)
    for epoch in range(config.epochs):
        train_epoch_accuracy, train_epoch_loss, train_epoch_correct, model = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            config.epochs,
            epoch,
        )
        train_accuracy_list.append(train_epoch_accuracy)
        train_loss_list.append(train_epoch_loss)
        train_correct_list.append(train_epoch_correct)

    # TensorBoard のログ出力先
    root_log_dir = Path("./tensorboard_log")

    # ケース番号
    case_id_path = Path("config/id/case.joblib")
    if case_id_path.exists():
        case_id = joblib.load(case_id_path)
    else:
        case_id_path.parent.mkdir(exist_ok=True)
        case_id = 1

    # ラベル名
    main_tags = ["accuracy", "loss", "correct"]
    value_tag = f"case{case_id}"
    train_histories = [train_accuracy_list, train_loss_list, train_correct_list]

    # 例として 100 ステップ分のデータを記録
    for i in range(len(main_tags)):
        log_dir = root_log_dir.joinpath(main_tags[i])
        write_scalar_graph(log_dir, train_histories[i], main_tags[i], value_tag)

    test_dataset = torch.tensor(test_data_preprocessed, dtype=torch.float32)

    pred_list = test_loop(test_dataset, model)

    logger.debug(len(test_data[ID_COLUMN].to_numpy()))
    logger.debug(len(pred_list))

    pred_df = pd.DataFrame(
        {
            ID_COLUMN: test_data[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: pred_list,
        },
    )
    CsvUtility.output_csv(pred_df, "torch_neuralnetwork")

    input_tensor = torch.rand((1, 1, feature_size), dtype=torch.float32)

    onnx_dir_path = Path(f"model/onnx/case{case_id}")
    onnx_dir_path.mkdir(parents=True, exist_ok=True)

    onnx_file_name = Path(f"case{case_id}.onnx")
    onnx_file_path = onnx_dir_path.joinpath(onnx_file_name)

    set_onnx_logger()

    torch.onnx.export(
        model,
        (input_tensor,),
        onnx_file_path,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
    )

    joblib.dump(case_id + 1, case_id_path)


def set_onnx_logger(
    level_onnxscript: LOGGING_LEVEL_LITERALS = logging.ERROR,
    level_onnx_ir: LOGGING_LEVEL_LITERALS = logging.ERROR,
) -> None:
    logging.getLogger("onnxscript").setLevel(level_onnxscript)
    logging.getLogger("onnx_ir").setLevel(level_onnx_ir)


def write_scalar_graph(
    log_dir: Path,
    plot_list: list,
    main_tag: str,
    value_tag: str,
) -> None:
    writer = SummaryWriter(log_dir=log_dir)
    for step in range(len(plot_list)):
        # add_scalars を使うと、1 つのグラフに複数線が色分けされて表示される
        writer.add_scalars(main_tag, {value_tag: plot_list[step]}, step)

    writer.close()


def predict(
    logger: Logger,
    model_path: str,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    logger.debug(SELECTED_FEATURES)

    train_data_filtered = train_data.loc[:, SELECTED_FEATURES]
    logger.debug(train_data_filtered.columns)

    train_data_mean = train_data_filtered.mean(numeric_only=True)
    train_fill_values_round = round(train_data_mean)
    train_data_preprocessed = train_data_filtered.fillna(train_fill_values_round)
    logger.debug(train_data_preprocessed.columns)

    test_data_filtered = test_data.loc[:, SELECTED_FEATURES]
    logger.debug(test_data_filtered.columns)

    test_data_mean = test_data_filtered.mean(numeric_only=True)
    test_fill_values_round = round(test_data_mean)
    test_data_preprocessed = test_data_filtered.fillna(test_fill_values_round)
    logger.debug(test_data_preprocessed.columns)

    logger.debug(NUMERIC_FEATURES)
    logger.debug(CATEGORICAL_FEATURES)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor)])

    train_data_preprocessed = pipeline.fit_transform(train_data_preprocessed)
    test_data_preprocessed = pipeline.transform(test_data_preprocessed)

    logger.debug(test_data_preprocessed.shape)

    # データセット
    test_dataset = np.array(test_data_preprocessed, dtype=np.float32)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name: str = session.get_inputs()[0].name
    output_name: str = session.get_outputs()[0].name

    for onnx_input in session.get_inputs():
        print(onnx_input.name, onnx_input.shape, onnx_input.type)

    logger.debug(test_dataset.shape)

    output_list = []

    for test_data in test_dataset:
        input_data = np.expand_dims(test_data, axis=(0, 1))
        output = session.run(
            output_names=[output_name],
            input_feed={input_name: input_data},
            run_options=None,
        )
        output_list.append(output)

    output_df = pd.DataFrame(output_list)
    logger.debug(output_df.shape)

    model_file_name = Path(model_path).stem
    output_df_folder_path = Path(f"output/onnx_inference/{model_file_name}")
    output_df_folder_path.mkdir(parents=True, exist_ok=True)
    output_df_file_name = Path(f"{model_file_name}_output.csv")
    output_df_file_path = output_df_folder_path.joinpath(output_df_file_name)
    output_df.to_csv(output_df_file_path)
