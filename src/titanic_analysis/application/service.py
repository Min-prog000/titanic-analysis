"""データセットの分析を行うモジュール"""

import logging
from logging import Logger
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import pydotplus
import torch
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
)
from sklearn.tree import export_graphviz
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from yaml import safe_dump

from titanic_analysis.application.constants import (
    ADDITIONAL_ENCODING_COLUMN,
    CATEGORICAL_FEATURES,
    COLUMN_NOT_MATCH_MESSAGE,
    ID_COLUMN,
    LOGGING_LEVEL_LITERALS,
    NUMERIC_FEATURES,
    PYTORCH_CASE_ID_PATH,
    PYTORCH_CONFIG_PATH,
    PYTORCH_TENSORBOARD_PATH,
    SEED,
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
    GRADIENT_BOOSTING_DECISION_TREE,
    LOGISTIC_REGRESSION,
    PATH_TEST,
    PATH_TRAIN,
)
from titanic_analysis.infrastructure.io.utils import CsvUtility
from titanic_analysis.infrastructure.logic.analysis.display import (
    describe_dataset,
    prepare_display,
)
from titanic_analysis.infrastructure.logic.build.test import test_loop
from titanic_analysis.infrastructure.logic.build.train import train_loop
from titanic_analysis.infrastructure.logic.build.utils import fix_seed, load_case_id
from titanic_analysis.infrastructure.logic.preprocess.preprocessor import (
    DatasetPreprocessor,
)

__all__ = [
    "analyze",
    "predict",
    "run_training_gradient_boosting",
    "run_training_logistic_regression",
    "run_training_pipeline_pytorch",
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


def run_training_logistic_regression(
    logger: Logger,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """Train with Logistic regression

    This function preprocess, training, and generate submission csv
        with logistic regression method.

    Args:
        logger (Logger): Logger generated in `main`.
        train_dataset_path (str, optional): Dataset path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional): Dataset path. Defaults to PATH_TEST.

    Raises:
        FalseComponentError: Raise when missing columns.
    """
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
    try:
        _ = x_train.columns.to_numpy().all() and x_test.columns.to_numpy().all()
    except FalseComponentError as _:
        raise FalseComponentError(COLUMN_NOT_MATCH_MESSAGE) from None

    # 正規化
    scaler = MinMaxScaler()

    # モデル生成
    weight = {0: 1.0, 1: 1.5}
    logreg = LogisticRegression(random_state=0, class_weight=weight)

    # パイプライン生成
    pipe_logreg = make_pipeline(scaler, logreg)

    # max_iterの範囲生成
    max_iter_scope = [np.int16(max_iter) for max_iter in np.linspace(100, 1000, num=10)]

    # ハイパーパラメータ設定
    params_logreg = {
        "logisticregression__C": np.logspace(-3, 3, num=7),
        "logisticregression__max_iter": max_iter_scope,
    }

    # LogisticRegression
    # グリッドサーチ
    search = GridSearchCV(pipe_logreg, params_logreg, n_jobs=2)
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
    best_logreg: LogisticRegression = search.best_estimator_.named_steps[
        "logisticregression"
    ]
    y_pred = best_logreg.predict(np.array(x_test))

    # 提出用データの作成
    y_pred_df = pd.DataFrame(y_pred, columns=[TARGET_COLUMN])
    y_pred_df_submission = pd.concat(
        [test_dataset.x[ID_COLUMN], y_pred_df],
        axis=1,
    )
    CsvUtility.output_csv(y_pred_df_submission, LOGISTIC_REGRESSION)

    # 提出用データの表示
    logger.info(y_pred_df_submission)


def run_training_gradient_boosting(
    logger: Logger,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """Train with Gradient boosting

    This function preprocess, training, and generate submission csv
        with gradient boosting decision tree

    Args:
        logger (Logger): Logger generated in `main`.
        train_dataset_path (str, optional): Dataset path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional): Dataset path. Defaults to PATH_TEST.

    Raises:
        FalseComponentError: Raise when missing columns.
    """
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    train_dataset_preprocessed, test_dataset_preprocessed = preprocess_load_data(
        logger,
        train_data,
        test_data,
    )

    # データセット
    train_labels = np.array(train_data.loc[:, TARGET_COLUMN])

    # train_dataset = TrainDataset(train_dataset_path)
    # test_dataset = TestDataset(test_dataset_path)

    # train_dataset_preprocessed = DatasetPreprocessor.preprocess_dataset(
    #     dataset=train_dataset.x,
    #     selected_features=SELECTED_FEATURES,
    #     encode_columns=CATEGORICAL_FEATURES,
    #     logger=logger,
    # )

    # test_dataset_preprocessed = DatasetPreprocessor.preprocess_dataset(
    #     dataset=test_dataset.x,
    #     selected_features=SELECTED_FEATURES,
    #     encode_columns=CATEGORICAL_FEATURES,
    #     logger=logger,
    # )

    logger.info(train_dataset_preprocessed)
    logger.info(test_dataset_preprocessed)

    # 訓練データ
    # x_train = train_dataset_preprocessed
    # y_train = train_dataset.y
    x_train = train_dataset_preprocessed
    y_train = train_labels

    # テストデータ
    x_test = test_dataset_preprocessed

    # 列名の数と名前が等しいことの確認
    # try:
    # _ = x_train.columns.to_numpy().all() and x_test.columns.to_numpy().all()
    # except FalseComponentError as _:
    # raise FalseComponentError(COLUMN_NOT_MATCH_MESSAGE) from None

    if not x_train.shape and x_test.shape:
        logger.info("Not match datasets shape.")
        return

    # 正規化
    scaler = MinMaxScaler()

    # GradientBoostingClassifier
    # グリッドサーチ
    gbdt = GradientBoostingClassifier(random_state=0)
    params_gbdt = {
        "gradientboostingclassifier__learning_rate": np.logspace(-2, -1, num=2),
        # "gradientboostingclassifier__n_estimators": range(100, 201, 100),
        "gradientboostingclassifier__max_depth": range(5, 8),
        "gradientboostingclassifier__max_features": range(7, x_train.shape[1]),
        # "gradientboostingclassifier__subsample": np.arange(0.1, 1.1, 0.1),
    }
    pipe_gbdt = make_pipeline(scaler, gbdt)
    search = GridSearchCV(pipe_gbdt, params_gbdt, n_jobs=2, verbose=10)
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
    best_gbdt: GradientBoostingClassifier = search.best_estimator_.named_steps[
        "gradientboostingclassifier"
    ]
    y_pred = best_gbdt.predict(np.array(x_test))

    # 提出用データの作成
    y_pred_df = pd.DataFrame(y_pred, columns=[TARGET_COLUMN])
    y_pred_df_submission = pd.concat(
        # [test_dataset.x[ID_COLUMN], y_pred_df],
        [test_data[ID_COLUMN], y_pred_df],
        axis=1,
    )
    CsvUtility.output_csv(y_pred_df_submission, GRADIENT_BOOSTING_DECISION_TREE)

    # 提出用データの表示
    logger.info(y_pred_df_submission)

    # 決定木の可視化
    # graphviz, pydotplus使用
    dot_data = export_graphviz(
        best_gbdt.estimators_[0, 0],
        out_file=None,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    logger.debug(type(graph))
    logger.debug(graph)

    if isinstance(graph, pydotplus.graphviz.Dot):
        graph.write(path="test_graph.png", format="png")

    # dtreeviz使用
    # logger.debug(train_data.columns.tolist())
    # viz = dtreeviz(
    #     best_gbdt.estimators_[0, 0],
    #     x_train,
    #     y_train,
    #     target_name="titanic",
    #     class_names=["not_survived", "survived"],
    #     feature_names=train_data.columns.tolist(),
    # )
    # filename_dtreeviz = Path("test_graph_dtreeviz.png")
    # viz.save(filename_dtreeviz)


def run_training_pipeline_pytorch(
    logger: Logger,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """Train and test using pytorch

    Args:
        logger (Logger): Logger for user information and debug.
        train_dataset_path (str, optional):
            Train dataset file path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional):
            Test dataset file path. Defaults to PATH_TEST.
    """
    fix_seed(SEED)

    prepare_display(ANALYSIS_CONFIG_PATH)

    config_path = Path(PYTORCH_CONFIG_PATH)
    config_loaded = load_training_config(config_path)

    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    train_data_preprocessed, test_data_preprocessed = preprocess_load_data(
        logger,
        train_data,
        test_data,
    )

    # データセット
    train_labels = np.array(train_data.loc[:, TARGET_COLUMN])

    logger.debug(train_labels)
    bin_count = np.bincount(train_labels)
    logger.debug(bin_count)

    false_percentage: np.float64 = bin_count[0] / np.sum(bin_count)
    true_percentage: np.float64 = bin_count[1] / np.sum(bin_count)
    logger.debug("false_percentage type: %s", type(false_percentage))
    logger.debug("true_percentage type: %s", type(true_percentage))
    logger.debug("False: %s %%", float(false_percentage))
    logger.debug("True: %s %%", float(true_percentage))

    train_dataset = TitanicTorchDataset(
        train_data_preprocessed,
        train_labels,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config_loaded.batch_size,
        shuffle=False,
    )

    feature_size = train_data_preprocessed.shape[1]
    model = NeuralNetwork(feature_size)
    logger.info(summary(model, (1, feature_size)))

    # 1出力
    pos_weight = torch.tensor([config_loaded.pos_weight])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # loss_fn = nn.BCELoss()

    # 2出力
    # weight = torch.tensor([0.9, 1.0])
    # loss_fn = nn.CrossEntropyLoss(weight=weight)

    train_accuracy_list = []
    train_loss_list = []
    train_correct_list = []

    optimizer = optim.Adam(
        model.parameters(),
        config_loaded.learning_rate,
        weight_decay=config_loaded.weight_decay,
    )

    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: config_loaded.gamma**epoch,
    )

    for epoch in range(config_loaded.epochs):
        train_epoch_accuracy, train_epoch_loss, train_epoch_correct, model = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            config_loaded.epochs,
            epoch,
        )
        train_accuracy_list.append(train_epoch_accuracy)
        train_loss_list.append(train_epoch_loss)
        train_correct_list.append(train_epoch_correct)

    # ケース番号
    case_id_path = Path(PYTORCH_CASE_ID_PATH)
    case_id = load_case_id(case_id_path)

    # TensorBoard のログ出力先
    root_log_dir = Path(PYTORCH_TENSORBOARD_PATH)
    # ラベル名
    main_tags = ["accuracy", "loss", "correct"]
    value_tag = f"case{case_id}"
    train_histories = [train_accuracy_list, train_loss_list, train_correct_list]
    for i in range(len(main_tags)):
        log_dir = root_log_dir.joinpath(main_tags[i])
        write_scalar_graph(log_dir, train_histories[i], main_tags[i], value_tag)

    # データセット
    test_labels = np.array([0] * test_data_preprocessed.shape[0])
    print(test_labels.shape)
    test_dataset = TitanicTorchDataset(
        test_data_preprocessed,
        test_labels,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config_loaded.batch_size,
        shuffle=False,
    )

    # test_dataset = torch.tensor(test_data_preprocessed, dtype=torch.float32)
    pred_list = test_loop(test_dataloader, model)
    logger.debug(len(test_data[ID_COLUMN].to_numpy()))
    logger.debug(len(pred_list))
    # logger.debug()

    submission_data = pd.DataFrame(
        {
            ID_COLUMN: test_data[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: pred_list,
        },
    )
    CsvUtility.output_csv(submission_data, "torch_neural-network")

    create_onnx_model(feature_size, model, case_id)

    config_save = {
        "model": {
            "case_id": case_id,
        },
    }
    config_save["model"].update(config_loaded.model_dump())
    yaml_output_path = Path(f"output/config/case{case_id}")
    yaml_output_path.mkdir(parents=True, exist_ok=True)
    config_file_name = Path(f"config_case{case_id}.yaml")
    config_file_path = yaml_output_path.joinpath(config_file_name)
    with config_file_path.open(mode="w", encoding="utf-8") as f:
        safe_dump(config_save, f, sort_keys=False)

    joblib.dump(case_id + 1, case_id_path)


def create_onnx_model(feature_size: int, model: NeuralNetwork, case_id: int) -> None:
    input_tensor = torch.rand((1, feature_size), dtype=torch.float32)

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
        # dynamo=True,
    )


def preprocess_load_data(
    logger: Logger,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_data_cleaned = clean_data(logger, train_data, SELECTED_FEATURES)
    test_data_cleaned = clean_data(logger, test_data, SELECTED_FEATURES)

    preprocessor = generate_preprocessor(
        StandardScaler(),
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor)])

    train_data_preprocessed = pipeline.fit_transform(
        train_data_cleaned,
    )
    test_data_preprocessed = pipeline.transform(test_data_cleaned)

    logger.debug("Train data shape: %s", train_data_preprocessed.shape)
    logger.debug("Test data shape: %s", test_data_preprocessed.shape)

    logger.debug("Column names: %s", preprocessor.get_feature_names_out())

    return train_data_preprocessed, test_data_preprocessed


def clean_data(
    logger: Logger,
    data_loaded: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    # Choose column
    data_filtered = data_loaded.loc[:, selected_features]
    logger.debug(data_filtered.columns)

    # Fill numeric column with mean
    mean_series = data_filtered.mean(numeric_only=True)
    mean_round = round(mean_series)
    data_cleaned = data_filtered.fillna(mean_round)

    # Fill "Embarked" column with mode
    # "S" is the most numerous category
    dataframe_groupby_embarked = data_cleaned.groupby(
        ADDITIONAL_ENCODING_COLUMN,
        dropna=False,
    )
    size_groupby_embarked = dataframe_groupby_embarked.size()
    mode_embarked_index = size_groupby_embarked.idxmax()
    data_cleaned[ADDITIONAL_ENCODING_COLUMN] = data_cleaned[
        ADDITIONAL_ENCODING_COLUMN
    ].fillna(
        mode_embarked_index,
    )

    return data_cleaned


def generate_preprocessor(
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_encoder: OneHotEncoder | None = None,
) -> ColumnTransformer:
    if categorical_encoder is None:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")

    transformers = []
    if numeric_features:
        transformers.append(("numeric", scaler, numeric_features))

    if categorical_features:
        transformers.append(("categorical", categorical_encoder, categorical_features))

    return ColumnTransformer(transformers=transformers)


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
    """Infer with ONNX model.

    Args:
        logger (Logger): Logger for user information and debug
        model_path (str): ONNX (`*.onnx`) file path
        train_dataset_path (str, optional): Train dataset path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional): Test dataset path. Defaults to PATH_TEST.
    """
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    logger.debug(SELECTED_FEATURES)

    _, test_data_preprocessed = preprocess_load_data(
        logger,
        train_data,
        test_data,
    )

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
