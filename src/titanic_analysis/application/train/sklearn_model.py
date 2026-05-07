"""Training use case using sklearn"""

import sys
from logging import Logger
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydotplus
from pandas import Series
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz

from titanic_analysis.application.constants import (
    CASE_ID_PATH,
    GBDT_CONFIG_PATH,
    ID_COLUMN,
    LOGREG_CONFIG_PATH,
    PIPELINE_PREFIX_GBDT,
    PIPELINE_PREFIX_LOGREG,
    TARGET_COLUMN,
)
from titanic_analysis.application.preprocess import preprocess_load_data
from titanic_analysis.domain.model.types import SklearnModelTypes
from titanic_analysis.infrastructure.io.analysis.config_loader import (
    load_gradient_boosting_classifier_config,
    load_logistic_regression_config,
)
from titanic_analysis.infrastructure.io.constants import (
    GRADIENT_BOOSTING_DECISION_TREE,
    LOGISTIC_REGRESSION,
    PATH_TEST,
    PATH_TRAIN,
)
from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    GradientBoostingClassifierConfigDTO,
    LogisticRegressionConfigDTO,
)
from titanic_analysis.infrastructure.io.utils import CsvUtility
from titanic_analysis.infrastructure.logic.build.utils import load_case_id
from titanic_analysis.infrastructure.user.constants import TrainMethod

__all__ = ["train_sklearn_model"]


def train_sklearn_model(
    logger: Logger,
    method_id: int,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """Train using sklearn models

    This function preprocess, training, and generate submission csv
        using sklearn models (ex. LogisticRegression, GradientBoostingClassifier, ...)

    Args:
        logger (Logger): Logger generated in `main`.
        method_id (int): Training method id.
        train_dataset_path (str, optional): Dataset path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional): Dataset path. Defaults to PATH_TEST.

    Raises:
        FalseComponentError: Raise when missing columns.
    """
    # data loading
    passenger_id, x_train, y_train, x_test = create_dataset(
        logger,
        train_dataset_path,
        test_dataset_path,
    )

    # training
    csv_postfix, dump_folder_name, model_best = run_grid_search(
        logger,
        method_id,
        x_train,
        y_train,
    )

    # prediction(create submission file)
    predict_with_sklearn_method(logger, passenger_id, x_test, csv_postfix, model_best)

    # model save
    # 1. save visualized tree
    if isinstance(model_best, GradientBoostingClassifier):
        save_tree_graph(model_best)
    # 2. save model
    save_model(dump_folder_name, model_best)


def create_dataset(
    logger: Logger,
    train_dataset_path: str,
    test_dataset_path: str,
) -> tuple[Series, np.ndarray, np.ndarray, np.ndarray]:
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    # Preprocess
    train_dataset_preprocessed, test_dataset_preprocessed = preprocess_load_data(
        logger,
        train_data,
        test_data,
    )

    # データセット
    train_labels = np.array(train_data.loc[:, TARGET_COLUMN])

    logger.info(train_dataset_preprocessed)
    logger.info(test_dataset_preprocessed)

    # 訓練データ
    x_train = train_dataset_preprocessed
    y_train = train_labels

    # テストデータ
    x_test = test_dataset_preprocessed

    # データセットのサイズが等しいことの確認
    # TODO: Revise to be able to compare preprocessed data columns
    if not (x_train.shape and x_test.shape):
        logger.info("Not match datasets shape.")
        sys.exit()

    return test_data[ID_COLUMN], x_train, y_train, x_test


def run_grid_search(
    logger: Logger,
    method_id: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[str, str, SklearnModelTypes]:
    # Scaling
    scaler = MinMaxScaler()

    # Parameters
    model = None
    params_grid = {}
    pipeline_prefix = ""
    csv_postfix = ""
    dump_folder_name = ""

    # LogisticRegression
    if method_id == TrainMethod.LOGISTIC_REGRESSION.value:
        config_path = Path(LOGREG_CONFIG_PATH)
        config_loaded = load_logistic_regression_config(config_path)
        model = LogisticRegression(random_state=config_loaded.random_state)
        params_grid: dict = get_params_grid_logreg(config_loaded)
        pipeline_prefix = PIPELINE_PREFIX_LOGREG
        csv_postfix = LOGISTIC_REGRESSION
        dump_folder_name = "logreg"
    # GradientBoostingClassifier
    elif method_id == TrainMethod.GRADIENT_BOOSTING.value:
        config_path = Path(GBDT_CONFIG_PATH)
        config_loaded = load_gradient_boosting_classifier_config(config_path)
        model = GradientBoostingClassifier(random_state=config_loaded.random_state)
        params_grid: dict = get_params_grid_gbdt(x_train.shape[1], config_loaded)
        pipeline_prefix = PIPELINE_PREFIX_GBDT
        csv_postfix = GRADIENT_BOOSTING_DECISION_TREE
        dump_folder_name = "gbdt"

    pipeline = make_pipeline(scaler, model)

    # グリッドサーチ
    search = GridSearchCV(pipeline, params_grid, n_jobs=2, verbose=10)
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
    model_best: SklearnModelTypes = search.best_estimator_.named_steps[pipeline_prefix]

    return csv_postfix, dump_folder_name, model_best


def get_params_grid_logreg(
    config_loaded: LogisticRegressionConfigDTO,
) -> dict:
    # max_iterの範囲生成
    max_iter_scope = [
        np.int16(max_iter)
        for max_iter in np.linspace(config_loaded.max_iter, 1000, num=10)
    ]
    return {
        "logisticregression__C": np.logspace(config_loaded.C, 3, num=7),
        "logisticregression__class_weight": [
            config_loaded.class_weight,
            {0: 1.0, 1: 0.5},
        ],
        "logisticregression__max_iter": max_iter_scope,
    }


def get_params_grid_gbdt(
    max_features_max: int,
    config_loaded: GradientBoostingClassifierConfigDTO,
) -> dict:
    return {
        f"{PIPELINE_PREFIX_GBDT}__learning_rate": np.logspace(
            config_loaded.learning_rate,
            -1,
            num=2,
        ),
        # f"{PIPELINE_PREFIX_GBDT}__n_estimators": range(100, 201, 100),
        f"{PIPELINE_PREFIX_GBDT}__max_depth": range(config_loaded.max_depth, 8),
        f"{PIPELINE_PREFIX_GBDT}__max_features": range(
            config_loaded.max_features,
            max_features_max,
        ),
        # f"{PIPELINE_PREFIX_GBDT}__subsample": np.arange(0.1, 1.1, 0.1),
    }


def predict_with_sklearn_method(
    logger: Logger,
    passenger_id: Series,
    x_test: np.ndarray,
    csv_postfix: str,
    model_best: SklearnModelTypes,
) -> None:
    # predict
    y_pred = model_best.predict(np.array(x_test))

    # create submission data
    # 提出用データの作成
    y_pred_df = pd.DataFrame(y_pred, columns=[TARGET_COLUMN])
    y_pred_df_submission = pd.concat(
        [passenger_id, y_pred_df],
        axis=1,
    )

    # output
    CsvUtility.output_csv(y_pred_df_submission, csv_postfix)

    # 提出用データの表示
    logger.info(y_pred_df_submission)


def save_tree_graph(model_best: GradientBoostingClassifier) -> None:
    # graphviz, pydotplus使用
    dot_data = export_graphviz(
        model_best.estimators_[0, 0],
        out_file=None,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    if isinstance(graph, pydotplus.graphviz.Dot):
        graph.write(path="test_graph.png", format="png")

    # dtreeviz使用
    # logger.debug(train_data.columns.tolist())
    # viz = dtreeviz.dtreeviz(
    #     best_gbdt.estimators_[0, 0],
    #     x_train,
    #     y_train,
    #     target_name="titanic",
    #     class_names=["not_survived", "survived"],
    #     feature_names=train_data.columns.tolist(),
    # )
    # filename_dtreeviz = Path("test_graph_dtreeviz.png")
    # viz.save(filename_dtreeviz)


def save_model(dump_folder_name: str, model_best: SklearnModelTypes) -> None:
    # case番号はPytorchと共有
    case_id_path = Path(CASE_ID_PATH)
    case_id = load_case_id(case_id_path)
    dump_folder_path = Path(f".\\model\\{dump_folder_name}\\case_{case_id}")
    model_file_name = Path(f"case_{case_id}.joblib")
    dump_folder_path.mkdir(parents=True, exist_ok=True)
    model_dump_path = dump_folder_path.joinpath(model_file_name)
    joblib.dump(model_best, model_dump_path, protocol=5)
    joblib.dump(case_id + 1, CASE_ID_PATH)
