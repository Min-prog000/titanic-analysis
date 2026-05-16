"""Training use case using sklearn"""

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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz

from titanic_analysis.application.constants import (
    CASE_ID_PATH,
    GBDT_CONFIG_PATH,
    LOGREG_CONFIG_PATH,
    PIPELINE_PREFIX_GBDT,
    PIPELINE_PREFIX_LOGREG,
)
from titanic_analysis.application.train.utils import (
    create_dataset,
    generate_submission_dataframe,
    get_case_id,
    save_case_id,
)
from titanic_analysis.domain.model.types import SklearnModelTypes
from titanic_analysis.infrastructure.io.analysis.config_loader import (
    load_gradient_boosting_classifier_config,
    load_logistic_regression_config,
)
from titanic_analysis.infrastructure.io.constants import (
    GRADIENT_BOOSTING_DECISION_TREE,
    LOGISTIC_REGRESSION,
    MODEL_SAVE_PROTOCOL,
    PATH_TEST,
    PATH_TRAIN,
)
from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    GradientBoostingClassifierConfigDTO,
    LogisticRegressionConfigDTO,
)
from titanic_analysis.infrastructure.io.utils import CsvUtility
from titanic_analysis.infrastructure.user.constants import TrainMethod

__all__ = ["train_sklearn_model"]


def train_sklearn_model(
    logger: Logger,
    method_id: int,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """Train using sklearn models

    This function performs preprocess, training, and generate submission csv
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
    x_train, y_train, x_test, passenger_ids = create_dataset(
        logger,
        train_dataset_path,
        test_dataset_path,
    )

    # training
    csv_postfix, dump_folder_name, best_model = run_grid_search(
        logger,
        method_id,
        x_train,
        y_train,
    )

    # prediction(create submission file)
    y_pred = predict(logger, passenger_ids, x_test, best_model)

    # output
    CsvUtility.output_csv(y_pred, csv_postfix)

    # Save experiment results and case id
    save_artifacts(dump_folder_name, best_model)


def save_artifacts(dump_folder_name: str, best_model: SklearnModelTypes) -> None:
    # 1. Get current case id
    case_id = get_case_id(CASE_ID_PATH)

    # 2. Save tree visualization (if model is tree)
    if isinstance(best_model, GradientBoostingClassifier):
        save_tree_graph(best_model)

    # 3. Save model
    save_model(dump_folder_name, case_id, best_model)

    # 4. Save next case id
    save_case_id(case_id, CASE_ID_PATH)


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
        dump_folder_name = LOGISTIC_REGRESSION
    # GradientBoostingClassifier
    elif method_id == TrainMethod.GRADIENT_BOOSTING.value:
        config_path = Path(GBDT_CONFIG_PATH)
        config_loaded = load_gradient_boosting_classifier_config(config_path)
        model = GradientBoostingClassifier(random_state=config_loaded.random_state)
        params_grid: dict = get_params_grid_gbdt(x_train.shape[1], config_loaded)
        pipeline_prefix = PIPELINE_PREFIX_GBDT
        csv_postfix = GRADIENT_BOOSTING_DECISION_TREE
        dump_folder_name = GRADIENT_BOOSTING_DECISION_TREE

    pipeline = make_pipeline(scaler, model)

    # グリッドサーチ
    search = GridSearchCV(pipeline, params_grid, n_jobs=2, verbose=10)
    search.fit(x_train, y_train)

    # グリッドサーチ結果の表示
    log_grid_search_result(logger, search)

    # グリッドサーチのベストスコア表示
    log_best_model_info(logger, search)

    # 最高精度のモデルによる推論
    best_model: SklearnModelTypes = get_search_best_model(pipeline_prefix, search)

    return csv_postfix, dump_folder_name, best_model


def get_search_best_model(
    pipeline_prefix: str,
    search: GridSearchCV[Pipeline],
) -> SklearnModelTypes:
    return search.best_estimator_.named_steps[pipeline_prefix]


def log_best_model_info(logger: Logger, search: GridSearchCV) -> None:
    log_grid_search_best_score(logger, search)
    log_grid_search_best_parameters(logger, search)


def log_grid_search_best_parameters(logger: Logger, search: GridSearchCV) -> None:
    best_params = get_search_best_params(search)

    logger.info("Grid search Best hyper parameters: %s", best_params)


def get_search_best_params(search: GridSearchCV) -> dict:
    return search.best_params_


def log_grid_search_best_score(logger: Logger, search: GridSearchCV) -> None:
    best_score = get_search_best_score(search)

    logger.info("Grid search best score: %s", best_score)


def get_search_best_score(search: GridSearchCV) -> float:
    return search.best_score_


def log_grid_search_result(logger: Logger, search: GridSearchCV) -> None:
    # Preprocess
    result_rounded = generate_grid_search_result(search)

    # Log grid search result
    logger.info("\n%s", result_rounded)


def generate_grid_search_result(search: GridSearchCV) -> pd.DataFrame:
    result_dict = get_grid_search_result(search)
    result_df = dict_to_df(result_dict)
    result_without_execution_time = eliminate_execution_time(result_df)

    return round_result_figure(result_without_execution_time)


def round_result_figure(
    result_search_df: pd.DataFrame,
    figure: int = 3,
) -> pd.DataFrame:
    return result_search_df.round(figure)


def eliminate_execution_time(search_result: pd.DataFrame) -> pd.DataFrame:
    return search_result.iloc[:, 4:]


def dict_to_df(result_search: dict) -> pd.DataFrame:
    return pd.DataFrame(result_search)


def get_grid_search_result(search: GridSearchCV) -> dict:
    return search.cv_results_


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


def predict(
    logger: Logger,
    passenger_ids: Series,
    x_test: np.ndarray,
    best_model: SklearnModelTypes,
) -> pd.DataFrame:
    # predict
    y_pred = best_model.predict(x_test)

    # create submission data
    # 提出用データの作成
    y_pred_submission = generate_submission_dataframe(passenger_ids, y_pred)

    # 提出用データの表示
    logger.info(y_pred_submission)

    return y_pred_submission


def save_tree_graph(best_model: GradientBoostingClassifier) -> None:
    # graphviz, pydotplus使用
    dot_data = export_graphviz(
        best_model.estimators_[0, 0],
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


def save_model(
    dump_folder_name: str,
    case_id: int,
    best_model: SklearnModelTypes,
) -> None:
    # Generate save path
    save_folder_path = Path(f".\\model\\{dump_folder_name}\\case_{case_id}")
    model_file_name = Path(f"case_{case_id}.joblib")
    save_folder_path.mkdir(parents=True, exist_ok=True)
    model_save_path = save_folder_path.joinpath(model_file_name)

    # Save model information
    save_model_data(best_model, model_save_path, MODEL_SAVE_PROTOCOL)


def save_model_data(
    best_model: SklearnModelTypes,
    model_save_path: Path,
    protocol: int,
) -> None:
    joblib.dump(best_model, model_save_path, protocol=protocol)
