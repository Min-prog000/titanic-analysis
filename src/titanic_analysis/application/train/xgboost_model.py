"""Training use case using sklearn"""

import sys
from logging import Logger
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from graphviz import Source
from pandas import Series
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from yaml import safe_dump

from titanic_analysis.application.constants import (
    CASE_ID_PATH,
    ID_COLUMN,
    PIPELINE_PREFIX_XGBOOST,
    TARGET_COLUMN,
    XGBOOST_CONFIG_PATH,
    XGBOOST_TREE_PATH,
)
from titanic_analysis.application.preprocess import preprocess_load_data
from titanic_analysis.infrastructure.io.analysis.config_loader import (
    load_xgboost_config,
)
from titanic_analysis.infrastructure.io.constants import (
    PATH_TEST,
    PATH_TRAIN,
    XGBOOST,
)
from titanic_analysis.infrastructure.io.utils import CsvUtility
from titanic_analysis.infrastructure.logic.build.utils import load_case_id

__all__ = ["train_xgboost_model"]


def train_xgboost_model(
    logger: Logger,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """Train using xgboost model.

    This function performs preprocess, training, and generate submission csv
        using XGBoostClassifier

    Args:
        logger (Logger): Logger generated in `main`.
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
    parameters, model = train(x_train, y_train)

    # prediction(create submission file)
    predict(logger, passenger_ids, x_test, model)

    # model save
    save_artifacts(parameters, model)


def save_artifacts(parameters: dict, model: xgb.XGBClassifier) -> None:
    case_id = get_case_id(CASE_ID_PATH)

    # 1. save tree visualization
    save_tree_graph(model, case_id)

    # 2. save config
    save_config(parameters, case_id)

    # 3. save model
    save_model(model, case_id)


def create_dataset(
    logger: Logger,
    train_dataset_path: str,
    test_dataset_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Series]:
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)

    # Preprocess
    # training data
    x_train, x_test = preprocess_load_data(
        logger,
        df_train,
        df_test,
    )

    # label
    y_train = np.array(df_train.loc[:, TARGET_COLUMN])
    passenger_ids = df_test[ID_COLUMN]

    logger.info("\n%s", x_train)
    logger.info("\n%s", x_test)

    # データセットのサイズが等しいことの確認
    # TODO: Revise to be able to compare preprocessed data columns
    if not (x_train.shape and x_test.shape):
        logger.info("Not match datasets shape.")
        sys.exit()

    return x_train, y_train, x_test, passenger_ids


def train(x_train: np.ndarray, y_train: np.ndarray) -> tuple[dict, xgb.XGBClassifier]:
    # Scaler
    scaler = MinMaxScaler()

    # Model setting
    config_path = Path(XGBOOST_CONFIG_PATH)
    parameters = load_xgboost_config(config_path)
    model = xgb.XGBClassifier(**parameters)

    # Pipeline setting
    pipeline = make_pipeline(scaler, model)

    # Training
    pipeline.fit(x_train, y_train)
    model = pipeline.named_steps[PIPELINE_PREFIX_XGBOOST]

    # Return model
    return parameters, model


# def run_grid_search(
#     logger: Logger,
#     x_train: np.ndarray,
#     y_train: np.ndarray,
# ) -> tuple[str, str, SklearnModelTypes]:
#     # Scaling
#     scaler = MinMaxScaler()

#     # Parameters
#     config_path = Path(XGBOOST_CONFIG_PATH)
#     # TODO: Create config loading function for xgboost
#     config_loaded = load_xgboost_config(config_path)
#     model = xgb.XGBClassifier()
#     params_grid = {""}
#     pipeline_prefix = "xgboost"
#     csv_postfix = "xgboost"
#     dump_folder_name = "xgboost"

#     pipeline = make_pipeline(scaler, model)

#     # グリッドサーチ
#     # TODO: Update single execution
#     search = GridSearchCV(pipeline, params_grid, n_jobs=2, verbose=10)
#     search.fit(x_train, y_train)

#     # グリッドサーチ結果の表示
#     result_search = search.cv_results_
#     result_search_df = pd.DataFrame(result_search).iloc[:, 4:]
#     result_search_df_rounded = result_search_df.round(3)
#     logger.info(result_search_df_rounded)

#     # グリッドサーチのベストスコア表示
#     logger.info("Grid search best score: %s", search.best_score_)
#     logger.info("Hyper parameters: %s", search.best_params_)

#     # 最高精度のモデルによる推論
#     model_best: SklearnModelTypes = search.best_estimator_.named_steps[pipeline_prefix]

#     return csv_postfix, dump_folder_name, model_best


def predict(
    logger: Logger,
    passenger_ids: Series,
    x_test: np.ndarray,
    model: xgb.XGBClassifier,
) -> None:
    # Predict
    # TODO: Check type of `x_test` and remove type conversion using `np.array()`
    y_pred = model.predict(x_test)

    # Create submission data
    y_pred_df = pd.DataFrame(y_pred, columns=[TARGET_COLUMN])
    y_pred_df_submission = pd.concat([passenger_ids, y_pred_df], axis=1)

    # Log submission data
    logger.info(y_pred_df_submission)

    # Output submission file
    CsvUtility.output_csv(y_pred_df_submission, XGBOOST)


def get_case_id(case_id_path_str: str) -> int:
    # Get case id
    case_id_path = Path(case_id_path_str)

    return load_case_id(case_id_path)


def save_tree_graph(model: xgb.XGBClassifier, case_id: int) -> None:
    # Get tree data
    booster = model.get_booster()
    dot_data = booster.get_dump(dump_format="dot")[0]  # 0番目の木

    # Save as "PNG"
    graph = Source(dot_data)
    graph.format = "png"
    graph_path = Path(f"{XGBOOST_TREE_PATH}\\case{case_id}_tree_0")
    graph.render(graph_path, cleanup=True)


def save_config(parameters: dict, case_id: int) -> None:
    # Generate path
    yaml_output_path = Path(f".\\output\\config\\case{case_id}")
    yaml_output_path.mkdir(parents=True, exist_ok=True)
    config_file_name = Path(f"config_case{case_id}.yaml")
    config_file_path = yaml_output_path.joinpath(config_file_name)

    # Save config
    with config_file_path.open(mode="w", encoding="utf-8") as f:
        safe_dump(parameters, f, sort_keys=False)


def save_model(model: xgb.XGBClassifier, case_id: int) -> None:
    # Save model
    save_folder_path = Path(f".\\model\\{XGBOOST}\\case_{case_id}")
    model_file_name = Path(f"case_{case_id}.json")
    save_folder_path.mkdir(parents=True, exist_ok=True)
    save_file_path = save_folder_path.joinpath(model_file_name)
    model.save_model(save_file_path)

    # Save next case id
    joblib.dump(case_id + 1, CASE_ID_PATH)
