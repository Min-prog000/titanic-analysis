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
    BOOSTER_DUMP_FORMAT_XGBOOST,
    CASE_ID_PATH,
    ID_COLUMN,
    PIPELINE_PREFIX_XGBOOST,
    TARGET_COLUMN,
    TREE_RENDER_FORMAT_XGBOOST,
    XGBOOST_CONFIG_PATH,
    XGBOOST_TREE_PATH,
)
from titanic_analysis.application.preprocess import preprocess_load_data
from titanic_analysis.application.train.utils import (
    generate_config_path,
    generate_model_save_path,
    generate_next_case_id,
    generate_output_path,
    get_case_id,
)
from titanic_analysis.infrastructure.io.analysis.config_loader import (
    load_xgboost_config,
)
from titanic_analysis.infrastructure.io.constants import (
    PATH_TEST,
    PATH_TRAIN,
    SAVE_TREE_FILE_INDEX_PREFIX_XGBOOST,
    SAVE_TREE_FILE_PREFIX_XGBOOST,
    SAVE_TREE_INDEX,
    XGBOOST,
)
from titanic_analysis.infrastructure.io.utils import CsvUtility

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


def create_dataset(
    logger: Logger,
    train_dataset_path: str,
    test_dataset_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Series]:
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)

    # Preprocess
    # Training data
    x_train, x_test = preprocess_load_data(
        logger,
        df_train,
        df_test,
    )

    # Training label
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
    y_pred = model.predict(x_test)

    # Create submission data
    y_pred_df = pd.DataFrame(y_pred, columns=[TARGET_COLUMN])
    y_pred_df_submission = pd.concat([passenger_ids, y_pred_df], axis=1)

    # Log submission data
    logger.info(y_pred_df_submission)

    # Output submission file
    CsvUtility.output_csv(y_pred_df_submission, XGBOOST)


def save_artifacts(parameters: dict, model: xgb.XGBClassifier) -> None:
    # 1. get current case id
    case_id = get_case_id(CASE_ID_PATH)

    # 2. save tree visualization
    save_tree(model, case_id)

    # 3. save config
    save_config(parameters, case_id)

    # 4. save model
    save_model(model, case_id)

    # 5. save next case id
    save_case_id(case_id)


def save_case_id(case_id: int, file_name_path: str = CASE_ID_PATH) -> None:
    joblib.dump(generate_next_case_id(case_id), Path(file_name_path))


def save_tree(model: xgb.XGBClassifier, case_id: int) -> None:
    # Get tree data
    dot_data = get_tree_data(model, SAVE_TREE_INDEX)  # Initial tree (index 0)

    # Save as "PNG"
    tree_to_image(case_id, dot_data, SAVE_TREE_INDEX, TREE_RENDER_FORMAT_XGBOOST)


def tree_to_image(
    case_id: int,
    dot_data: str,
    save_tree_index: int,
    render_format: str,
) -> None:
    graph = Source(dot_data)
    # graph.format = "png"
    graph_path = get_tree_save_path(case_id, save_tree_index)
    graph.render(graph_path, cleanup=True, format=render_format)


def get_tree_save_path(case_id: int, save_tree_index: int) -> Path:
    folder_path = get_tree_folder_path()
    file_path = get_tree_file_path(case_id, save_tree_index)

    return generate_output_path(folder_path, file_path)


def get_tree_folder_path(xgboost_tree_path: str = XGBOOST_TREE_PATH) -> Path:
    return Path(xgboost_tree_path)


def get_tree_file_path(case_id: int, save_tree_index: int) -> Path:
    return Path(generate_tree_file_name_path(case_id, save_tree_index))


def generate_tree_file_name_path(case_id: int, save_tree_index: int) -> str:
    file_prefix = SAVE_TREE_FILE_PREFIX_XGBOOST
    index_prefix = SAVE_TREE_FILE_INDEX_PREFIX_XGBOOST

    return f"{file_prefix}{case_id}{index_prefix}{save_tree_index}"


def get_tree_data(model: xgb.XGBClassifier, index: int) -> str:
    booster = model.get_booster()
    dot_data = booster.get_dump(dump_format=BOOSTER_DUMP_FORMAT_XGBOOST)

    return dot_data[index]


def save_config(parameters: dict, case_id: int) -> None:
    # Generate path
    config_folder_path, config_file_path = generate_config_path(case_id)

    # Make parent directory
    config_folder_path.mkdir(parents=True, exist_ok=True)

    # Output config
    with config_file_path.open(mode="w", encoding="utf-8") as f:
        safe_dump(parameters, f, sort_keys=False)


def save_model(model: xgb.XGBClassifier, case_id: int) -> None:
    # Generate path
    save_folder_path, save_file_path = generate_model_save_path(case_id)

    # Make parent directory
    save_folder_path.mkdir(parents=True, exist_ok=True)

    # Save model
    # NOTE: XGBoost has save method in default.
    model.save_model(save_file_path)
