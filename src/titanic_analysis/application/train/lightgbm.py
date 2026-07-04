"""Training use case using sklearn"""

from logging import Logger
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from graphviz import Digraph
from pandas import Series
from yaml import safe_dump

from titanic_analysis.application.constants import (
    CASE_ID_PATH,
    LIGHTGBM_CONFIG_PATH,
    LIGHTGBM_TREE_PATH,
    TREE_RENDER_FORMAT,
    UTF_8,
    WRITE_ONLY,
)
from titanic_analysis.application.train.utils import (
    create_dataset,
    generate_config_path,
    generate_model_save_path,
    generate_submission_dataframe,
    generate_tree_save_path,
    get_case_id,
    save_case_id,
)
from titanic_analysis.infrastructure.io.analysis.config_loader import (
    load_lightgbm_config,
)
from titanic_analysis.infrastructure.io.constants import (
    CONFIG_FILE_PREFIX_LIGHTGBM,
    LIGHTGBM,
    PATH_TEST,
    PATH_TRAIN,
    SAVE_MODEL_FILE_EXTENSION_LIGHTGBM,
    SAVE_MODEL_FILE_PARENT_LIGHTGBM,
    SAVE_MODEL_FILE_PREFIX_LIGHTGBM,
    SAVE_MODEL_ROOT_LIGHTGBM,
    SAVE_TREE_INDEX,
)
from titanic_analysis.infrastructure.io.utils import CsvUtility

__all__ = ["train_lightgbm_model"]


def train_lightgbm_model(
    logger: Logger,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """Train using lightgbm model.

    This function performs preprocess, training, and generate submission csv
        using LGBMClassifier

    Args:
        logger (Logger): Logger generated in `main`.
        train_dataset_path (str, optional): Dataset path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional): Dataset path. Defaults to PATH_TEST.

    Raises:
        FalseComponentError: Raise when missing columns.
    """
    # Create dataset
    x_train, y_train, x_test, passenger_ids = create_dataset(
        logger,
        train_dataset_path,
        test_dataset_path,
    )

    # Train
    config_path = Path(LIGHTGBM_CONFIG_PATH)
    parameters = load_lightgbm_config(config_path)
    model = train(x_train, y_train, parameters)

    logger.debug("params:\n%s", model.get_params())

    # Predict
    y_pred = predict(logger, passenger_ids, x_test, model)

    # Output submission file
    CsvUtility.output_csv(y_pred, LIGHTGBM)

    # Save model
    save_artifacts(parameters, model)


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    parameters: dict,
) -> lgb.LGBMClassifier:
    # Model setting
    model = lgb.LGBMClassifier(**parameters)

    # Training
    model.fit(
        x_train,
        y_train,
        # NOTE: early_stoppingを使うには検証データが必要
        # callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)],
    )

    return model


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
    model: lgb.LGBMClassifier,
) -> pd.DataFrame:
    # Predict
    y_pred = model.predict(x_test)

    # Generate submission dataframe
    if isinstance(y_pred, np.ndarray):
        y_pred_submission = generate_submission_dataframe(passenger_ids, y_pred)
        # Log submission dataframe
        logger.info("\n%s", y_pred_submission)
        return y_pred_submission

    msg = f"model.predict() must return np.ndarray, got {type(y_pred)}"
    raise TypeError(msg)


def save_artifacts(parameters: dict, model: lgb.LGBMClassifier) -> None:
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


def save_tree(model: lgb.LGBMClassifier, case_id: int) -> None:
    # Get tree data
    graph = get_tree_data(model, SAVE_TREE_INDEX)  # Initial tree (index 0)

    # Save as "PNG"
    tree_to_image(case_id, graph, SAVE_TREE_INDEX, TREE_RENDER_FORMAT)


def tree_to_image(
    case_id: int,
    graph: Digraph,
    save_tree_index: int,
    render_format: str,
) -> None:
    graph_folder_path, graph_file_path = generate_tree_save_path(
        LIGHTGBM_TREE_PATH,
        case_id,
        save_tree_index,
    )

    graph_folder_path.mkdir(parents=True, exist_ok=True)

    graph.render(graph_file_path, cleanup=True, format=render_format)


def get_tree_data(model: lgb.LGBMClassifier, index: int) -> Digraph:
    return lgb.create_tree_digraph(model, tree_index=index, format="png")


def save_config(parameters: dict, case_id: int) -> None:
    # Generate path
    config_folder_path, config_file_path = generate_config_path(
        case_id,
        CONFIG_FILE_PREFIX_LIGHTGBM,
    )

    # Make parent directory
    config_folder_path.mkdir(parents=True, exist_ok=True)

    # Output config
    with config_file_path.open(mode=WRITE_ONLY, encoding=UTF_8) as f:
        safe_dump(parameters, f, sort_keys=False)


def save_model(model: lgb.LGBMClassifier, case_id: int) -> None:
    # Generate path
    path_parts = {
        "folder_root_name": SAVE_MODEL_ROOT_LIGHTGBM,
        "model_code": LIGHTGBM,
        "folder_parent_name": SAVE_MODEL_FILE_PARENT_LIGHTGBM,
        "file_prefix": SAVE_MODEL_FILE_PREFIX_LIGHTGBM,
        "file_extension": SAVE_MODEL_FILE_EXTENSION_LIGHTGBM,
    }
    save_folder_path, save_file_path = generate_model_save_path(
        case_id,
        path_parts,
    )

    # Make parent directory
    save_folder_path.mkdir(parents=True, exist_ok=True)

    # Save model
    # NOTE: LightGBM has save method in default.
    joblib.dump(model, save_file_path, compress=3)
