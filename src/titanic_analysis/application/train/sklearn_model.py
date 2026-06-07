"""Training use case using sklearn"""

from logging import Logger
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydotplus
from pandas import Series
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz

from titanic_analysis.application.constants import (
    CASE_ID_PATH,
    NOT_PIPELINE_INSTANCE_MESSAGE,
)
from titanic_analysis.application.train.strategy import ModelStrategy
from titanic_analysis.application.train.utils import (
    create_dataset,
    generate_submission_dataframe,
    get_case_id,
    get_strategy,
    save_case_id,
)
from titanic_analysis.domain.model.types import SklearnModelTypes
from titanic_analysis.infrastructure.io.constants import (
    MODEL_SAVE_PROTOCOL,
    PATH_TEST,
    PATH_TRAIN,
)
from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    GradientBoostingClassifierConfigDTO,
)
from titanic_analysis.infrastructure.io.utils import CsvUtility

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
    # Get strategy
    strategy = get_strategy(method_id)

    # Load data
    x_train, y_train, x_test, passenger_ids = create_dataset(
        logger,
        train_dataset_path,
        test_dataset_path,
    )

    # Train model
    best_model = run_grid_search(
        logger,
        strategy,
        x_train,
        y_train,
    )

    # Predict
    y_pred = predict(logger, passenger_ids, x_test, best_model)

    # Generate submission file
    csv_postfix = strategy.get_csv_postfix()
    CsvUtility.output_csv(y_pred, csv_postfix)

    # Save experiment results and case id
    save_artifacts(strategy, best_model)


def save_artifacts(strategy: ModelStrategy, best_model: SklearnModelTypes) -> None:
    """Save artifacts such as model, case id, and tree graph.

    Args:
        strategy (ModelStrategy): Method strategy.
        best_model (SklearnModelTypes): Best model in grid search.
    """
    save_folder_name = strategy.get_save_folder_name()

    # 1. Get current case id
    case_id = get_case_id(CASE_ID_PATH)

    # 2. Save tree visualization (if model is tree)
    if isinstance(best_model, GradientBoostingClassifier):
        save_tree_graph(best_model)

    # 3. Save model
    save_model(save_folder_name, case_id, best_model)

    # 4. Save next case id
    save_case_id(case_id, CASE_ID_PATH)


def run_grid_search(
    logger: Logger,
    strategy: ModelStrategy,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> SklearnModelTypes:
    """Run a grid search to find the best model for the strategy and data.

    This function loads model configuration from the given strategy, generates
    a parameter grid, constructs a preprocessing-model pipeline, and performs
    a grid search using `GridSearchCV`. After fitting, it logs the search
    results and extracts the best estimator from the fitted pipeline.

    Args:
        logger (Logger):
            Logger instance used to output grid search progress and results.
        strategy (ModelStrategy):
            Strategy object responsible for loading configuration, generating
            parameter grids, creating model instances, and providing pipeline
            prefixes.
        x_train (np.ndarray):
            Training feature matrix.
        y_train (np.ndarray):
            Training target vector.

    Returns:
        SklearnModelTypes:
            The best model selected by grid search, extracted from the fitted
            pipeline.

    Raises:
        NotImplementedError:
            Raised when the configuration for `GradientBoostingClassifier`
            specifies a `max_features` value that exceeds the number of
            available columns in `x_train`.

    Notes:
        - A MinMaxScaler is always applied before the model.
        - Grid search uses `n_jobs=2` and `verbose=10`.
        - The best model is extracted using the strategy's pipeline prefix.

    """
    # Scaler
    scaler = MinMaxScaler()

    # Parameters
    config_loaded = strategy.load_config()
    col_num = get_array_col_num(x_train)
    if (
        isinstance(config_loaded, GradientBoostingClassifierConfigDTO)
        and config_loaded.max_features["max"] + 1 > col_num
    ):
        raise NotImplementedError

    params_grid = strategy.generate_params(config_loaded)

    # Model setting
    model = strategy.create_model(config_loaded.random_state)
    pipeline = make_pipeline(scaler, model)

    # Grid search setting
    search = GridSearchCV(pipeline, params_grid, n_jobs=2, verbose=10)

    # Execute grid search
    search.fit(x_train, y_train)

    # Log grid search result
    log_grid_search_result(logger, search)
    log_best_model_info(logger, search)

    # Predict with best model
    pipeline_prefix = strategy.get_pipeline_prefix()
    best_model: SklearnModelTypes = get_search_best_model(pipeline_prefix, search)

    return best_model


def get_array_col_num(x_train: np.ndarray) -> int:
    return x_train.shape[1]


def get_search_best_model(
    pipeline_prefix: str,
    search: GridSearchCV,
) -> SklearnModelTypes:
    best_model = search.best_estimator_

    if not isinstance(best_model, Pipeline):
        raise TypeError(NOT_PIPELINE_INSTANCE_MESSAGE)

    if pipeline_prefix not in best_model.named_steps:
        keys = best_model.named_steps.keys()
        msg = f"'{pipeline_prefix}' not found in pipeline steps: {list(keys)}"
        raise KeyError(msg)

    return best_model.named_steps[pipeline_prefix]


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


def save_model(
    save_folder_name: str,
    case_id: int,
    best_model: SklearnModelTypes,
) -> None:
    # Generate save path
    save_folder_path = Path(f".\\model\\{save_folder_name}\\case_{case_id}")
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
