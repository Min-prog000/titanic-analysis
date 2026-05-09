"""Utility module for `main.py`"""

from logging import Logger

from titanic_analysis.application.analysis import analyze
from titanic_analysis.application.prediction import predict
from titanic_analysis.application.train.sklearn_model import train_sklearn_model
from titanic_analysis.application.train.torch_model import train_neural_network
from titanic_analysis.infrastructure.user.constants import ExecutionMode, TrainMethod
from titanic_analysis.interface.log.logger import TitanicLogger
from titanic_analysis.interface.log.utils import generate_log_file_path

__all__ = [
    "generate_logger",
    "get_mode_handlers",
    "log_execution_mode",
]


def generate_logger() -> Logger:
    """Generate logger and set file handler.

    Returns:
        Logger: Logger after setting.
    """
    log_file_path = generate_log_file_path()
    titanic_logger = TitanicLogger(
        logger_name="titanic",
        log_file_name=log_file_path,
    )

    return titanic_logger.logger


def log_execution_mode(logger: Logger, execution_mode: int) -> None:
    """Log execution mode.

    Args:
        logger (Logger): Logger used to inform execution mode.
        execution_mode (int): Execution mode user directed.
    """
    mode_name = ExecutionMode(execution_mode)
    logger.info("Execution mode: %s", mode_name.name)


def get_mode_handlers(
    logger: Logger,
    execution_mode: int,
    train_method: int,
    model_path: str,
) -> dict:
    """Return execution mode handler.

    Args:
        logger (Logger): Logger.
        execution_mode (int): Execution mode ID.
        train_method (int): Train method ID.
        model_path (str): Model used to predict.

    Returns:
        dict: Handlers dictionary correlated `ExecutionMode` value and function.
    """
    return {
        ExecutionMode.ANALYSIS.value: lambda: analyze(logger),
        ExecutionMode.TRAIN.value: lambda: dispatch_training_model(
            logger,
            execution_mode,
            train_method,
        ),
        ExecutionMode.PREDICT.value: lambda: predict(logger, model_path),
    }


def dispatch_training_model(
    logger: Logger,
    execution_mode: int,
    train_method: int,
) -> None:
    """Dispatch training method.

    Args:
        logger (Logger): Logger.
        execution_mode (int): Execution mode ID.
        train_method (int): Training method ID.
    """
    sklearn_methods = {
        TrainMethod.LOGISTIC_REGRESSION.value,
        TrainMethod.GRADIENT_BOOSTING.value,
    }

    if train_method in sklearn_methods:
        train_sklearn_model(logger, execution_mode)
    elif train_method == TrainMethod.NEURAL_NETWORK.value:
        train_neural_network(logger)
    else:
        logger.warning("Invalid train method.")
