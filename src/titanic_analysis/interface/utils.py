from logging import Logger

from titanic_analysis.application.train.sklearn_model import train_sklearn_model
from titanic_analysis.application.train.torch_model import train_neural_network
from titanic_analysis.infrastructure.user.constants import TrainMethod


def train_dispatcher(logger: Logger, train_method: int, execution_mode: int) -> None:
    """Dispatch training method.

    Args:
        logger (Logger): Logger.
        train_method (int): Training method ID.
        execution_mode (int): Execution mode ID.
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
