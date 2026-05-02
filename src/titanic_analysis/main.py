"""Main module for the titanic dataset analysis codes."""

from titanic_analysis.application.analysis import analyze
from titanic_analysis.application.prediction import predict
from titanic_analysis.application.train import (
    train_neural_network,
    train_sklearn_model,
)
from titanic_analysis.infrastructure.user.constants import ExecutionMode
from titanic_analysis.infrastructure.user.parser import generate_parser
from titanic_analysis.interface.log.logger import TitanicLogger
from titanic_analysis.interface.log.utils import generate_log_file_path


def main() -> None:
    """Main entry point of the application."""
    parser = generate_parser()
    args = parser.parse_args()

    log_file_path = generate_log_file_path()
    titanic_logger = TitanicLogger(
        logger_name="titanic",
        log_file_name=log_file_path,
    )

    logger = titanic_logger.logger

    mode: int = args.mode
    model_path: str = args.model_path

    mode_name = ExecutionMode(mode)
    logger.info("Execution mode: %s", mode_name.name)

    if mode == ExecutionMode.ANALYSIS.value:
        analyze(logger)
    elif mode in (
        ExecutionMode.LOGISTIC_REGRESSION.value,
        ExecutionMode.GRADIENT_BOOSTING.value,
    ):
        train_sklearn_model(logger, mode)
    elif mode == ExecutionMode.NEURAL_NETWORK.value:
        train_neural_network(logger)
    elif mode == ExecutionMode.PREDICT.value:
        predict(logger, model_path)
    else:
        logger.warning("Invalid mode inputted.")


if __name__ == "__main__":
    main()
