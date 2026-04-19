"""Main module for the titanic dataset analysis codes."""

from titanic_analysis.application.service import (
    analyze,
    predict,
    run_training_gradient_boosting,
    run_training_logistic_regression,
    run_training_neural_network,
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

    exec_mode = ExecutionMode

    mode_name = exec_mode(mode)

    logger.info("Execution mode: %s", mode_name.name)

    if mode == exec_mode.ANALYSIS.value:
        analyze(logger)
    elif mode == exec_mode.LOGISTIC_REGRESSION.value:
        run_training_logistic_regression(logger)
    elif mode == exec_mode.GRADIENT_BOOSTING.value:
        run_training_gradient_boosting(logger)
    elif mode == exec_mode.NEURAL_NETWORK.value:
        run_training_neural_network(logger)
    elif mode == exec_mode.PREDICT.value:
        predict(logger, model_path)
    else:
        logger.warning("Invalid mode inputted.")


if __name__ == "__main__":
    main()
