"""Main module for the titanic dataset analysis codes."""

from titanic_analysis.application.service import (
    analyze,
    predict,
    run_training_gradient_boosting,
    run_training_logistic_regression,
    run_training_pipeline_pytorch,
)
from titanic_analysis.infrastructure.user.constants import (
    ANALYSIS,
    GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
    PREDICT,
    PYTORCH,
)
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

    logger.info("Execution mode: %s", mode)

    if mode == ANALYSIS:
        analyze(logger)
    elif mode == LOGISTIC_REGRESSION:
        run_training_logistic_regression(logger)
    elif mode == GRADIENT_BOOSTING:
        run_training_gradient_boosting(logger)
    elif mode == PYTORCH:
        run_training_pipeline_pytorch(logger)
    elif mode == PREDICT:
        predict(logger, model_path)
    else:
        logger.warning("Invalid mode inputted.")


if __name__ == "__main__":
    main()
