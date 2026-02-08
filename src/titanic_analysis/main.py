"""Main module for the titanic dataset analysis codes."""

from titanic_analysis.application.service import (
    analyze,
    infer,
    run_torch_training_pipeline,
    run_training_pipeline,
)
from titanic_analysis.infrastructure.user.constants import (
    ANALYSIS,
    INFERENCE,
    TRAINING_PYTORCH,
    TRAINING_SKLEARN,
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
        logger_name=__name__,
        log_file_name=log_file_path,
    )

    logger = titanic_logger.logger
    mode: int = args.mode
    model_path: str = args.model_path

    logger.info("Execution mode: %s", mode)

    if mode == ANALYSIS:
        analyze()
    elif mode == TRAINING_SKLEARN:
        run_training_pipeline(logger)
    elif mode == TRAINING_PYTORCH:
        run_torch_training_pipeline(logger)
    elif mode == INFERENCE:
        infer(logger, model_path)
    else:
        logger.warning("Invalid mode inputted.")


if __name__ == "__main__":
    main()
