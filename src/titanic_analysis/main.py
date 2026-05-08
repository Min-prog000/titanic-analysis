"""Main module for the titanic dataset analysis codes."""

from titanic_analysis.application.analysis import analyze
from titanic_analysis.application.prediction import predict
from titanic_analysis.infrastructure.user.constants import ExecutionMode
from titanic_analysis.infrastructure.user.parser import generate_parser
from titanic_analysis.interface.log.logger import TitanicLogger
from titanic_analysis.interface.log.utils import generate_log_file_path
from titanic_analysis.interface.utils import train_dispatcher


def main() -> None:
    """Main entry point of the application."""
    # logger setting
    log_file_path = generate_log_file_path()
    titanic_logger = TitanicLogger(
        logger_name="titanic",
        log_file_name=log_file_path,
    )

    logger = titanic_logger.logger

    # get arguments
    parser = generate_parser()
    args = parser.parse_args()

    execution_mode: int = args.execution_mode
    train_method: int = args.train_method
    model_path: str = args.model_path

    mode_name = ExecutionMode(execution_mode)
    logger.info("Execution mode: %s", mode_name.name)

    # strategy choice
    mode_handlers = {
        ExecutionMode.ANALYSIS.value: lambda: analyze(logger),
        ExecutionMode.TRAIN.value: lambda: train_dispatcher(
            logger,
            train_method,
            execution_mode,
        ),
        ExecutionMode.PREDICT.value: lambda: predict(logger, model_path),
    }

    handler = mode_handlers.get(execution_mode)

    if handler:
        handler()
    else:
        logger.warning("Invalid mode inputted.")


if __name__ == "__main__":
    main()
