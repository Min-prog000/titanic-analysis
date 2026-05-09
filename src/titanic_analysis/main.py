"""Main module for the titanic dataset analysis codes."""

from titanic_analysis.infrastructure.user.parser import generate_parser
from titanic_analysis.interface.utils import (
    generate_logger,
    get_mode_handlers,
    log_execution_mode,
)


def main() -> None:
    """Main entry point of the application."""
    # generate logger
    logger = generate_logger()

    # get arguments
    parser = generate_parser()
    args = parser.parse_args()

    execution_mode: int = args.execution_mode
    train_method: int = args.train_method
    model_path: str = args.model_path

    log_execution_mode(logger, execution_mode)

    # choose strategy
    mode_handlers = get_mode_handlers(logger, execution_mode, train_method, model_path)
    handler = mode_handlers.get(execution_mode)

    if handler:
        handler()
    else:
        logger.warning("Invalid mode inputted.")


if __name__ == "__main__":
    main()
