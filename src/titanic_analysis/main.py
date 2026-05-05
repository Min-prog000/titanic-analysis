"""Main module for the titanic dataset analysis codes."""

from titanic_analysis.application.analysis import analyze
from titanic_analysis.application.prediction import predict
from titanic_analysis.application.train.sklearn_model import train_sklearn_model
from titanic_analysis.application.train.torch_model import train_neural_network
from titanic_analysis.infrastructure.user.constants import ExecutionMode, TrainMethod
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

    execution_mode: int = args.execution_mode
    train_method: int = args.train_method
    model_path: str = args.model_path

    mode_name = ExecutionMode(execution_mode)
    logger.info("Execution mode: %s", mode_name.name)

    is_analysis = execution_mode == ExecutionMode.ANALYSIS.value
    is_train = execution_mode == ExecutionMode.TRAIN.value
    is_prediction = execution_mode == ExecutionMode.PREDICT.value

    if is_analysis:
        analyze(logger)
    elif is_train:
        if train_method in (
            TrainMethod.LOGISTIC_REGRESSION.value,
            TrainMethod.GRADIENT_BOOSTING.value,
        ):
            train_sklearn_model(logger, execution_mode)
        elif train_method == TrainMethod.NEURAL_NETWORK.value:
            train_neural_network(logger)
    elif is_prediction:
        predict(logger, model_path)
    else:
        logger.warning("Invalid mode inputted.")


if __name__ == "__main__":
    main()
