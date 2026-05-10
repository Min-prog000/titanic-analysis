"""コマンドライン引数生成用関数のモジュール"""

import argparse

from titanic_analysis.infrastructure.user.constants import ExecutionMode, TrainMethod


def generate_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを生成する関数

    Returns:
        argparse.ArgumentParser: コマンドライン引数パーサー
    """
    parser = argparse.ArgumentParser(
        description="Analysis method definition for titanic dataset analysis",
    )

    parser.add_argument(
        "-m",
        "--execution_mode",
        type=int,
        default=ExecutionMode.ANALYSIS.value,
        choices=[
            ExecutionMode.ANALYSIS.value,
            ExecutionMode.TRAIN.value,
            ExecutionMode.PREDICT.value,
        ],
        help="Type of the execution mode (default: 0, meaning ANALYSIS).",
    )

    parser.add_argument(
        "-t",
        "--train_method",
        type=int,
        default=TrainMethod.NEURAL_NETWORK.value,
        choices=[
            TrainMethod.LOGISTIC_REGRESSION.value,
            TrainMethod.GRADIENT_BOOSTING.value,
            TrainMethod.XGBOOST.value,
            TrainMethod.NEURAL_NETWORK.value,
        ],
        help="Type of the training method (default: 2, meaning NEURAL_NETWORK).",
    )

    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        default=None,
        help="Model path (ONNX)",
    )

    return parser
