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
        choices=[execution_mode.name for execution_mode in ExecutionMode],
        help="Type of the execution mode (default: 0, meaning ANALYSIS).",
    )

    parser.add_argument(
        "-t",
        "--train_method",
        type=int,
        default=TrainMethod.NEURAL_NETWORK.value,
        choices=[train_method.name for train_method in TrainMethod],
        help="Type of the training method (default: 4, meaning NEURAL_NETWORK).",
    )

    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        default=None,
        help="Model path (ONNX)",
    )

    return parser
