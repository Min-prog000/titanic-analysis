"""コマンドライン引数生成用関数のモジュール"""

import argparse

from titanic_analysis.infrastructure.user.constants import ExecutionMode


def generate_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを生成する関数

    Returns:
        argparse.ArgumentParser: コマンドライン引数パーサー
    """
    exec_mode = ExecutionMode

    parser = argparse.ArgumentParser(
        description="Analysis method definition for titanic dataset analysis",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        default=exec_mode.ANALYSIS.value,
        choices=[
            exec_mode.ANALYSIS.value,
            exec_mode.LOGISTIC_REGRESSION.value,
            exec_mode.GRADIENT_BOOSTING.value,
            exec_mode.NEURAL_NETWORK.value,
            exec_mode.PREDICT.value,
        ],
        help="Type of the execution mode (default: 0, meaning analysis mode).",
    )

    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        default=None,
        help="Model path (ONNX)",
    )

    return parser
