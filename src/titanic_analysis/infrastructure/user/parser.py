"""コマンドライン引数生成用関数のモジュール"""

import argparse

from titanic_analysis.infrastructure.user.constants import (
    ANALYSIS,
    GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
    PREDICT,
    PYTORCH,
)


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
        "--mode",
        type=int,
        default=ANALYSIS,
        choices=[ANALYSIS, LOGISTIC_REGRESSION, GRADIENT_BOOSTING, PYTORCH, PREDICT],
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
