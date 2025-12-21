"""コマンドライン引数生成用関数のモジュール"""

import argparse

from titanic_analysis.infrastructure.user.constants import ANALYSIS


def generate_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを生成する関数

    Returns:
        argparse.ArgumentParser: コマンドライン引数パーサー
    """
    parser = argparse.ArgumentParser(
        description="Analysis method definition for titanic dataset analysis",
    )

    parser.add_argument(
        "--mode",
        type=int,
        default=ANALYSIS,
        help="Type of the execution mode (default: 0, meaning analysis mode).",
    )

    return parser
