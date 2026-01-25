"""データセットの表示設定に関するユーティリティモジュール"""

import pandas as pd


def set_display_options(max_rows: int, max_columns: int) -> None:
    """Pandasの表示設定を行う

    Args:
        max_rows (int): 表示する最大行数
        max_columns (int): 表示する最大列数
    """
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_columns)
