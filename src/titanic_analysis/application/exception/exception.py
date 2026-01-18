"""datasetモジュール用例外クラス定義"""


class FalseComponentError(Exception):
    """二つの配列の行列数が異なる場合に呼び出す例外クラス

    Args:
        Exception: 継承する例外クラス

    """

    def __init__(self, msg: str) -> None:
        """コンストラクタ

        Args:
            msg (str): 例外発生時に出力する文字列

        """
        self.msg = msg

    def __str__(self) -> str:
        """文字列を出力する場合に呼び出される関数

        Args:
            msg (str): 出力する例外の内容

        Returns:
            str: 出力する例外の内容

        """
        return self.msg
