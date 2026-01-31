"""ファイル入出力用ユーティリティークラス定義"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


class DisplayUtility:
    """データ表示用のユーティリティクラス"""

    @staticmethod
    def output_divider(title: str) -> None:
        """区切り線を出力する

        Args:
            title (str): 区切り線の中央に表示する文字列

        """
        print(f"-------------------- {title} --------------------")


class CsvUtility:
    """提出用CSVファイルのユーティリティークラス"""

    # csv出力
    @classmethod
    def output_csv(cls, df: pd.DataFrame, postfix_method_name: str) -> None:
        """データフレームをcsvファイルに出力する

        Args:
            df (pd.DataFrame): csvファイルにするデータフレーム
            postfix_method_name (str): csvファイルの接尾辞に使用する手法名

        """
        jst = timezone(timedelta(hours=+9), "JST")
        train_datetime = datetime.now(jst)

        # 保存先フォルダ名の接尾辞（日付）
        save_folder_name = cls._generate_save_folder_name(train_datetime)
        save_folder_path = Path(save_folder_name)

        # 保存先フォルダの作成
        save_folder_path.mkdir(parents=True, exist_ok=True)

        # 保存ファイル名の接尾辞（日付と日時）
        save_file_name = cls._generate_save_file_name(
            postfix_method_name,
            train_datetime,
        )

        # 保存ファイルのパス（カレントディレクトリの直下に作成する）
        save_path = cls._generate_save_path(save_folder_name, save_file_name)

        df.to_csv(save_path, index=False)

    @classmethod
    def _generate_save_folder_name(cls, train_datetime: datetime) -> str:
        """ファイル保存先フォルダパスを生成する

        Args:
            train_datetime (datetime): フォルダの作成日時

        Returns:
            str: ファイル保存先フォルダパス

        """
        postfix_save_folder_name = train_datetime.strftime("%Y%m%d")

        # 保存先フォルダ名
        return f"output/{postfix_save_folder_name}"

    @classmethod
    def _generate_save_file_name(
        cls,
        postfix_method_name: str,
        train_datetime: datetime,
    ) -> str:
        """保存ファイル名を生成する

        Args:
            postfix_method_name (str): ファイル名の末尾につける学習手法の名前
            train_datetime (datetime): ファイルの作成日時

        Returns:
            str: 保存ファイル名

        """
        postfix_datetime = train_datetime.strftime("%Y%m%d%H%M%S")

        # 保存ファイル名
        return f"submission_{postfix_method_name}_{postfix_datetime}.csv"

    @classmethod
    def _generate_save_path(cls, save_folder_name: str, save_file_name: str) -> str:
        """ファイル保存先パスを生成する

        Args:
            save_folder_name (str): ファイル保存先フォルダ名
            save_file_name (str): 保存ファイル名

        Returns:
            str: ファイル保存先パス

        """
        return f"{save_folder_name}/{save_file_name}"
