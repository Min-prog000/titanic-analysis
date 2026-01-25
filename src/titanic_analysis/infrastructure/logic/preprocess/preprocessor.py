"""データセットの前処理用ユーティリティークラス定義"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum, auto
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from logging import Logger

    import numpy as np


class NanFillMethod(Enum):
    """欠損値の置換方法の識別用Enumクラス

    Args:
        Enum: 継承するクラス

    """

    MEAN = auto()
    MEDIAN = auto()
    MODE = auto()


class DatasetPreprocessor:
    """前処理用ユーティリティークラス"""

    @classmethod
    def preprocess_dataset(
        cls,
        dataset: pd.DataFrame,
        selected_columns: list[str],
        encode_columns: list[str],
        logger: Logger,
    ) -> pd.DataFrame:
        """データセットの前処理を実行するクラス

        1. 特徴量の抽出
        2. カテゴリ変数のエンコード
        3. 欠損値の置換

        Args:
            dataset (pd.DataFrame): 処理対象のデータフレーム
            selected_columns (List[str]): 抽出対象の列
            encode_columns (List[str]): エンコード対象の列
            logger (Logger): ロガー

        Returns:
            pd.DataFrame: 処理後のデータフレーム

        """
        # 1. 特徴量の抽出
        dataset_selected = cls.select_features(dataset, selected_columns)

        # 2. カテゴリ変数のエンコード
        dataset_encoded = cls.encode_by_one_hot(dataset_selected, encode_columns)

        # 3. 欠損値の置換（平均値）
        return cls._fill_nan(
            df=dataset_encoded,
            nan_fill_method=NanFillMethod.MEAN,
            round_figure=1,
            logger=logger,
        )

    @classmethod
    def select_features(
        cls,
        dataset: pd.DataFrame,
        selected_columns: list[str],
    ) -> pd.DataFrame:
        """データセットから必要な列を抽出する

        Args:
            dataset (pd.DataFrame): 抽出対象のデータフレーム
            selected_columns (List[str]): 抽出する列

        Returns:
            pd.DataFrame: 抽出後のデータフレーム

        """
        return dataset.loc[:, selected_columns]

    # ワンホットエンコーディング
    @classmethod
    def encode_by_one_hot(
        cls,
        df: pd.DataFrame,
        encoding_column_list: list[str],
    ) -> pd.DataFrame:
        """データフレームにワンホットエンコーディングを実行する

        Args:
            df (pd.DataFrame): エンコード対象のデータフレーム
            encoding_column_list (List[str]): エンコード対象の列

        Returns:
            pd.DataFrame: エンコード後のデータフレーム

        """
        df_encoded = deepcopy(df)
        for column in encoding_column_list:
            # エンコードしたデータフレームの取得
            df_dummy = pd.get_dummies(df[column], dtype=int, prefix=column)

            # 挿入位置の取得（エンコードする列の番号）
            insert_location = df_encoded.columns.get_loc(column)

            # 列の削除
            df_encoded = df_encoded.drop(column, axis=1)

            # データフレームの挿入
            df_encoded = cls.insert_dataframe(df_encoded, df_dummy, insert_location)

        return df_encoded

    @classmethod
    def insert_dataframe(
        cls,
        df_base: pd.DataFrame,
        df_insert: pd.DataFrame,
        insert_location: int | slice | np.ndarray,
    ) -> pd.DataFrame:
        """特定のデータフレームを別のデータフレームに挿入する

        Args:
            df_base (pd.DataFrame): 挿入先のデータフレーム
            df_insert (pd.DataFrame): 挿入対象のデータフレーム
            insert_location (Union[int, slice, np.ndarray]): 挿入位置

        Returns:
            pd.DataFrame: 挿入後のデータフレーム

        """
        # 同名の列があった場合はdf_baseを返す
        if not set(df_base.columns).isdisjoint(df_insert.columns):
            print("There is a same column between df_base and df_insert.")
            return df_base

        # 挿入位置から左のデータフレーム
        df_divided_left = df_base.iloc[:, :insert_location]

        # 挿入位置から右のデータフレーム
        df_divided_right = df_base.iloc[:, insert_location:]

        # 結合したデータフレーム
        return pd.concat([df_divided_left, df_insert, df_divided_right], axis=1)

    @classmethod
    def _fill_nan(
        cls,
        df: pd.DataFrame,
        nan_fill_method: NanFillMethod,
        round_figure: int,
        logger: Logger,
    ) -> pd.DataFrame:
        """欠損値（NaN）を置換する

        Args:
            df (pd.DataFrame): 欠損値のあるデータフレーム
            nan_fill_method (NanFillMethod): 欠損値の置換方法（平均値のみ有効）
            round_figure (int): 置換に使用する値の有効桁数（小数点以下）
            logger (Logger): エラー出力用のロガー

        Returns:
            pd.DataFrame:
            置換後のデータフレーム
            nan_fill_methodが予期しない値の場合はdfをそのまま返す

        """
        if nan_fill_method == NanFillMethod.MEAN:
            fill_values = df.mean(numeric_only=True)
            fill_values_round = round(fill_values, round_figure)

            return df.fillna(fill_values_round)

        logger.warning(
            msg="Incorrect method inputted: please choose from class NanFillMethod.",
        )

        return df
