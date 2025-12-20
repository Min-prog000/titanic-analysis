"""データセット解析用ユーティリティークラス定義"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from titanic_analysis.domain.dataset.dataset import Dataset
from titanic_analysis.infrastructure.io.io_utils import DisplayUtility

if TYPE_CHECKING:
    import pandas as pd

T = TypeVar("T", bound=Dataset)


class DatasetAnalyzer(Generic[T]):
    """データセット確認用ユーティリティークラス"""

    __all__: ClassVar[list[str]] = ["display_summary", "display_categorized_columns"]

    @classmethod
    def display_summary(cls, dataset: T) -> None:
        """データフレームの概要を出力する

        出力内容
        - データセット本体
        - 統計量（describe()）
        - 欠損値かどうか（isnull()）
        - 各列の欠損値の合計（isnull().sum()）

        それぞれディバイダ―付きで表示する

        Arg:
            dataset (Generic[T]): 出力対象のデータフレーム

        """
        cls._display_data(dataset.x)
        cls._display_statistics(dataset.x)
        cls._display_is_nan(dataset.x)
        cls._display_nan_sum(dataset.x)

    @classmethod
    def _display_data(cls, features: pd.DataFrame) -> None:
        """データフレームのデータを表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        DisplayUtility.output_divider("Data")
        print(features)

    @classmethod
    def _display_statistics(cls, features: pd.DataFrame) -> None:
        """データフレームの各列の統計量を表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        DisplayUtility.output_divider("Statistics")
        print(features.describe())

    @classmethod
    def _display_is_nan(cls, features: pd.DataFrame) -> None:
        """データフレームのデータのうち欠損値のみTrue、それ以外をFalseで表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        DisplayUtility.output_divider("Is NaN")
        print(features.isna())

    @classmethod
    def _display_nan_sum(cls, features: pd.DataFrame) -> None:
        """データフレームの各列の欠損値の合計を表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        DisplayUtility.output_divider("Sum of NaN")
        print(cls._calculate_nan_sum(features))

    @classmethod
    def _calculate_nan_sum(cls, features: pd.DataFrame) -> pd.Series:
        """データフレームの各列の欠損値の合計を計算する

        Args:
            features (pd.DataFrame): 計算対象のデータフレーム

        Returns:
            pd.Series[int]: 各列の欠損値の合計

        """
        return features.isna().sum()

    @classmethod
    def display_categorized_columns(cls, features: pd.DataFrame) -> None:
        """データフレームの各列のユニークなデータとその数をdisplayメソッドで表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        columns = features.columns
        for column in columns:
            df_categorized = cls._categorize(features=features, column=column)
            print(df_categorized)

    @classmethod
    def _categorize(cls, features: pd.DataFrame, column: str) -> pd.Series:
        """データフレームの列をユニークなデータごとにグルーピングする

        Args:
            features (pd.DataFrame): グルーピング対象のデータフレーム
            column (str): グルーピング対象の列

        Returns:
            pd.Series: グルーピング後の一次元データ

        """
        df_groupby = features.groupby(column)

        return df_groupby.size()
