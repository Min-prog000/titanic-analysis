"""データセット解析用ユーティリティークラス定義"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar

from titanic_analysis.domain.dataset.sklearn_dataset import Dataset
from titanic_analysis.infrastructure.io.utils import DisplayUtility

if TYPE_CHECKING:
    from logging import Logger

    import pandas as pd

T = TypeVar("T", bound=Dataset)


class DatasetAnalyzer[T: Dataset]:
    """データセット確認用ユーティリティークラス"""

    __all__: ClassVar[list[str]] = ["display_summary", "display_categorized_columns"]

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def display_summary(self, dataset: T) -> None:
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
        self._display_data(dataset.x)
        self._display_statistics(dataset.x)
        self._display_is_nan(dataset.x)
        self._display_nan_sum(dataset.x)

    def _display_data(self, features: pd.DataFrame) -> None:
        """データフレームのデータを表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        DisplayUtility.output_divider("Data")
        self.logger.info("\n%s", features)

    def _display_statistics(self, features: pd.DataFrame) -> None:
        """データフレームの各列の統計量を表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        DisplayUtility.output_divider("Statistics")
        self.logger.info("\n%s", features.describe())

    def _display_is_nan(self, features: pd.DataFrame) -> None:
        """データフレームのデータのうち欠損値のみTrue、それ以外をFalseで表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        DisplayUtility.output_divider("Is NaN")
        self.logger.info("\n%s", features.isna())

    def _display_nan_sum(self, features: pd.DataFrame) -> None:
        """データフレームの各列の欠損値の合計を表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        DisplayUtility.output_divider("Sum of NaN")
        self.logger.info("\n%s", self._calculate_nan_sum(features))

    def _calculate_nan_sum(self, features: pd.DataFrame) -> pd.Series:
        """データフレームの各列の欠損値の合計を計算する

        Args:
            features (pd.DataFrame): 計算対象のデータフレーム

        Returns:
            pd.Series[int]: 各列の欠損値の合計

        """
        return features.isna().sum()

    def display_categorized_columns(self, features: pd.DataFrame) -> None:
        """データフレームの各列のユニークなデータとその数をdisplayメソッドで表示する

        Args:
            features (pd.DataFrame): 表示対象のデータフレーム

        """
        columns = features.columns
        for column in columns:
            df_categorized = self._categorize(features=features, column=column)
            self.logger.info("\n%s", df_categorized)

    def _categorize(self, features: pd.DataFrame, column: str) -> pd.Series:
        """データフレームの列をユニークなデータごとにグルーピングする

        Args:
            features (pd.DataFrame): グルーピング対象のデータフレーム
            column (str): グルーピング対象の列

        Returns:
            pd.Series: グルーピング後の一次元データ

        """
        df_groupby = features.groupby(column)

        return df_groupby.size()
