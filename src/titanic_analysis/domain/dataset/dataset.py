"""データセットクラス用モジュール"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator


class Dataset(ABC):
    """データセットクラス"""

    __slots__ = ("_x", "_y")

    _x: pd.DataFrame
    _y: pd.Series

    @property
    @abstractmethod
    def x(self) -> pd.DataFrame:
        """特徴量を返す"""

    @abstractmethod
    def __len__(self) -> int:
        """データセットの長さを返す抽象メソッド"""

    @abstractmethod
    def _load(self) -> None:
        """データセットを読み込む抽象メソッド"""


class TrainDataset(Dataset):
    """訓練データセットクラス"""

    def __init__(self, data_path: str) -> None:
        """コンストラクタ

        Args:
            data_path (str): データセットのパス
        """
        self.data_path = data_path
        self._load()

    def _load(self) -> None:
        """データセットを読み込む"""
        dataframe = pd.read_csv(self.data_path)
        self._y = dataframe["Survived"]
        self._x = dataframe.drop(columns=["Survived"])

    @property
    def x(self) -> pd.DataFrame:
        """特徴量データフレームを返す"""
        return self._x

    @property
    def y(self) -> pd.Series:
        """特徴量データフレームを返す"""
        return self._y

    @y.setter
    def y(self, label: pd.Series) -> None:
        """目的変数データフレームを設定する"""
        self._y = label

    def __len__(self) -> int:
        """データセットの長さを返す"""
        return len(self._x)

    def __getitem__(self, index: int) -> tuple[pd.Series, pd.Series]:
        """データセットのイテレータを返す"""
        return self._x.iloc[index], self._y.iloc[index]

    def __iter__(self) -> Iterator[tuple[pd.Series, pd.Series]]:
        """データセットのイテレータを返す"""
        for index in range(len(self)):
            yield self[index]


class TestDataset(Dataset):
    """テストデータセットクラス"""

    def __init__(self, data_path: str) -> None:
        """コンストラクタ

        Args:
            data_path (str): データセットのパス
        """
        self.data_path = data_path
        self._load()

    def _load(self) -> None:
        """データセットを読み込む"""
        self._x = pd.read_csv(self.data_path)

    @property
    def x(self) -> pd.DataFrame:
        """特徴量データフレームを返す"""
        return self._x

    def __len__(self) -> int:
        """データセットの長さを返す"""
        return len(self._x)

    def __getitem__(self, index: int) -> pd.Series:
        """データセットのイテレータを返す"""
        return self._x.iloc[index]

    def __iter__(self) -> Iterator[pd.Series]:
        """データセットのイテレータを返す"""
        for index in range(len(self)):
            yield self[index]
