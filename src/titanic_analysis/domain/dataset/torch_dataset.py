"""Pytorchのモデルのためのデータセットクラス"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TitanicTorchDataset(Dataset):
    """PyTorch用のデータセットクラス

    Args:
        Dataset : Pytorchのデータセットクラス
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """コンストラクタ

        Args:
            data (np.ndarray): 訓練データ
            labels (np.ndarray): 訓練ラベル
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """データサイズを取得する

        Returns:
            int: データサイズ
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """特定のデータを取得する

        Args:
            idx (int): 取得したいデータのインデックス

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 訓練データとラベルのタプル
        """
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return data, label
