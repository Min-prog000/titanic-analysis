import numpy as np
import torch
from torch.utils.data import Dataset


class TitanicTorchDataset(Dataset):
    """PyTorch用のデータセットクラス

    Args:
        Dataset : Pytorch用のデータセットクラス
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """コンストラクタ

        Args:
            data (DataFrame): 訓練データ
            labels (Series): ラベル
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """データサイズを取得するため

        Returns:
            int: データサイズ
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """特定のデータを取得する

        Args:
            idx (int): 取得したいデータのインデックス

        Returns:
            dict: 訓練データとラベルの辞書
        """
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return data, label
