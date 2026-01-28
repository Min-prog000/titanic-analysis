import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class TitanicTorchDataset(Dataset):
    """PyTorch用のデータセットクラス

    Args:
        Dataset : Pytorch用のデータセットクラス
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.DataFrame,
        transform: None | ToTensor = None,
        target_transform: None | ToTensor = None,
    ) -> None:
        """コンストラクタ

        Args:
            data (Tensor): 訓練データ
            labels (Tensor): ラベル
            transform (None | Tensor, optional):
                訓練データの方変換方法. Defaults to None.
            target_transform (None | Tensor, optional):
                ラベルの型変換方法. Defaults to None.
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """データサイズを取得するため

        Returns:
            int: データサイズ
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """特定のデータを取得する

        Args:
            idx (int): 取得したいデータのインデックス

        Returns:
            dict: 訓練データとラベルの辞書
        """
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return {"data": data, "label": label}
