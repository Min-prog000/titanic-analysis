"""Pytorchのニューラルネットワーク用モジュール"""

from torch import Tensor, nn


class NeuralNetwork(nn.Module):
    """ニューラルネットワークのモデル

    Args:
        nn.Module : 全てのニューラルネットワークのためのクラス
    """

    def __init__(self, feature_size: int) -> None:
        """モデルの初期化を行う

        Args:
            feature_size (int): 訓練データの列数
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=16),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            # nn.Linear(in_features=16, out_features=32),
            # nn.BatchNorm1d(num_features=32),
            # nn.ReLU(),
            # nn.Linear(in_features=32, out_features=64),
            # nn.BatchNorm1d(num_features=64),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=128),
            # nn.BatchNorm1d(num_features=128),
            # nn.ReLU(),
            # nn.Dropout(p=0.3),
            # nn.Linear(in_features=128, out_features=64),
            # nn.BatchNorm1d(num_features=64),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=32),
            # nn.BatchNorm1d(num_features=32),
            # nn.ReLU(),
            # nn.Linear(in_features=32, out_features=16),
            # nn.BatchNorm1d(num_features=16),
            # nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features=16, out_features=8),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
            # nn.Sigmoid(),
        )

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(in_features=feature_size, out_features=8),
        #     nn.Linear(in_features=8, out_features=1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x: Tensor) -> Tensor:
        """Train model

        Args:
            x (Tensor): Data

        Returns:
            Tensor: predict output
        """
        outputs: Tensor = self.linear_relu_stack(x)

        return outputs
