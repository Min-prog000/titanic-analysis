from torch import Tensor, nn


class NeuralNetwork(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            # nn.Linear(in_features=8, out_features=1),
            nn.Linear(in_features=8, out_features=2),
            # nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs: Tensor = self.linear_relu_stack(x)

        return outputs
