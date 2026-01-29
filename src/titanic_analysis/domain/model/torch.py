from torch import Tensor, nn


class NeuralNetwork(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.linear_relu_stack(x)

        return logits  # noqa: RET504
