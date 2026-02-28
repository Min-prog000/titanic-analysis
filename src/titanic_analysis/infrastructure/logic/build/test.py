import torch

from titanic_analysis.domain.model.torch import NeuralNetwork
from titanic_analysis.infrastructure.logic.build.constants import THRESHOLD


@torch.no_grad()
def test_loop(
    x_train_tensor: torch.Tensor,
    model: NeuralNetwork,
) -> list[int]:
    pred_list = []

    model.eval()
    for x in x_train_tensor:
        outputs = model(x)

        # BCEWithLogitsLoss
        pred = int(outputs >= THRESHOLD)

        # BCELoss
        # threshold = 0.5
        # pred = int(outputs >= threshold)

        # CrossEntropyLoss
        # pred = int(torch.argmax(outputs))

        pred_list.append(pred)

    return pred_list