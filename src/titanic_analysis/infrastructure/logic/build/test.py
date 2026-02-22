import torch

from ....domain.model.torch import NeuralNetwork


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
        threshold = 0.5
        pred = int(outputs >= threshold)

        # BCELoss
        # threshold = 0.5
        # pred = int(outputs >= threshold)

        # CrossEntropyLoss
        # pred = int(torch.argmax(outputs))

        pred_list.append(pred)

    return pred_list