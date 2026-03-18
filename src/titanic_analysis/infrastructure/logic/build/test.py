from itertools import chain

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from titanic_analysis.domain.model.torch import NeuralNetwork
from titanic_analysis.infrastructure.logic.build.constants import THRESHOLD


@torch.no_grad()
def test_loop(
    test_dataloader: DataLoader,
    model: NeuralNetwork,
) -> list[int]:
    pred_list = []

    model.eval()
    for x, _ in test_dataloader:
        outputs: Tensor = model(x)

        # BCEWithLogitsLoss
        pred = outputs >= THRESHOLD
        print(pred)
        print(pred.shape)


        # BCELoss
        # threshold = 0.5
        # pred = int(outputs >= threshold)

        # CrossEntropyLoss
        # pred = int(torch.argmax(outputs))

        print(list(chain.from_iterable(pred.cpu().numpy().astype(int))))
        pred_list.extend(list(chain.from_iterable(pred.cpu().numpy().astype(int))))

        # print(pred.cpu().numpy().astype(int))
        # pred_list.extend(pred.cpu().numpy().astype(int))

    return pred_list