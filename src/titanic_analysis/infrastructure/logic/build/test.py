# fromとimportの間に書けるのはモジュール名だけ
# -> chainはimportの後に書く
from itertools import chain

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from titanic_analysis.domain.model.torch import NeuralNetwork
from titanic_analysis.infrastructure.logic.build.constants import THRESHOLD

__all__ = ["test_loop"]

@torch.no_grad()
def test_loop(
    test_dataloader: DataLoader,
    model: NeuralNetwork,
) -> list[int]:
    pred_list = []

    model.eval()
    for x, _ in test_dataloader:
        outputs: Tensor = model(x)
        print(outputs)

        # Loss function: BCEWithLogitsLoss
        # model側のnn.Sigmoidを削除したためスケーリングが必要
        scaled_outputs = torch.sigmoid(outputs)
        print(scaled_outputs)
        pred = scaled_outputs >= THRESHOLD
        # print(pred)

        # Loss function: BCELoss
        # threshold = 0.5
        # pred = int(outputs >= threshold)

        # Loss function: CrossEntropyLoss
        # pred = int(torch.argmax(outputs))

        pred_as_int = pred.cpu().numpy().astype(dtype=np.int8)
        pred_as_int_flatten = list(chain.from_iterable(pred_as_int))
        print(pred_as_int_flatten)
        pred_list.extend(pred_as_int_flatten)

        # print(pred.cpu().numpy().astype(int))
        # pred_list.extend(pred.cpu().numpy().astype(int))

    return pred_list