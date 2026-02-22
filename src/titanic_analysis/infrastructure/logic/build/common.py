from torch import Tensor


def get_data_with_type_annotation(batch: list) -> tuple[Tensor, Tensor]:
    data: Tensor = batch[0]
    labels: Tensor = batch[1]
    return data, labels
