from torch import Tensor, nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from titanic_analysis.infrastructure.logic.build.common import get_data_with_type_annotation

from titanic_analysis.domain.model.torch import NeuralNetwork
from titanic_analysis.infrastructure.logic.build.constants import THRESHOLD


def train_loop(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.BCEWithLogitsLoss | nn.BCELoss | nn.CrossEntropyLoss,
    optimizer: optim.Adam | optim.SGD,
    scheduler: lr_scheduler.LambdaLR,
    epochs: int,
    epoch: int,
) -> tuple[float, float, int, NeuralNetwork]:
    epoch_accuracy = 0
    epoch_loss = 0
    epoch_correct = 0
    total_count = 0

    model.train()
    batch_size = len(dataloader)

    with tqdm(dataloader) as pbar:
        pbar.set_description(f"[Epoch {epoch + 1}/{epochs}]")
        for batch in pbar:
            data, labels = get_data_with_type_annotation(batch)
            batch_size = labels.shape[0]
            # 予測と損失の計算
            outputs: Tensor = model(data)
            # print(outputs)
            # labels = labels.squeeze(1).long()

            loss: Tensor = loss_fn(outputs, labels)

            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = (outputs >= THRESHOLD).float()

            batch_correct = int((pred == labels).sum().item())
            batch_accuracy = batch_correct / batch_size

            epoch_correct += batch_correct
            total_count += batch_size
            epoch_accuracy = epoch_correct / total_count
            epoch_loss += loss.item()

            pbar_postfix = {
                "batch_acc": batch_accuracy,
                "batch_loss": loss.item(),
                "epoch_acc": epoch_accuracy,
                "epoch_loss": epoch_loss,
            }

            pbar.set_postfix(pbar_postfix)

        scheduler.step()

    return epoch_accuracy, epoch_loss, epoch_correct, model
