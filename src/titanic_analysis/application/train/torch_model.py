"""Training use case using torch"""

import logging
from logging import Logger
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from yaml import safe_dump

from titanic_analysis.application.constants import (
    CASE_ID_PATH,
    ID_COLUMN,
    LOGGING_LEVEL_LITERALS,
    PYTORCH_CONFIG_PATH,
    PYTORCH_TENSORBOARD_PATH,
    SEED,
    TARGET_COLUMN,
)
from titanic_analysis.application.preprocess import preprocess_load_data
from titanic_analysis.domain.dataset.torch_dataset import TitanicTorchDataset
from titanic_analysis.domain.model.torch import NeuralNetwork
from titanic_analysis.infrastructure.io.analysis.config_loader import (
    load_pytorch_config,
)
from titanic_analysis.infrastructure.io.analysis.constants import (
    CONFIG_PATH as ANALYSIS_CONFIG_PATH,
)
from titanic_analysis.infrastructure.io.constants import (
    PATH_TEST,
    PATH_TRAIN,
)
from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    PytorchConfigDTO,
)
from titanic_analysis.infrastructure.io.utils import CsvUtility
from titanic_analysis.infrastructure.logic.analysis.display import (
    prepare_display,
)
from titanic_analysis.infrastructure.logic.build.test import test_loop
from titanic_analysis.infrastructure.logic.build.train import train_loop
from titanic_analysis.infrastructure.logic.build.utils import fix_seed, load_case_id

__all__ = ["train_neural_network"]


def train_neural_network(
    logger: Logger,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """Train and test using pytorch

    Args:
        logger (Logger): Logger for user information and debug.
        train_dataset_path (str, optional):
            Train dataset file path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional):
            Test dataset file path. Defaults to PATH_TEST.
    """
    fix_seed(SEED)

    prepare_display(ANALYSIS_CONFIG_PATH)

    config_path = Path(PYTORCH_CONFIG_PATH)
    config_loaded = load_pytorch_config(config_path)

    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    train_data_preprocessed, test_data_preprocessed = preprocess_load_data(
        logger,
        train_data,
        test_data,
    )

    # データセット
    train_labels = np.array(train_data.loc[:, TARGET_COLUMN])

    log_label_distribution(logger, train_labels)

    train_dataset = TitanicTorchDataset(
        train_data_preprocessed,
        train_labels,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config_loaded.batch_size,
        shuffle=False,
    )

    feature_size = train_data_preprocessed.shape[1]
    model = NeuralNetwork(feature_size)
    logger.info("\n%s", summary(model, (1, feature_size)))

    # 1出力
    pos_weight = torch.tensor([config_loaded.pos_weight])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # loss_fn = nn.BCELoss()

    # 2出力
    # weight = torch.tensor([0.9, 1.0])
    # loss_fn = nn.CrossEntropyLoss(weight=weight)

    train_accuracy_list = []
    train_loss_list = []
    train_correct_list = []

    optimizer = optim.Adam(
        model.parameters(),
        config_loaded.learning_rate,
        weight_decay=config_loaded.weight_decay,
    )

    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: config_loaded.gamma**epoch,
    )

    for epoch in range(config_loaded.epochs):
        train_epoch_accuracy, train_epoch_loss, train_epoch_correct, model = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            config_loaded.epochs,
            epoch,
        )
        train_accuracy_list.append(train_epoch_accuracy)
        train_loss_list.append(train_epoch_loss)
        train_correct_list.append(train_epoch_correct)

    # ケース番号
    case_id_path = Path(CASE_ID_PATH)
    case_id = load_case_id(case_id_path)

    # TensorBoard のログ出力先
    root_log_dir = Path(PYTORCH_TENSORBOARD_PATH)
    # ラベル名
    main_tags = ["accuracy", "loss", "correct"]
    value_tag = f"case{case_id}"
    train_histories = [train_accuracy_list, train_loss_list, train_correct_list]
    for i in range(len(main_tags)):
        log_dir = root_log_dir.joinpath(main_tags[i])
        write_scalar_graph(log_dir, train_histories[i], main_tags[i], value_tag)

    # データセット
    test_labels = np.array([0] * test_data_preprocessed.shape[0])
    logger.debug(test_labels.shape)
    test_dataset = TitanicTorchDataset(
        test_data_preprocessed,
        test_labels,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config_loaded.batch_size,
        shuffle=False,
    )

    pred_list = test_loop(test_dataloader, model)
    logger.debug(len(test_data[ID_COLUMN].to_numpy()))
    logger.debug(len(pred_list))

    submission_data = pd.DataFrame(
        {
            ID_COLUMN: test_data[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: pred_list,
        },
    )
    CsvUtility.output_csv(submission_data, "torch_neural-network")

    create_onnx_model(feature_size, model, case_id)

    save_config(config_loaded, case_id)

    joblib.dump(case_id + 1, case_id_path)


def log_label_distribution(logger: Logger, train_labels: np.ndarray) -> None:
    logger.debug("\n%s", train_labels)
    bin_count = np.bincount(train_labels)
    logger.debug(bin_count)

    false_percentage: np.float64 = bin_count[0] / np.sum(bin_count)
    true_percentage: np.float64 = bin_count[1] / np.sum(bin_count)
    logger.debug("false_percentage type: %s", type(false_percentage))
    logger.debug("true_percentage type: %s", type(true_percentage))
    logger.debug("False: %s %%", float(false_percentage))
    logger.debug("True: %s %%", float(true_percentage))


def save_config(config_loaded: PytorchConfigDTO, case_id: int) -> None:
    config_save = {
        "model": {
            "case_id": case_id,
        },
    }
    config_save["model"].update(config_loaded.model_dump())
    yaml_output_path = Path(f"output/config/case{case_id}")
    yaml_output_path.mkdir(parents=True, exist_ok=True)
    config_file_name = Path(f"config_case{case_id}.yaml")
    config_file_path = yaml_output_path.joinpath(config_file_name)
    with config_file_path.open(mode="w", encoding="utf-8") as f:
        safe_dump(config_save, f, sort_keys=False)


def create_onnx_model(feature_size: int, model: NeuralNetwork, case_id: int) -> None:
    input_tensor = torch.rand((1, feature_size), dtype=torch.float32)

    onnx_dir_path = Path(f"model/onnx/case{case_id}")
    onnx_dir_path.mkdir(parents=True, exist_ok=True)

    onnx_file_name = Path(f"case{case_id}.onnx")
    onnx_file_path = onnx_dir_path.joinpath(onnx_file_name)

    set_onnx_logger()

    torch.onnx.export(
        model,
        (input_tensor,),
        onnx_file_path,
        input_names=["input"],
        output_names=["output"],
        optimize=True,
        dynamo=True,
    )


def set_onnx_logger(
    level_onnxscript: LOGGING_LEVEL_LITERALS = logging.ERROR,
    level_onnx_ir: LOGGING_LEVEL_LITERALS = logging.ERROR,
) -> None:
    logging.getLogger("onnxscript").setLevel(level_onnxscript)
    logging.getLogger("onnx_ir").setLevel(level_onnx_ir)


def write_scalar_graph(
    log_dir: Path,
    plot_list: list,
    main_tag: str,
    value_tag: str,
) -> None:
    writer = SummaryWriter(log_dir=log_dir)
    for step in range(len(plot_list)):
        # add_scalars を使うと、1 つのグラフに複数線が色分けされて表示される
        writer.add_scalars(main_tag, {value_tag: plot_list[step]}, step)

    writer.close()
