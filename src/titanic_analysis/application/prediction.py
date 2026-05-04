"""Prediction use case"""

from collections.abc import Sequence
from datetime import datetime
from logging import Logger
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch import sigmoid

from titanic_analysis.application.constants import (
    ID_COLUMN,
    JST,
    PREDICT_SUBMISSION_FORMAT,
    SELECTED_FEATURES,
    TARGET_COLUMN,
)
from titanic_analysis.application.train.torch_model import preprocess_load_data
from titanic_analysis.application.types import OutputItem
from titanic_analysis.infrastructure.io.constants import PATH_TEST, PATH_TRAIN
from titanic_analysis.infrastructure.logic.build.constants import THRESHOLD

__all__ = ["predict"]


def predict(
    logger: Logger,
    model_path: str,
    train_dataset_path: str = PATH_TRAIN,
    test_dataset_path: str = PATH_TEST,
) -> None:
    """Infer with ONNX model.

    Args:
        logger (Logger): Logger for user information and debug
        model_path (str): ONNX (`*.onnx`) file path
        train_dataset_path (str, optional): Train dataset path. Defaults to PATH_TRAIN.
        test_dataset_path (str, optional): Test dataset path. Defaults to PATH_TEST.
    """
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    logger.debug(test_data.columns)

    logger.debug(SELECTED_FEATURES)

    _, test_data_preprocessed = preprocess_load_data(
        logger,
        train_data,
        test_data,
    )

    logger.debug(test_data_preprocessed.shape)

    # データセット
    test_dataset = np.array(test_data_preprocessed, dtype=np.float32)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name: str = session.get_inputs()[0].name
    output_name: str = session.get_outputs()[0].name

    logger.debug("input name: %s", input_name)
    logger.debug("output name: %s", output_name)

    for onnx_input in session.get_inputs():
        logger.debug("input name: %s", onnx_input.name)
        logger.debug("input shape: %s", onnx_input.shape)
        logger.debug("input type: %s", onnx_input.type)

    logger.debug(test_dataset.shape)

    output_list = []

    for data in test_dataset:
        input_data = np.expand_dims(data, axis=(0, 1))
        input_data = input_data.reshape(1, -1).astype(np.float32)
        output = session.run(
            output_names=[output_name],
            input_feed={input_name: input_data},
            run_options=None,
        )
        output = extract_scalar(output)
        output = sigmoid(torch.tensor(output))
        output = (output >= THRESHOLD).numpy().astype(int)
        output_list.append(output.item())

    submission_data = pd.DataFrame(
        {
            ID_COLUMN: test_data[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: output_list,
        },
    )
    logger.debug(submission_data.shape)

    model_file_name = Path(model_path).stem
    predict_id = generate_now_datetime()
    submission_folder_path = Path(
        f"output/onnx_inference/{model_file_name}/{predict_id}",
    )
    submission_folder_path.mkdir(parents=True, exist_ok=True)
    submission_file_name = Path(f"{model_file_name}_output.csv")
    submission_file_path = submission_folder_path.joinpath(submission_file_name)
    submission_data.to_csv(submission_file_path, index=False)


def extract_scalar(output: Sequence[OutputItem]) -> OutputItem:
    """Extract predict result from numpy array

    Args:
        output (Sequence[OutputItem]): raw predict result

    Raises:
        ValueError: raise when predict result is empty

    Returns:
        OutputItem: extracted result
    """
    if not output:
        msg = "Output list is empty"
        raise ValueError(msg)

    item = output[0]

    if isinstance(item, np.ndarray):
        return item.flatten()[0]

    return item


def generate_now_datetime(datetime_format: str = PREDICT_SUBMISSION_FORMAT) -> str:
    """Generate datetime formatted string for submission file prefix

    Args:
        datetime_format (str, optional): datetime format. Defaults to "%Y%m%d%H%M%S".

    Returns:
        str: datetime formatted string
    """
    datetime_now = datetime.now(JST)

    return datetime_now.strftime(datetime_format)
