from pathlib import Path
from typing import TypeVar

import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from titanic_analysis.application.constants import (
    PIPELINE_PREFIX_GBDT,
    PIPELINE_PREFIX_LOGREG,
)
from titanic_analysis.infrastructure.io.analysis.config_loader import (
    load_gradient_boosting_classifier_config,
    load_logistic_regression_config,
)
from titanic_analysis.infrastructure.io.constants import (
    GRADIENT_BOOSTING_DECISION_TREE,
    LOGISTIC_REGRESSION,
)
from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    GradientBoostingClassifierConfigDTO,
    LogisticRegressionConfigDTO,
)

C = TypeVar("C")
M = TypeVar("M")


class ModelStrategy[C, M]:
    def load_config(self) -> C:
        """Load config file.

        Raises:
            NotImplementedError: Run if no-defined method implemented.

        Returns:
            ConfigDtoTypes: Config file type related sklearn models.
        """
        raise NotImplementedError

    def create_model(self, random_state: int) -> M:
        raise NotImplementedError

    def generate_params(self, config: C) -> dict:
        raise NotImplementedError

    def get_pipeline_prefix(self) -> str:
        raise NotImplementedError


class LogisticRegressionStrategy(
    ModelStrategy[LogisticRegressionConfigDTO, LogisticRegression],
):
    def load_config(self) -> LogisticRegressionConfigDTO:
        return load_logistic_regression_config(Path(LOGISTIC_REGRESSION))

    def load_logistic_regression_config(
        self,
        config_path: Path,
    ) -> LogisticRegressionConfigDTO:
        with config_path.open() as file:
            config = yaml.safe_load(file)

        return LogisticRegressionConfigDTO(**config["model"])

    def create_model(self, random_state: int) -> LogisticRegression:
        return LogisticRegression(random_state=random_state)

    def generate_params(self, config: LogisticRegressionConfigDTO) -> dict:
        # max_iterの範囲生成
        max_iter_scope = [
            np.int16(max_iter)
            for max_iter in np.linspace(config.max_iter, 1000, num=10)
        ]
        return {
            "logisticregression__C": np.logspace(config.C, 3, num=7),
            "logisticregression__class_weight": [
                config.class_weight,
                {0: 1.0, 1: 0.5},
            ],
            "logisticregression__max_iter": max_iter_scope,
        }

    def get_pipeline_prefix(self) -> str:
        return PIPELINE_PREFIX_LOGREG

    def get_save_folder_name(self) -> str:
        return LOGISTIC_REGRESSION


class GradientBoostingStrategy(
    ModelStrategy[GradientBoostingClassifierConfigDTO, GradientBoostingClassifier],
):
    def load_config(self) -> GradientBoostingClassifierConfigDTO:
        return load_gradient_boosting_classifier_config(
            Path(GRADIENT_BOOSTING_DECISION_TREE),
        )

    def load_gradient_boosting_classifier_config(
        self,
        config_path: Path,
    ) -> GradientBoostingClassifierConfigDTO:
        with config_path.open() as file:
            config = yaml.safe_load(file)

        return GradientBoostingClassifierConfigDTO(**config["model"])

    def create_model(self, random_state: int) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(random_state=random_state)

    def generate_params(self, config: GradientBoostingClassifierConfigDTO) -> dict:
        return {
            f"{PIPELINE_PREFIX_GBDT}__learning_rate": np.logspace(
                config.learning_rate,
                -1,
                num=2,
            ),
            # f"{PIPELINE_PREFIX_GBDT}__n_estimators": range(100, 201, 100),
            f"{PIPELINE_PREFIX_GBDT}__max_depth": range(config.max_depth, 8),
            f"{PIPELINE_PREFIX_GBDT}__max_features": range(
                config.max_features["min"],
                config.max_features["max"],
            ),
            # f"{PIPELINE_PREFIX_GBDT}__subsample": np.arange(0.1, 1.1, 0.1),
        }

    def get_pipeline_prefix(self) -> str:
        return PIPELINE_PREFIX_GBDT

    def get_save_folder_name(self) -> str:
        return GRADIENT_BOOSTING_DECISION_TREE
