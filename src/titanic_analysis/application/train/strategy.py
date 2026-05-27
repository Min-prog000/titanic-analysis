"""Defines strategy abstractions for model configuration, creation, and tuning.

This module provides a generic ModelStrategy interface and concrete strategies
for Logistic Regression and Gradient Boosting models. Each strategy handles
loading configuration from YAML files, constructing reproducible model
instances, and generating hyperparameter search spaces for tuning. Centralizing
these behaviors enables consistent and extensible model management across the
training pipeline.
"""

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
    """Strategy interface"""

    def load_config(self) -> C:
        """Load config file.

        Raises:
            NotImplementedError: Run if no-defined method implemented.

        Returns:
            ConfigDtoTypes: Config file type related sklearn models.
        """
        raise NotImplementedError

    def create_model(self, random_state: int) -> M:
        """Build model.

        Args:
            random_state (int): Parameter for model reproducibility

        Raises:
            NotImplementedError: Not implemented error

        Returns:
            M: Sklearn model
        """
        raise NotImplementedError

    def generate_params(self, config: C) -> dict:
        """Generate parameters dict for grid search.

        Args:
            config (C): Config DTO types

        Raises:
            NotImplementedError: Not implemented error

        Returns:
            dict: Parameters dict for grid search
        """
        raise NotImplementedError

    def get_pipeline_prefix(self) -> str:
        """Get model identifier prefix for pipeline.

        Raises:
            NotImplementedError: Not implemented error

        Returns:
            str: Prefix
        """
        raise NotImplementedError

    def get_save_folder_name(self) -> str:
        """Get folder name string for model saving.

        Raises:
            NotImplementedError: Not implemented error

        Returns:
            str: Folder name string
        """
        raise NotImplementedError

    def get_csv_postfix(self) -> str:
        """Get csv name postfix string for submission.

        Raises:
            NotImplementedError: Not implemented error

        Returns:
            str: Csv output file name postfix
        """
        raise NotImplementedError


class LogisticRegressionStrategy(
    ModelStrategy[LogisticRegressionConfigDTO, LogisticRegression],
):
    """Strategy for logistic regression.

    Args:
        ModelStrategy (LogisticRegressionConfigDTO, LogisticRegression):
            Strategy interface
    """

    def load_config(self) -> LogisticRegressionConfigDTO:
        """Load config file for `LogisticRegression`.

        Returns:
            LogisticRegressionConfigDTO: Parameters DTO
        """
        config_path = Path("config/model/base_logreg.yaml")

        with config_path.open() as file:
            config = yaml.safe_load(file)

        return LogisticRegressionConfigDTO(**config["model"])

    def create_model(self, random_state: int) -> LogisticRegression:
        """Build `LogisticRegression` model.

        Args:
            random_state (int): `random_state` for `LogisticRegression` argument

        Returns:
            LogisticRegression: Model instance
        """
        return LogisticRegression(random_state=random_state)

    def generate_params(self, config: LogisticRegressionConfigDTO) -> dict:
        """Generate parameters dict for `LogisticRegression`.

        Args:
            config (LogisticRegressionConfigDTO): Config DTO

        Returns:
            dict: Config dict
        """
        # max_iterの範囲生成
        return {
            "logisticregression__C": np.logspace(config.C, 3, num=7),
            "logisticregression__class_weight": [
                config.class_weight,
                {0: 1.0, 1: 0.5},
            ],
            "logisticregression__max_iter": self.get_max_iter(config),
        }

    def get_max_iter(self, config: LogisticRegressionConfigDTO) -> list:
        """Get `max_iter` scope for grid search.

        Args:
            config (LogisticRegressionConfigDTO): Config DTO

        Returns:
            list: `max_iter` scope
        """
        return [
            np.int16(max_iter)
            for max_iter in np.linspace(config.max_iter, 1000, num=10)
        ]

    def get_pipeline_prefix(self) -> str:
        """Get prefix string for model identifier of `Pipeline`.

        Returns:
            str: Model prefix in `Pipeline`
        """
        return PIPELINE_PREFIX_LOGREG

    def get_save_folder_name(self) -> str:
        """Get folder name string for model saving.

        Returns:
            str: Folder name string
        """
        return LOGISTIC_REGRESSION

    def get_csv_postfix(self) -> str:
        """Return output csv file name postfix for submission.

        Returns:
            str: Csv file name postfix
        """
        return LOGISTIC_REGRESSION


class GradientBoostingStrategy(
    ModelStrategy[GradientBoostingClassifierConfigDTO, GradientBoostingClassifier],
):
    """Strategy class for managing configuration loading, model creation, and hyperparameter generation for a Gradient Boosting Classifier.

    This class implements the ModelStrategy interface and provides utilities
    specific to Gradient Boosting models.
    """

    def load_config(self) -> GradientBoostingClassifierConfigDTO:
        """Load the Gradient Boosting model configuration from a YAML file.

        Returns:
            GradientBoostingClassifierConfigDTO:
                A configuration object containing model hyperparameters
                loaded from the YAML file.
        """
        config_path = Path("config/model/base_gbdt.yaml")

        with config_path.open() as file:
            config = yaml.safe_load(file)

        return GradientBoostingClassifierConfigDTO(**config["model"])

    def create_model(self, random_state: int) -> GradientBoostingClassifier:
        """Create a GradientBoostingClassifier instance using the provided random state.

        Args:
            random_state (int):
                Seed value for ensuring reproducible model behavior.

        Returns:
            GradientBoostingClassifier:
                A new instance of the Gradient Boosting classifier.
        """
        return GradientBoostingClassifier(random_state=random_state)

    def generate_params(self, config: GradientBoostingClassifierConfigDTO) -> dict:
        """Generate a dictionary of hyperparameter search ranges for use in grid search or randomized search.

        Args:
            config (GradientBoostingClassifierConfigDTO):
                Configuration object containing parameter ranges.

        Returns:
            dict:
                A mapping of pipeline-prefixed parameter names to iterable
                ranges or distributions for hyperparameter tuning.
        """
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
        """Retrieve the pipeline prefix used for Gradient Boosting parameters.

        Returns:
            str: The pipeline prefix string.
        """
        return PIPELINE_PREFIX_GBDT

    def get_save_folder_name(self) -> str:
        """Get the folder name where model artifacts should be saved.

        Returns:
            str: The folder name associated with Gradient Boosting models.
        """
        return GRADIENT_BOOSTING_DECISION_TREE

    def get_csv_postfix(self) -> str:
        """Return output csv file name postfix for submission.

        Returns:
            str: Csv file name postfix
        """
        return GRADIENT_BOOSTING_DECISION_TREE
