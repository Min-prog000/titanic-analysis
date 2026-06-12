"""Defines type aliases for models and config DTOs in the Titanic training pipeline.

This module aggregates configuration DTO types for scikit-learn models
(Logistic Regression and Gradient Boosting Classifier) and provides common
type unions for the corresponding model classes. It also includes PyTorch's
nn.Module to support neural-network-based model implementations.

By centralizing these type definitions, the training pipeline can handle
multiple model families in a consistent and type-safe manner.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    GradientBoostingClassifierConfigDTO,
    LogisticRegressionConfigDTO,
    RandomForestClassifierConfigDTO,
)

ConfigDtoTypes = (
    LogisticRegressionConfigDTO
    | RandomForestClassifierConfigDTO
    | GradientBoostingClassifierConfigDTO
)
SklearnModelTypes = (
    LogisticRegression | RandomForestClassifier | GradientBoostingClassifier
)
ModelTypes = (
    LogisticRegression | RandomForestClassifier | GradientBoostingClassifier | nn.Module
)
