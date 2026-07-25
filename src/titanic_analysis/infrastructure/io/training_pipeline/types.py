"""Types module for DTO"""

from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    GradientBoostingClassifierConfigDTO,
    LogisticRegressionConfigDTO,
    PytorchConfigDTO,
    RandomForestClassifierConfigDTO,
)

DTOs = (
    LogisticRegressionConfigDTO
    | RandomForestClassifierConfigDTO
    | GradientBoostingClassifierConfigDTO
    | PytorchConfigDTO
)
