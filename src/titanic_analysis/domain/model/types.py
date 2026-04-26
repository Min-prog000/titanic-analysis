from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

from titanic_analysis.infrastructure.io.training_pipeline.dto import (
    GradientBoostingClassifierConfigDTO,
    LogisticRegressionConfigDTO,
)

ConfigDtoTypes = LogisticRegressionConfigDTO | GradientBoostingClassifierConfigDTO
SklearnModelTypes = LogisticRegression | GradientBoostingClassifier
ModelTypes = LogisticRegression | GradientBoostingClassifier | nn.Module
