"""Constants module for model building pipeline"""

from typing import Literal

SEED = 42

SELECTED_FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]

ID_COLUMN = "PassengerId"
TARGET_COLUMN = "Survived"
ADDITIONAL_ENCODING_COLUMN = "Embarked"

CASE_ID_PATH = "./config/id/case.joblib"
PYTORCH_CONFIG_PATH = "./config/model/base.yaml"
PYTORCH_TENSORBOARD_PATH = "./tensorboard_log"

LOGGING_LEVEL_LITERALS = Literal[10, 20, 30, 40, 50]

COLUMN_NOT_MATCH_MESSAGE = (
    "NotMatchSizeError: either array has one or more false components."
)
