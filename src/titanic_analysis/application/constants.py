"""Constants module for model building pipeline"""

from datetime import timedelta, timezone
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
LOGREG_CONFIG_PATH = "./config/model/base_logreg.yaml"
GBDT_CONFIG_PATH = "./config/model/base_gbdt.yaml"
XGBOOST_CONFIG_PATH = "./config/model/base_xgboost.yaml"
PYTORCH_CONFIG_PATH = "./config/model/base_pytorch.yaml"

XGBOOST_TREE_PATH = "./output/xgboost"
PYTORCH_TENSORBOARD_PATH = "./tensorboard_log"

LOGGING_LEVEL_LITERALS = Literal[10, 20, 30, 40, 50]

COLUMN_NOT_MATCH_MESSAGE = (
    "NotMatchSizeError: either array has one or more false components."
)

PIPELINE_PREFIX_LOGREG = "logisticregression"
PIPELINE_PREFIX_GBDT = "gradientboostingclassifier"
PIPELINE_PREFIX_XGBOOST = "xgbclassifier"

PREDICT_SUBMISSION_FORMAT = "%Y%m%d%H%M%S"
JST = timezone(timedelta(hours=+9), "JST")

BOOSTER_DUMP_FORMAT_XGBOOST = "dot"
TREE_RENDER_FORMAT_XGBOOST = "png"
