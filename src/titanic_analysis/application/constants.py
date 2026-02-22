from typing import Literal

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

LOGGING_LEVEL_LITERALS = Literal[10, 20, 30, 40, 50]
