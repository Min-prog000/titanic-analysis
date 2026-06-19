"""実行内容の定数定義モジュール"""

from enum import Enum, auto


class ExecutionMode(Enum):
    """Execution mode analysis, training and predict process"""

    ANALYSIS = auto()  # 1
    TRAIN = auto()  # 2
    PREDICT = auto()  # 3


class TrainMethod(Enum):
    """Train methods"""

    LOGISTIC_REGRESSION = auto()  # 1
    RANDOM_FOREST = auto()  # 2
    GRADIENT_BOOSTING = auto()  # 3
    XGBOOST = auto()  # 4
    LIGHTGBM = auto()  # 5
    NEURAL_NETWORK = auto()  # 6
