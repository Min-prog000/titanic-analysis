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
    GRADIENT_BOOSTING = auto()  # 2
    XGBOOST = auto()  # 3
    NEURAL_NETWORK = auto()  # 4
