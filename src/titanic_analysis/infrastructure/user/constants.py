"""実行内容の定数定義モジュール"""

from enum import Enum


class ExecutionMode(Enum):
    """Execution mode analysis, training and predict process"""

    ANALYSIS = 0
    TRAIN = 1
    PREDICT = 2


class TrainMethod(Enum):
    """Train methods"""

    LOGISTIC_REGRESSION = 0
    GRADIENT_BOOSTING = 1
    NEURAL_NETWORK = 2
