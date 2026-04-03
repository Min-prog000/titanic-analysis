"""実行内容の定数定義モジュール"""

from enum import Enum


class ExecutionMode(Enum):
    """Execution mode analysis, training and predict process"""

    ANALYSIS = 0
    LOGISTIC_REGRESSION = 1
    GRADIENT_BOOSTING = 2
    PYTORCH = 3
    PREDICT = 4
