"""configファイル読み込み用DTOモジュール"""

from pydantic import BaseModel


class LogisticRegressionConfigDTO(BaseModel):
    """DTO for LogisticRegression"""

    random_state: int
    C: float
    class_weight: dict
    max_iter: int


class GradientBoostingClassifierConfigDTO(BaseModel):
    """DTO for GradientBoostingClassifier"""

    random_state: int
    learning_rate: float
    n_estimators: int
    max_depth: int
    max_features: dict[str, int]
    subsample: float


class XGBoostConfigDTO(BaseModel):
    """DTO for xgboost"""

    random_state: int
    n_estimators: int
    max_depth: int


class PytorchConfigDTO(BaseModel):
    """DTO for Pytorch"""

    batch_size: int
    learning_rate: float
    gamma: float
    epochs: int
    pos_weight: float
    weight_decay: float
