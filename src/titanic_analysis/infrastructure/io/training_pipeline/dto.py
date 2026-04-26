"""configファイル読み込み用DTOモジュール"""

from pydantic import BaseModel


class LogisticRegressionConfigDTO(BaseModel):
    """ロジスティック回帰用DTO"""

    random_state: int
    C: float
    class_weight: dict
    max_iter: int


class GradientBoostingClassifierConfigDTO(BaseModel):
    """勾配ブースティング用DTO"""

    random_state: int
    learning_rate: float
    n_estimators: int
    max_depth: int
    max_features: int
    subsample: float


class PytorchConfigDTO(BaseModel):
    """訓練・テスト用configファイルの読み込み内容のためのDTO"""

    batch_size: int
    learning_rate: float
    gamma: float
    epochs: int
    pos_weight: float
    weight_decay: float
