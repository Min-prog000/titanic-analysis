"""configファイル読み込み用DTOモジュール"""

from pydantic import BaseModel


class TrainingPipelineDTO(BaseModel):
    """訓練・テスト用configファイルの読み込み内容のためのDTO"""

    batch_size: int
    learning_rate: float
    epochs: int
