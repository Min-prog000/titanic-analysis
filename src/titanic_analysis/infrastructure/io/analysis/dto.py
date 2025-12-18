"""configファイル読み込み用DTOモジュール"""

from pydantic import BaseModel


class AnalysisDTO(BaseModel):
    """分析用configファイルの読み込み内容のためのDTO"""

    max_rows: int
    max_columns: int
