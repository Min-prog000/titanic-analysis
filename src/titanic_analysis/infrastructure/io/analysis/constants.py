"""データセット読み込み用の定数定義モジュール"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# 分析用configファイル（analysis/base.yaml）のパス
CONFIG_PATH = BASE_DIR / "base.yaml"
