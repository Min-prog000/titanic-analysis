"""ロガーユーティリティモジュール"""

from datetime import datetime, timedelta, timezone

from titanic_analysis.framework.log.constants import PATH_LOG


def generate_log_file_path(folder_path: str = PATH_LOG) -> str:
    """ログファイル名を生成する

    Returns:
        str: ログファイル名
    """
    jst = timezone(timedelta(hours=+9), "JST")
    log_datetime = f"{datetime.now(jst):%Y%m%d%H%M%S}"

    return rf"{folder_path}\log_{log_datetime}.log"
