"""ロガーユーティリティモジュール"""

from datetime import datetime, timedelta, timezone

from titanic_analysis.interface.log.constants import PATH_LOG


def generate_log_file_path(folder_path: str = PATH_LOG) -> str:
    """ログファイル名を生成する

    Returns:
        str: ログファイル名
    """
    jst = timezone(timedelta(hours=+9), "JST")
    datetime_now = datetime.now(jst)

    log_date = datetime_now.strftime("%Y%m%d")
    log_datetime = datetime_now.strftime("%Y%m%d%H%M%S")

    return rf"{folder_path}\{log_date}\log_{log_datetime}.log"
