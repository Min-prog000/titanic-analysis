"""logger定義"""

# TODO(minplu): ロガーの出力メッセージをカスタマイズする
# 4

from __future__ import annotations

import logging
from logging import DEBUG, FileHandler, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path

STREAM_HANDLER_LOG_FORMAT = (
    "%(asctime)s %(name)s [%(levelname)s] %(funcName)s: %(message)s"
)
FILE_HANDLER_LOG_FORMAT = (
    "%(asctime)s %(name)s [%(levelname)s] %(funcName)s: %(message)s"
)


class TitanicLogger:
    """ログ設定用のクラス"""

    def __init__(
        self,
        logger_name: str,
        log_level_stream: int = DEBUG,
        log_level_file: int = DEBUG,
        log_file_name: str | None = None,
        stream_format: str = STREAM_HANDLER_LOG_FORMAT,
        file_format: str = FILE_HANDLER_LOG_FORMAT,
    ) -> None:
        """コンストラクタ"""
        self.logger_name = logger_name
        self.log_level_stream = log_level_stream
        self.log_level_file = log_level_file
        self.log_file_name = log_file_name
        self.stream_format = stream_format
        self.file_format = file_format

        self._set_logger()

    def _set_logger(self) -> None:
        """ロガーの出力設定を行う"""
        logging.basicConfig(level=DEBUG)
        self._logger = getLogger(self.logger_name)
        self._logger.handlers.clear()
        self._logger.propagate = False

        self._add_stream_handler()

        if self.log_file_name is not None:
            log_folder_path = Path(self.log_file_name).parent
            log_folder_path.mkdir(parents=True, exist_ok=True)
            self._add_file_handler(self.log_file_name)

    def _add_stream_handler(self) -> None:
        """ロガーにストリームハンドラーを追加する"""
        stream_handler = StreamHandler()

        stream_handler.setLevel(self.log_level_stream)

        stream_formatter = Formatter(self.stream_format)
        stream_handler.setFormatter(stream_formatter)

        self._logger.addHandler(stream_handler)

    def _add_file_handler(self, log_file_name: str) -> None:
        """ロガーにファイルハンドラーを追加する

        Args:
            log_file_name (str): ログファイル名
        """
        file_handler = FileHandler(log_file_name, encoding="utf-8")

        file_handler.setLevel(self.log_level_file)

        file_formatter = Formatter(self.file_format)
        file_handler.setFormatter(file_formatter)

        self._logger.addHandler(file_handler)

    @property
    def logger(self) -> Logger:
        """ロガーを取得する

        Returns:
            Logger: ロガー

        """
        return self._logger
