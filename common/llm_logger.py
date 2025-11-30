"""LLM入出力ログ管理モジュール

Foundry Local（ローカルLLM）への入出力をログとして記録し、
Streamlit UIなどでリアルタイム表示するためのコールバック機構を提供する。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional


@dataclass
class LLMLogEntry:
    """LLM呼び出しの入出力ログエントリ"""

    timestamp: datetime
    tool_name: str
    system_prompt: str
    user_content: str
    response_text: Optional[str] = None
    parsed_result: Optional[Any] = None


class LLMLogger:
    """LLM入出力を記録するロガー

    コールバックを設定することで、ログ追加時にリアルタイムで通知を受け取れる。
    """

    def __init__(self):
        self.entries: list[LLMLogEntry] = []
        self._on_log_callback: Optional[Callable[[LLMLogEntry], None]] = None

    def set_callback(self, callback: Callable[[LLMLogEntry], None]) -> None:
        """ログ追加時に呼び出されるコールバックを設定

        Args:
            callback: LLMLogEntryを受け取るコールバック関数
        """
        self._on_log_callback = callback

    def log_request(
        self, tool_name: str, system_prompt: str, user_content: str
    ) -> LLMLogEntry:
        """リクエスト開始をログに記録

        Args:
            tool_name: ツール名（例: "analyze_financial_assets"）
            system_prompt: システムプロンプト
            user_content: ユーザーコンテンツ（入力データ）

        Returns:
            作成されたログエントリ（後でlog_responseで更新する）
        """
        entry = LLMLogEntry(
            timestamp=datetime.now(),
            tool_name=tool_name,
            system_prompt=system_prompt,
            user_content=user_content,
        )
        self.entries.append(entry)
        if self._on_log_callback:
            self._on_log_callback(entry)
        return entry

    def log_response(
        self, entry: LLMLogEntry, response_text: str, parsed_result: Any = None
    ) -> None:
        """レスポンスをログに記録

        Args:
            entry: log_requestで作成されたログエントリ
            response_text: LLMからの生レスポンステキスト
            parsed_result: パース済みの結果（通常はdict）
        """
        entry.response_text = response_text
        entry.parsed_result = parsed_result
        if self._on_log_callback:
            self._on_log_callback(entry)

    def clear(self) -> None:
        """ログをクリアし、コールバックをリセット"""
        self.entries.clear()
        self._on_log_callback = None


# グローバルインスタンス
llm_logger = LLMLogger()
