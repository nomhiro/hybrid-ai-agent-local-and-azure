"""
MCPサーバー状態管理モジュール

MCPサーバーの起動状態、リクエスト履歴を管理し、
Streamlit UIへのリアルタイム表示をサポートする。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import threading


class ServerStatus(Enum):
    """MCPサーバーの状態"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class TunnelStatus(Enum):
    """Dev Tunnelの状態"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    NOT_INSTALLED = "not_installed"
    NOT_LOGGED_IN = "not_logged_in"


@dataclass
class MCPRequestLog:
    """MCPリクエストのログエントリ"""
    timestamp: datetime
    method: str  # "initialize", "tools/list", "tools/call"
    request_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None
    response: Optional[Any] = None
    error: Optional[str] = None
    llm_input: Optional[str] = None  # Foundry Localへの入力
    llm_output: Optional[str] = None  # Foundry Localからの出力
    duration_ms: Optional[float] = None


@dataclass
class MCPState:
    """MCPサーバーの状態を管理するクラス"""

    status: ServerStatus = ServerStatus.STOPPED
    port: int = 8081
    local_url: str = ""
    tunnel_url: str = ""
    error_message: str = ""
    request_logs: list[MCPRequestLog] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _on_update_callback: Optional[Callable[["MCPState"], None]] = field(default=None, repr=False)

    # Dev Tunnel関連フィールド
    tunnel_status: TunnelStatus = TunnelStatus.STOPPED
    tunnel_error: str = ""
    tunnel_auto_started: bool = False

    def set_callback(self, callback: Callable[["MCPState"], None]) -> None:
        """状態更新時に呼び出されるコールバックを設定"""
        self._on_update_callback = callback

    def _notify_update(self) -> None:
        """状態更新を通知"""
        if self._on_update_callback:
            self._on_update_callback(self)

    def set_status(self, status: ServerStatus, error_message: str = "") -> None:
        """サーバー状態を更新"""
        with self._lock:
            self.status = status
            self.error_message = error_message
            if status == ServerStatus.RUNNING:
                self.local_url = f"http://localhost:{self.port}"
            elif status == ServerStatus.STOPPED:
                self.local_url = ""
                self.tunnel_url = ""
        self._notify_update()

    def set_tunnel_url(self, url: str, auto_started: bool = False) -> None:
        """Dev Tunnel URLを設定"""
        with self._lock:
            self.tunnel_url = url
            self.tunnel_auto_started = auto_started
        self._notify_update()

    def set_tunnel_status(self, status: TunnelStatus, error: str = "") -> None:
        """Dev Tunnel状態を更新"""
        with self._lock:
            self.tunnel_status = status
            self.tunnel_error = error
        self._notify_update()

    def add_request_log(self, log: MCPRequestLog) -> None:
        """リクエストログを追加"""
        print(f"[MCPState] ログ追加: method={log.method}, timestamp={log.timestamp}")
        with self._lock:
            self.request_logs.append(log)
            print(f"[MCPState] 現在のログ数: {len(self.request_logs)}")
            # 最大100件まで保持
            if len(self.request_logs) > 100:
                self.request_logs = self.request_logs[-100:]
        self._notify_update()

    def update_request_log(
        self,
        log: MCPRequestLog,
        response: Any = None,
        error: str = None,
        llm_input: str = None,
        llm_output: str = None,
        duration_ms: float = None,
    ) -> None:
        """既存のリクエストログを更新"""
        with self._lock:
            if response is not None:
                log.response = response
            if error is not None:
                log.error = error
            if llm_input is not None:
                log.llm_input = llm_input
            if llm_output is not None:
                log.llm_output = llm_output
            if duration_ms is not None:
                log.duration_ms = duration_ms
        self._notify_update()

    def clear_logs(self) -> None:
        """リクエストログをクリア"""
        with self._lock:
            self.request_logs.clear()
        self._notify_update()

    def get_recent_logs(self, count: int = 10) -> list[MCPRequestLog]:
        """最新のログを取得"""
        with self._lock:
            return list(self.request_logs[-count:])

    def is_running(self) -> bool:
        """サーバーが実行中かどうか"""
        return self.status == ServerStatus.RUNNING


# グローバルインスタンス（医療トリアージ用）
mcp_state = MCPState()
