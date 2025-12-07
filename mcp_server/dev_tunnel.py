"""
Dev Tunnel管理モジュール

Azure Dev Tunnelプロセスを起動・停止し、
URLを自動取得する機能を提供する。
"""

import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from .mcp_state import TunnelStatus


@dataclass
class TunnelResult:
    """Dev Tunnel起動結果"""

    success: bool
    url: str = ""
    error_message: str = ""
    status: TunnelStatus = TunnelStatus.STOPPED


class DevTunnelManager:
    """Dev Tunnelプロセスを管理するクラス"""

    # URL抽出用正規表現
    # "Connect via browser:" 行から接続用URLを抽出
    # 例: "Connect via browser: https://abc123.asse.devtunnels.ms:8081, https://abc123-8081.asse.devtunnels.ms"
    # Inspect URLは除外（-inspect が含まれるURLは無視）
    CONNECT_URL_PATTERN = re.compile(
        r"Connect via browser:\s*(https://[a-zA-Z0-9\-]+(?:\.[a-z0-9]+)?\.devtunnels\.ms(?::\d+)?)",
        re.IGNORECASE,
    )

    # ログイン要求の検出パターン
    LOGIN_REQUIRED_PATTERN = re.compile(
        r"(not logged in|login required|please login|unauthorized|"
        r"Run 'devtunnel user login')",
        re.IGNORECASE,
    )

    def __init__(
        self,
        port: int,
        on_url_ready: Optional[Callable[[str], None]] = None,
        on_status_change: Optional[Callable[[TunnelStatus, str], None]] = None,
    ):
        """
        Args:
            port: トンネルするローカルポート
            on_url_ready: URL取得時のコールバック
            on_status_change: 状態変更時のコールバック
        """
        self.port = port
        self.on_url_ready = on_url_ready
        self.on_status_change = on_status_change

        self.process: Optional[subprocess.Popen] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.url: str = ""
        self.status: TunnelStatus = TunnelStatus.STOPPED
        self.error_message: str = ""
        self._stop_event = threading.Event()
        self._url_ready_event = threading.Event()
        self._lock = threading.Lock()

    def _set_status(self, status: TunnelStatus, error_message: str = ""):
        """状態を更新してコールバックを呼び出す"""
        with self._lock:
            self.status = status
            self.error_message = error_message
        if self.on_status_change:
            self.on_status_change(status, error_message)

    def _check_devtunnel_installed(self) -> bool:
        """devtunnel CLIがインストールされているか確認"""
        return shutil.which("devtunnel") is not None

    def _check_devtunnel_logged_in(self) -> bool:
        """devtunnelにログイン済みか確認"""
        try:
            # Windows用のフラグ
            creation_flags = 0
            if os.name == "nt":
                creation_flags = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                ["devtunnel", "user", "show"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=creation_flags,
            )
            # ログイン済みの場合、ユーザー情報が出力される（returncode 0）
            # "Logged in as" や "Account:" などが含まれる
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            print(f"[DevTunnel] ログイン確認エラー: {e}")
            return False

    def start(self) -> TunnelResult:
        """
        Dev Tunnelを起動してURLを取得する。

        Returns:
            TunnelResult: 起動結果
        """
        # 既に実行中の場合
        if self.process is not None and self.process.poll() is None:
            return TunnelResult(
                success=True,
                url=self.url,
                error_message="Dev Tunnel is already running",
                status=TunnelStatus.RUNNING,
            )

        # devtunnelがインストールされているか確認
        if not self._check_devtunnel_installed():
            self._set_status(
                TunnelStatus.NOT_INSTALLED, "devtunnel CLIがインストールされていません"
            )
            return TunnelResult(
                success=False,
                error_message="devtunnel CLIがインストールされていません。\n"
                "インストール: winget install Microsoft.devtunnel",
                status=TunnelStatus.NOT_INSTALLED,
            )

        # ログイン状態を確認
        if not self._check_devtunnel_logged_in():
            self._set_status(TunnelStatus.NOT_LOGGED_IN, "devtunnelにログインが必要です")
            return TunnelResult(
                success=False,
                error_message="devtunnelにログインが必要です。\n"
                "ログイン: devtunnel user login",
                status=TunnelStatus.NOT_LOGGED_IN,
            )

        self._stop_event.clear()
        self._url_ready_event.clear()
        self._set_status(TunnelStatus.STARTING)

        try:
            # devtunnel host コマンドを実行
            cmd = ["devtunnel", "host", "-p", str(self.port), "--allow-anonymous"]

            print(f"[DevTunnel] 起動コマンド: {' '.join(cmd)}")

            # Windows用のフラグ
            creation_flags = 0
            if os.name == "nt":
                creation_flags = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # stderrをstdoutにマージ
                text=True,
                bufsize=1,  # 行バッファリング
                creationflags=creation_flags,
            )

            # 出力監視スレッドを開始
            self.monitor_thread = threading.Thread(
                target=self._monitor_output, daemon=True
            )
            self.monitor_thread.start()

            # URLが取得されるまで待機（最大30秒）
            if self._url_ready_event.wait(timeout=30):
                return TunnelResult(
                    success=True, url=self.url, status=TunnelStatus.RUNNING
                )
            else:
                # タイムアウト - プロセスを停止
                error_msg = "URLの取得がタイムアウトしました（30秒）"
                self._set_status(TunnelStatus.ERROR, error_msg)
                self.stop()
                return TunnelResult(
                    success=False,
                    error_message=error_msg,
                    status=TunnelStatus.ERROR,
                )

        except FileNotFoundError:
            self._set_status(TunnelStatus.NOT_INSTALLED)
            return TunnelResult(
                success=False,
                error_message="devtunnelコマンドが見つかりません",
                status=TunnelStatus.NOT_INSTALLED,
            )
        except Exception as e:
            self._set_status(TunnelStatus.ERROR, str(e))
            return TunnelResult(
                success=False, error_message=str(e), status=TunnelStatus.ERROR
            )

    def _monitor_output(self):
        """stdoutを監視してURLを抽出する"""
        if self.process is None or self.process.stdout is None:
            return

        url_found = False  # 最初のURLのみ取得

        try:
            for line in self.process.stdout:
                if self._stop_event.is_set():
                    break

                line = line.strip()
                if not line:
                    continue

                print(f"[DevTunnel] {line}")

                # ログインが必要かチェック
                if self.LOGIN_REQUIRED_PATTERN.search(line):
                    self._set_status(
                        TunnelStatus.NOT_LOGGED_IN, "devtunnelへのログインが必要です"
                    )
                    self._url_ready_event.set()  # 待機を解除
                    continue

                # "Connect via browser:" 行からURLを抽出（最初のURLのみ）
                if not url_found:
                    match = self.CONNECT_URL_PATTERN.search(line)
                    if match:
                        self.url = match.group(1).rstrip("/")
                        self._set_status(TunnelStatus.RUNNING)
                        print(f"[DevTunnel] 接続URL取得成功: {self.url}")
                        if self.on_url_ready:
                            self.on_url_ready(self.url)
                        self._url_ready_event.set()
                        url_found = True

        except Exception as e:
            print(f"[DevTunnel] 監視エラー: {e}")
            self._set_status(TunnelStatus.ERROR, str(e))
            self._url_ready_event.set()

    def stop(self) -> bool:
        """Dev Tunnelプロセスを停止する"""
        self._stop_event.set()

        if self.process is not None:
            try:
                self.process.terminate()
                # 最大5秒待機
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

                print("[DevTunnel] プロセス停止")
            except Exception as e:
                print(f"[DevTunnel] 停止エラー: {e}")
            finally:
                self.process = None

        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=2)
            self.monitor_thread = None

        self.url = ""
        self._set_status(TunnelStatus.STOPPED)
        return True

    def is_running(self) -> bool:
        """プロセスが実行中かどうか"""
        return self.process is not None and self.process.poll() is None
