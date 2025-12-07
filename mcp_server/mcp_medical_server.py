"""
医療トリアージ MCP サーバー

HTTP JSON-RPCプロトコルでMCPツールを公開する。
Azure AI Foundry AgentからDev Tunnel経由でアクセス可能。

参照実装: https://github.com/olivierb123/hybrid-ai-mcp-localhealthcoach
"""

import json
import os
import re
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
import threading

import requests
from dotenv import load_dotenv

from .mcp_state import MCPRequestLog, MCPState, ServerStatus, TunnelStatus, mcp_state
from .dev_tunnel import DevTunnelManager

# .env ファイルから環境変数を読み込み
load_dotenv()

# MCPサーバー設定
MCP_SERVER_PORT = 8081
MCP_SERVER_NAME = "LocalMedicalContextServer"
MCP_PROTOCOL_VERSION = "2025-06-18"

# Foundry Local設定
# 環境変数 FOUNDRY_LOCAL_URL から読み取り（`foundry service status` で確認可能）
FOUNDRY_LOCAL_BASE = os.getenv("FOUNDRY_LOCAL_URL", "http://127.0.0.1:5273")
FOUNDRY_LOCAL_CHAT_URL = FOUNDRY_LOCAL_BASE + "/v1/chat/completions"
FOUNDRY_LOCAL_MODEL_ID = "phi-4-mini-instruct-vitis-npu:2"

# 患者データのパス
MEDICAL_DATA_DIR = Path(__file__).parent.parent / "medical" / "data"


# ========= ローカルLLM用プロンプト =========

LOCAL_PATIENT_SYSTEM_PROMPT = """
あなたはユーザーのマシン上でローカルに動作する医療背景情報要約器です。

【絶対ルール】
- 出力はJSON構造のみ。それ以外は一切出力禁止。
- 説明文、注釈、コメント、マークダウン記法は禁止。
- バッククォート(```)で囲まない。
- 回答は必ず { で始まり } で終わること。

【プライバシー保護】
- 患者の具体的な氏名、住所、電話番号は絶対に出力しない
- 年齢は年代（30代、40代など）で表現
- 匿名化された医療情報のみを出力

【出力サンプル】
{"patient_context":{"age_group":"30代","chronic_conditions":["軽度喘息"],"allergies":["ペニシリン"],"current_medications":["アレグラ（抗ヒスタミン薬）"]},"symptom_relevance":{"related_history":["過去の呼吸器症状"],"potential_interactions":[],"risk_factors":["喘息の既往歴あり"]},"recommendations":["呼吸器症状には注意が必要","ペニシリン系抗生物質は禁忌"]}

【出力スキーマ】
- patient_context: 患者背景（age_group, chronic_conditions, allergies, current_medications）
- symptom_relevance: 症状との関連性（related_history, potential_interactions, risk_factors）
- recommendations: 医療従事者への推奨事項（配列）

症状と患者の医療背景を照らし合わせ、関連性のある情報のみを抽出してください。
""".strip()


def strip_code_fences(text: str) -> str:
    """コードフェンス（```json ... ```）を除去"""
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json_response(text: str) -> dict:
    """JSON文字列をパース（コードフェンス対応）"""
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw_text": cleaned[:500]}


def load_patient_pii() -> dict:
    """患者個人情報を読み込む"""
    pii_file = MEDICAL_DATA_DIR / "patient_pii.json"
    if pii_file.exists():
        return json.loads(pii_file.read_text(encoding="utf-8"))
    return {}


def load_patient_medical() -> dict:
    """患者医療情報を読み込む"""
    medical_file = MEDICAL_DATA_DIR / "patient_medical.json"
    if medical_file.exists():
        return json.loads(medical_file.read_text(encoding="utf-8"))
    return {}


def load_health_checkup_history() -> dict:
    """健康診断履歴を読み込む"""
    history_file = MEDICAL_DATA_DIR / "health_checkup_history.json"
    if history_file.exists():
        return json.loads(history_file.read_text(encoding="utf-8"))
    return {}


def format_flag(flag: str) -> str:
    """検査結果フラグを記号に変換"""
    if flag == "high":
        return "↑"
    elif flag == "low":
        return "↓"
    elif flag == "borderline":
        return "△"
    return ""


def build_patient_context() -> str:
    """
    個人情報、医療情報、健康診断履歴を組み合わせてコンテキストを構築。
    秘匿すべき個人情報を明示的にプロンプトに埋め込む。
    """
    pii = load_patient_pii()
    medical = load_patient_medical()
    history = load_health_checkup_history()

    context_parts = []

    # === 個人情報セクション（秘匿対象） ===
    if pii:
        patient = pii.get("patient", {})
        provider = pii.get("medical_provider", {})

        context_parts.append("【患者情報 - 秘匿対象】")
        context_parts.append(f"氏名: {patient.get('full_name', '不明')}")
        context_parts.append(f"年齢: {patient.get('age', '不明')}歳")
        context_parts.append(f"性別: {patient.get('gender', '不明')}")
        context_parts.append(f"生年月日: {patient.get('date_of_birth', '不明')}")
        context_parts.append(f"患者ID: {patient.get('patient_id', '不明')}")

        if patient.get("address"):
            addr = patient["address"]
            context_parts.append(f"住所: 〒{addr.get('postal_code', '')} {addr.get('prefecture', '')}{addr.get('city', '')}{addr.get('street', '')}")

        if patient.get("phone"):
            context_parts.append(f"電話番号: {patient['phone']}")

        if patient.get("emergency_contact"):
            ec = patient["emergency_contact"]
            context_parts.append(f"緊急連絡先: {ec.get('name', '')}（{ec.get('relationship', '')}）{ec.get('phone', '')}")

        if provider:
            context_parts.append(f"\n【医療機関情報 - 秘匿対象】")
            context_parts.append(f"医療機関: {provider.get('facility_name', '不明')}")
            context_parts.append(f"担当医: {provider.get('doctor_name', '不明')}")
            context_parts.append(f"医籍番号: {provider.get('license_number', '不明')}")

    # === 現在の医療情報セクション ===
    if medical:
        context_parts.append("\n【現在の医療情報】")
        context_parts.append(f"主訴: {medical.get('chief_complaint', '不明')}")

        vitals = medical.get("vital_signs", {})
        if vitals.get("temperature"):
            context_parts.append(f"体温: {vitals['temperature']}度")
        if vitals.get("pulse"):
            context_parts.append(f"脈拍: {vitals['pulse']}bpm")
        if vitals.get("blood_pressure"):
            bp = vitals["blood_pressure"]
            if isinstance(bp, dict):
                context_parts.append(f"血圧: {bp.get('systolic', '')}/{bp.get('diastolic', '')}mmHg")

        # アレルギー
        allergies = medical.get("allergies", [])
        if allergies:
            context_parts.append(f"アレルギー: {', '.join(allergies)}")

        # 服用中の薬
        medications = medical.get("current_medications", [])
        if medications:
            context_parts.append(f"服用中の薬: {', '.join(medications)}")
        else:
            context_parts.append("服用中の薬: なし")

        # 既往歴
        medical_history = medical.get("medical_history", [])
        if medical_history:
            context_parts.append("\n【既往歴 - 重要】")
            for history_item in medical_history:
                if isinstance(history_item, dict):
                    condition = history_item.get("condition", "不明")
                    diagnosed = history_item.get("diagnosed", "")
                    status = history_item.get("status", "")
                    treatment = history_item.get("treatment", "")
                    notes = history_item.get("notes", "")
                    context_parts.append(f"  疾患: {condition}")
                    if diagnosed:
                        context_parts.append(f"  診断時期: {diagnosed}")
                    if status:
                        context_parts.append(f"  現在の状態: {status}")
                    if treatment:
                        context_parts.append(f"  治療: {treatment}")
                    if notes:
                        context_parts.append(f"  備考: {notes}")

        # 検査結果
        lab = medical.get("lab_results", {})
        if lab:
            context_parts.append("\n【検査結果】")

            # 甲状腺機能検査（重要）
            thyroid = lab.get("thyroid_function", {})
            if thyroid:
                context_parts.append("甲状腺機能検査:")
                for key, data in thyroid.items():
                    if isinstance(data, dict):
                        flag = format_flag(data.get("flag", ""))
                        context_parts.append(f"  {key}: {data.get('value')} {data.get('unit', '')} {flag}")

            # CBC
            cbc = lab.get("cbc", {})
            if cbc:
                context_parts.append("血球計算 (CBC):")
                for key, data in cbc.items():
                    if isinstance(data, dict):
                        flag = format_flag(data.get("flag", ""))
                        context_parts.append(f"  {key}: {data.get('value')} {data.get('unit', '')} {flag}")

            # 炎症マーカー
            inflammatory = lab.get("inflammatory_markers", {})
            if inflammatory:
                context_parts.append("炎症マーカー:")
                for key, data in inflammatory.items():
                    if isinstance(data, dict):
                        flag = format_flag(data.get("flag", ""))
                        context_parts.append(f"  {key}: {data.get('value')} {data.get('unit', '')} {flag}")

            # BMP
            bmp = lab.get("bmp", {})
            if bmp:
                context_parts.append("生化学検査:")
                for key, data in bmp.items():
                    if isinstance(data, dict):
                        flag = format_flag(data.get("flag", ""))
                        context_parts.append(f"  {key}: {data.get('value')} {data.get('unit', '')} {flag}")

        # 臨床所見
        notes = medical.get("clinical_notes")
        if notes:
            context_parts.append(f"\n臨床所見: {notes}")

    # === 健康診断トレンド（超コンパクト版） ===
    if history:
        trend = history.get("trend_summary", {})
        if trend:
            parts = []

            # 甲状腺TSHトレンド
            tsh = trend.get("thyroid_tsh", {})
            if tsh.get("2020") and tsh.get("2024"):
                tsh_trend = tsh.get("trend", "")
                parts.append(f"TSH:{tsh['2020']}→{tsh['2024']}({tsh_trend[:10]})")

            # 甲状腺FT4トレンド
            ft4 = trend.get("thyroid_ft4", {})
            if ft4.get("2020") and ft4.get("2024"):
                parts.append(f"FT4:{ft4['2020']}→{ft4['2024']}")

            # TRAb（甲状腺自己抗体）トレンド
            trab = trend.get("thyroid_trab", {})
            if trab.get("2020") and trab.get("2022"):
                trab_trend = trab.get("trend", "")
                parts.append(f"TRAb:{trab['2020']}→{trab['2022']}({trab_trend[:10]})")

            # 体重トレンド
            wt = trend.get("weight", {})
            if wt.get("2020") and wt.get("2024"):
                diff = wt.get("2024", 0) - wt.get("2020", 0)
                parts.append(f"体重:{wt['2020']}→{wt['2024']}kg({diff:+.1f})")

            # 血糖トレンド
            gl = trend.get("glucose_fasting", {})
            if gl.get("2020") and gl.get("2024"):
                status = "正常化" if gl.get("2024", 999) < 100 else ""
                parts.append(f"血糖:{gl['2020']}→{gl['2024']}{status}")

            # LDLトレンド
            ldl = trend.get("ldl_cholesterol", {})
            if ldl.get("2020") and ldl.get("2024"):
                status = "正常化" if ldl.get("2024", 999) < 120 else ""
                parts.append(f"LDL:{ldl['2020']}→{ldl['2024']}{status}")

            # 総合評価
            assessment = trend.get("overall_assessment", "")
            if assessment:
                # 最初の50文字を抽出
                context_parts.append(f"\n【総合評価】{assessment[:80]}...")

            if parts:
                context_parts.append(f"【健診トレンド(5年)】{' '.join(parts)}")

    if not context_parts:
        return "患者データが見つかりません。"

    return "\n".join(context_parts)


def summarize_patient_locally(symptoms: str, state: MCPState) -> dict:
    """
    Foundry Local (Phi-4-mini) を使用して患者背景を要約する。

    Args:
        symptoms: ユーザーが報告した症状
        state: MCPサーバー状態（ログ記録用）

    Returns:
        匿名化された患者背景情報のJSON
    """
    # 患者データを読み込み（新スキーマ対応）
    patient_data = build_patient_context()

    # LLMへの入力を構築
    user_content = f"【報告された症状】\n{symptoms}\n\n{patient_data}"

    # リクエストペイロード
    payload = {
        "model": FOUNDRY_LOCAL_MODEL_ID,
        "messages": [
            {"role": "system", "content": LOCAL_PATIENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 2048,
        "temperature": 0.2,
    }

    # === Foundry Local INPUT 詳細ログ（デバッグ用） ===
    print(f"\n{'='*60}")
    print(f"[Foundry Local] INPUT - get_patient_background")
    print(f"{'='*60}")
    print(f"[System Message]")
    print(LOCAL_PATIENT_SYSTEM_PROMPT)
    print(f"\n[User Message]")
    print(user_content)
    print(f"{'='*60}\n")

    try:
        resp = requests.post(
            FOUNDRY_LOCAL_CHAT_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=600,
        )
        resp.raise_for_status()
        response_json = resp.json()

        # コンテンツを抽出
        content = response_json["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

        # === Foundry Local OUTPUT 詳細ログ（デバッグ用） ===
        print(f"\n{'='*60}")
        print(f"[Foundry Local] OUTPUT - get_patient_background")
        print(f"{'='*60}")
        print(content)
        print(f"{'='*60}\n")

        # JSONをパース
        result = parse_json_response(content)
        return result, user_content, content

    except requests.RequestException as e:
        print(f"[Foundry Local] 接続エラー: {str(e)}")
        return {"error": f"Foundry Local connection failed: {str(e)}"}, user_content, None
    except (KeyError, IndexError) as e:
        print(f"[Foundry Local] パースエラー: {str(e)}")
        return {"error": f"Response parsing failed: {str(e)}"}, user_content, None


def handle_mcp_request(request_data: dict, state: MCPState) -> dict:
    """
    MCP JSON-RPCリクエストを処理する。

    サポートするメソッド:
    - initialize: サーバー情報を返却
    - tools/list: 利用可能なツール一覧を返却
    - tools/call: ツールを実行して結果を返却
    """
    method = request_data.get("method", "")
    request_id = request_data.get("id")
    params = request_data.get("params", {})

    print(f"\n{'='*60}")
    print(f"[MCP Server] リクエスト受信: method={method}")

    # リクエストログを作成
    log = MCPRequestLog(
        timestamp=datetime.now(),
        method=method,
        request_id=str(request_id) if request_id else None,
    )
    state.add_request_log(log)
    start_time = time.time()

    try:
        if method == "initialize":
            result = {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": {
                    "name": MCP_SERVER_NAME,
                    "version": "1.0.0",
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                },
            }

        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "get_patient_background",
                        "description": (
                            "患者の医療背景情報を取得します。"
                            "アレルギー、既往歴、服用中の薬、関連する検査結果などを"
                            "ローカルで処理し、匿名化されたサマリーを返します。"
                            "機密性の高い患者データはローカルに留まり、"
                            "クラウドには送信されません。"
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "symptoms": {
                                    "type": "string",
                                    "description": "ユーザーが報告した症状の説明",
                                }
                            },
                            "required": ["symptoms"],
                        },
                    }
                ]
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            log.tool_name = tool_name
            log.tool_arguments = tool_args

            print(f"[MCP Server] ツール実行: {tool_name}")
            print(f"[MCP Server] 引数: {tool_args}")

            if tool_name == "get_patient_background":
                symptoms = tool_args.get("symptoms", "")
                print(f"[MCP Server] 患者データ読み込み中...")
                print(f"[MCP Server] Foundry Local (Phi-4-mini) 呼び出し中...")
                patient_summary, llm_input, llm_output = summarize_patient_locally(
                    symptoms, state
                )
                print(f"[MCP Server] 匿名化サマリー生成完了")

                # ログにLLM入出力を記録
                log.llm_input = llm_input
                log.llm_output = llm_output

                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(patient_summary, ensure_ascii=False, indent=2),
                        }
                    ]
                }
            else:
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                        }
                    ],
                    "isError": True,
                }

        else:
            result = {"error": {"code": -32601, "message": f"Method not found: {method}"}}

        # レスポンスを構築
        response = {"jsonrpc": "2.0", "id": request_id, "result": result}

        # ログを更新
        duration_ms = (time.time() - start_time) * 1000
        state.update_request_log(log, response=result, duration_ms=duration_ms)

        print(f"[MCP Server] レスポンス送信: {duration_ms:.1f}ms")
        print(f"{'='*60}\n")

        return response

    except Exception as e:
        print(f"[MCP Server] エラー発生: {str(e)}")
        print(f"{'='*60}\n")
        error_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": str(e)},
        }
        state.update_request_log(log, error=str(e))
        return error_response


class MCPHandler(BaseHTTPRequestHandler):
    """HTTP MCPリクエストハンドラー"""

    def __init__(self, *args, state: MCPState = None, **kwargs):
        self.state = state or mcp_state
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """デフォルトのログ出力を抑制"""
        pass

    def _send_response_safe(self, status_code: int, headers: dict, body: bytes):
        """レスポンスを安全に送信（接続エラーをハンドリング）"""
        try:
            self.send_response(status_code)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError) as e:
            print(f"[MCP Server] 接続エラー（クライアント切断）: {str(e)}")

    def do_GET(self):
        """GETリクエスト: ヘルスチェック用"""
        response = {
            "status": "running",
            "server": MCP_SERVER_NAME,
            "protocol_version": MCP_PROTOCOL_VERSION,
        }
        self._send_response_safe(
            200,
            {"Content-Type": "application/json"},
            json.dumps(response).encode("utf-8")
        )

    def do_POST(self):
        """POSTリクエスト: MCP JSON-RPC処理"""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError) as e:
            print(f"[MCP Server] リクエスト読み取りエラー: {str(e)}")
            return

        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            self._send_response_safe(
                400,
                {"Content-Type": "application/json"},
                b'{"error": "Invalid JSON"}'
            )
            return

        # MCPリクエストを処理
        response = handle_mcp_request(request_data, self.state)

        self._send_response_safe(
            200,
            {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            json.dumps(response, ensure_ascii=False).encode("utf-8")
        )

    def do_OPTIONS(self):
        """OPTIONSリクエスト: CORS対応"""
        self._send_response_safe(
            200,
            {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            b""
        )


class MCPServer:
    """MCPサーバー管理クラス"""

    def __init__(self, port: int = MCP_SERVER_PORT, state: MCPState = None):
        self.port = port
        self.state = state or mcp_state
        self.state.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.tunnel_manager: Optional[DevTunnelManager] = None

    def _create_handler(self):
        """カスタムハンドラーファクトリ"""
        state = self.state

        class CustomHandler(MCPHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, state=state, **kwargs)

        return CustomHandler

    def _on_tunnel_url_ready(self, url: str):
        """Dev Tunnel URLが取得された時のコールバック"""
        self.state.set_tunnel_url(url, auto_started=True)
        print(f"[MCP] Dev Tunnel URL自動設定: {url}")

    def _on_tunnel_status_change(self, status: TunnelStatus, error: str):
        """Dev Tunnel状態変更時のコールバック"""
        self.state.set_tunnel_status(status, error)

    def _start_tunnel(self) -> bool:
        """Dev Tunnelを起動"""
        if self.tunnel_manager is None:
            self.tunnel_manager = DevTunnelManager(
                port=self.port,
                on_url_ready=self._on_tunnel_url_ready,
                on_status_change=self._on_tunnel_status_change,
            )

        result = self.tunnel_manager.start()

        if not result.success:
            print(f"[MCP] Dev Tunnel起動失敗: {result.error_message}")
            # エラーはstateに記録済み（コールバック経由）

        return result.success

    def start(self, start_tunnel: bool = True) -> bool:
        """
        サーバーを起動

        Args:
            start_tunnel: Dev Tunnelも同時に起動するかどうか
        """
        if self.server is not None:
            return False

        try:
            self.state.set_status(ServerStatus.STARTING)
            self.server = HTTPServer(("0.0.0.0", self.port), self._create_handler())
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            self.state.set_status(ServerStatus.RUNNING)
            print(f"[MCP] Listening at http://0.0.0.0:{self.port}")

            # Dev Tunnelを自動起動
            if start_tunnel:
                self._start_tunnel()

            return True
        except Exception as e:
            self.state.set_status(ServerStatus.ERROR, str(e))
            return False

    def stop(self) -> bool:
        """サーバーを停止"""
        # Dev Tunnelを先に停止
        if self.tunnel_manager is not None:
            self.tunnel_manager.stop()
            self.tunnel_manager = None

        if self.server is None:
            return False

        try:
            self.server.shutdown()
            self.server = None
            self.thread = None
            self.state.set_status(ServerStatus.STOPPED)
            print("[MCP] Server stopped")
            return True
        except Exception as e:
            self.state.set_status(ServerStatus.ERROR, str(e))
            return False

    def is_running(self) -> bool:
        """サーバーが実行中かどうか"""
        return self.server is not None and self.state.is_running()


# グローバルサーバーインスタンス
_server_instance: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """MCPサーバーインスタンスを取得"""
    global _server_instance
    if _server_instance is None:
        _server_instance = MCPServer()
    return _server_instance


def run_mcp_server(port: int = MCP_SERVER_PORT) -> MCPServer:
    """MCPサーバーを起動して返す"""
    server = get_mcp_server()
    server.port = port
    server.state.port = port
    server.start()
    return server


def stop_mcp_server() -> bool:
    """MCPサーバーを停止"""
    global _server_instance
    if _server_instance is not None:
        return _server_instance.stop()
    return False


if __name__ == "__main__":
    # スタンドアロン実行
    print(f"Starting {MCP_SERVER_NAME} on port {MCP_SERVER_PORT}...")
    server = run_mcp_server()

    try:
        # メインスレッドを維持
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_mcp_server()
