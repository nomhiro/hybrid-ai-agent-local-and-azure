"""
Streamlit UI for Hybrid AI Agent System

ローカルLLM（Foundry Local）とクラウドLLM（Azure AI）を組み合わせた
ハイブリッドAIエージェントのUIを提供する。

MCPサーバー制御パネルとログ表示機能を含む。

実行方法:
    streamlit run app.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from azure.identity.aio import AzureCliCredential

from agent_framework import ChatAgent, MCPStreamableHTTPTool, FunctionResultContent, TextContent
from agent_framework.azure import AzureAIAgentClient

from common.llm_logger import LLMLogEntry, llm_logger
from common.prompt_loader import get_prompts_for_agent
from medical.agent import SYMPTOM_CHECKER_INSTRUCTIONS

# MCPサーバー関連のインポート（医療）
from mcp_server.mcp_state import mcp_state, ServerStatus, TunnelStatus
from mcp_server.mcp_medical_server import (
    get_mcp_server,
    run_mcp_server,
    stop_mcp_server,
    MCP_SERVER_PORT,
)


def append_runtime_data(message: str) -> str:
    """
    ユーザーメッセージをそのまま返す。

    機密データ（検査結果など）はローカルツール内で
    直接ファイルから読み込むため、ここでは追記しない。
    これにより、Azure AIエージェントには個人情報が送信されない。

    Args:
        message: ユーザーが入力したメッセージ

    Returns:
        ユーザーメッセージ（機密データなし）
    """
    # 機密データはローカルツール内で直接読み込むため、
    # ユーザーメッセージはそのまま返す
    return message


def create_log_callback(placeholder):
    """Streamlitプレースホルダーに書き込むコールバックを作成"""
    log_display = []

    def callback(entry: LLMLogEntry):
        # 入力フェーズ（response_textがまだない場合）
        if entry.response_text is None:
            log_display.append(f"### {entry.tool_name}")
            log_display.append("**System Prompt:**")
            # プロンプトが長い場合は切り詰め
            prompt_preview = entry.system_prompt[:500]
            if len(entry.system_prompt) > 500:
                prompt_preview += "..."
            log_display.append(f"```\n{prompt_preview}\n```")
            log_display.append("**User Content (Input):**")
            # 入力が長い場合は切り詰め
            input_preview = entry.user_content[:800]
            if len(entry.user_content) > 800:
                input_preview += "..."
            log_display.append(f"```\n{input_preview}\n```")
        else:
            # 出力フェーズ
            log_display.append("**Output:**")
            if entry.parsed_result:
                output_json = json.dumps(
                    entry.parsed_result, indent=2, ensure_ascii=False
                )
                # 出力が長い場合は切り詰め
                if len(output_json) > 1500:
                    output_json = output_json[:1500] + "\n..."
                log_display.append(f"```json\n{output_json}\n```")
            else:
                log_display.append(f"```\n{entry.response_text[:1000]}\n```")
            log_display.append("---")

        placeholder.markdown("\n".join(log_display))

    return callback


async def run_agent_stream(agent_type: str, user_message: str, placeholders: dict):
    """エージェントをストリーミング実行"""

    print(f"\n{'='*60}")
    print(f"[Streamlit] エージェント実行開始: {agent_type}")
    print(f"[Streamlit] ユーザーメッセージ: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")

    # 実行時に機密データを追記（ユーザーには見せないデータを埋め込む）
    expanded_message = append_runtime_data(user_message)

    # ロガーにコールバックを設定
    llm_logger.clear()
    llm_logger.set_callback(create_log_callback(placeholders["local_llm"]))

    # エージェント設定（医療診断）
    instructions = SYMPTOM_CHECKER_INSTRUCTIONS
    # MCPツールを使用（Dev Tunnel経由でローカルMCPサーバーに接続）
    # これにより、個人情報（患者データ）はローカルで処理され、
    # クラウドには匿名化されたサマリーのみが送信される
    mcp_url = mcp_state.tunnel_url
    if not mcp_url:
        placeholders["azure_llm"].error(
            "⚠️ 医療エージェントを使用するにはDev Tunnel URLを設定してください。\n\n"
            "1. MCPサーバーを起動\n"
            "2. Dev Tunnelを起動: `devtunnel host --port-numbers 8081`\n"
            "3. サイドバーにDev Tunnel URLを入力"
        )
        print(f"[Streamlit] エラー: Dev Tunnel URLが設定されていません")
        print(f"{'='*60}\n")
        return None
    print(f"[Streamlit] MCPツール使用: URL={mcp_url}")
    tools = MCPStreamableHTTPTool(
        name="LocalMedicalContext",
        url=mcp_url,
        timeout=600,           # HTTP POST: 600秒（10分）
        sse_read_timeout=600   # SSE読取: 600秒（10分）
    )
    name = "hybrid-symptom-checker"

    # --- システムメッセージとユーザーメッセージを先に表示 ---
    input_display = []
    input_display.append("### System Prompt (Instructions)")
    prompt_preview = instructions[:500]
    if len(instructions) > 500:
        prompt_preview += "..."
    input_display.append(f"```\n{prompt_preview}\n```")

    input_display.append("### User Message")
    user_preview = expanded_message[:800]
    if len(expanded_message) > 800:
        user_preview += "..."
    input_display.append(f"```\n{user_preview}\n```")

    input_display.append("---")
    input_display.append("### Response")

    base_display = "\n".join(input_display)
    placeholders["azure_llm"].markdown(base_display)

    print(f"[Streamlit] Azure AI Foundry Agent 作成中... (name={name})")

    async with (
        AzureCliCredential() as credential,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            instructions=instructions,
            tools=tools,
            name=name,
        ) as agent,
    ):
        print(f"[Streamlit] Azure AI へリクエスト送信...")
        full_text = ""
        tool_calls_displayed = []
        tool_results_displayed = []  # ツール結果表示済みのcall_idを追跡

        async for update in agent.run_stream(expanded_message):
            # ストリーミングテキストを表示
            if update.text:
                full_text += update.text
                placeholders["azure_llm"].markdown(base_display + "\n" + full_text + "▌")

            # ツール呼び出しを検出して表示
            for content in update.contents or []:
                if hasattr(content, "name") and hasattr(content, "call_id"):
                    # FunctionCallContent
                    call_id = getattr(content, "call_id", "")
                    if call_id not in tool_calls_displayed:
                        tool_calls_displayed.append(call_id)
                        print(f"[Streamlit] ツール呼び出し検出: {content.name}")
                        try:
                            args = content.parse_arguments()
                            args_json = json.dumps(args, indent=2, ensure_ascii=False)
                            print(f"[Streamlit] ツール引数: {args_json[:200]}{'...' if len(args_json) > 200 else ''}")
                        except Exception:
                            args_json = str(getattr(content, "arguments", ""))

                        placeholders["tool_calls"].info(
                            f"**Tool:** {content.name}\n\n"
                            f"**Arguments:**\n```json\n{args_json[:500]}\n```"
                        )

                # FunctionResultContent（ツール実行結果）を検出して表示
                elif isinstance(content, FunctionResultContent):
                    call_id = getattr(content, "call_id", "")
                    if call_id not in tool_results_displayed:
                        tool_results_displayed.append(call_id)

                        result_data = content.result
                        exception_data = content.exception

                        print(f"[Streamlit] ツール実行結果受信: call_id={call_id}")

                        # 結果のフォーマット
                        if exception_data:
                            result_display = f"Error: {str(exception_data)}"
                        else:
                            if isinstance(result_data, str):
                                try:
                                    parsed = json.loads(result_data)
                                    result_display = json.dumps(parsed, indent=2, ensure_ascii=False)
                                except json.JSONDecodeError:
                                    result_display = result_data
                            elif isinstance(result_data, dict):
                                result_display = json.dumps(result_data, indent=2, ensure_ascii=False)
                            elif isinstance(result_data, list):
                                # TextContentオブジェクトのリストを処理
                                texts = []
                                for item in result_data:
                                    if hasattr(item, 'text'):
                                        texts.append(item.text)
                                    else:
                                        texts.append(str(item))
                                combined_text = "\n".join(texts)
                                try:
                                    parsed = json.loads(combined_text)
                                    result_display = json.dumps(parsed, indent=2, ensure_ascii=False)
                                except json.JSONDecodeError:
                                    result_display = combined_text
                            else:
                                result_display = str(result_data) if result_data else "(empty)"

                        print(f"[Streamlit] ツール結果: {result_display[:200]}{'...' if len(result_display) > 200 else ''}")

                        # UI表示
                        with placeholders["tool_calls"]:
                            st.markdown("---")
                            st.markdown("**MCPツール応答:**")
                            if len(result_display) > 500:
                                with st.expander("詳細を表示", expanded=False):
                                    st.code(result_display, language="json")
                            else:
                                st.code(result_display[:1000], language="json")

        # 最終結果（カーソルなし）
        placeholders["azure_llm"].markdown(base_display + "\n" + full_text)
        print(f"[Streamlit] 処理完了")
        print(f"{'='*60}\n")
        return full_text


def render_mcp_server_panel():
    """MCPサーバー制御パネルを表示"""
    st.subheader("🌐 MCPサーバー (医療診断)")

    # サーバー状態の表示
    status = mcp_state.status
    if status == ServerStatus.RUNNING:
        st.success(f"✅ 実行中 - ポート {mcp_state.port}")
    elif status == ServerStatus.STARTING:
        st.warning("⏳ 起動中...")
    elif status == ServerStatus.ERROR:
        st.error(f"❌ エラー: {mcp_state.error_message}")
    else:
        st.info("⏹️ 停止中")

    # 起動/停止ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "▶️ 起動",
            disabled=(status == ServerStatus.RUNNING),
            use_container_width=True,
        ):
            run_mcp_server()
            st.rerun()

    with col2:
        if st.button(
            "⏹️ 停止",
            disabled=(status != ServerStatus.RUNNING),
            use_container_width=True,
        ):
            stop_mcp_server()
            st.rerun()

    # URL表示
    if status == ServerStatus.RUNNING:
        st.markdown("**ローカルURL:**")
        st.code(mcp_state.local_url, language=None)

        # Dev Tunnel状態表示
        st.markdown("**Dev Tunnel:**")
        tunnel_status = mcp_state.tunnel_status

        if tunnel_status == TunnelStatus.RUNNING:
            st.success("✅ 接続済み")
        elif tunnel_status == TunnelStatus.STARTING:
            st.warning("⏳ 接続中...")
        elif tunnel_status == TunnelStatus.NOT_INSTALLED:
            st.error("❌ devtunnel CLIがインストールされていません")
            st.caption("インストール: `winget install Microsoft.devtunnel`")
        elif tunnel_status == TunnelStatus.NOT_LOGGED_IN:
            st.error("❌ devtunnelへのログインが必要です")
            st.caption("ログイン: `devtunnel user login`")
        elif tunnel_status == TunnelStatus.ERROR:
            st.error(f"❌ エラー: {mcp_state.tunnel_error}")
        else:
            st.info("⏹️ 未起動")

        # Dev Tunnel URL表示
        st.markdown("**Dev Tunnel URL:**")

        # 自動取得されたURLがあれば表示
        if mcp_state.tunnel_url and mcp_state.tunnel_auto_started:
            st.code(mcp_state.tunnel_url, language=None)
            st.caption("✅ 自動取得されました")
        else:
            # 手動入力欄
            tunnel_url = st.text_input(
                "Dev Tunnel URLを入力（手動）",
                value=mcp_state.tunnel_url,
                placeholder="https://<tunnel-id>.devtunnels.ms",
                label_visibility="collapsed",
            )
            if tunnel_url != mcp_state.tunnel_url:
                mcp_state.set_tunnel_url(tunnel_url)

            st.caption(
                "自動取得に失敗した場合、手動でURLを入力できます。\n"
                "→ [Dev Tunnelセットアップ手順](docs/dev-tunnel-setup.md)"
            )


def render_mcp_logs():
    """MCPリクエストログを表示"""
    st.subheader("📋 MCPリクエストログ")

    # デバッグ: 現在のログ数を表示
    st.caption(f"ログ数: {len(mcp_state.request_logs)} | サーバー: {mcp_state.status.value}")

    # 更新ボタンとクリアボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 更新", key="refresh_mcp_logs"):
            st.rerun()
    with col2:
        if st.button("🗑️ クリア", key="clear_mcp_logs"):
            mcp_state.clear_logs()
            st.rerun()

    logs = mcp_state.get_recent_logs(10)

    if not logs:
        st.info("まだリクエストがありません。")
        return

    # ログを逆順で表示（新しいものが上）
    for log in reversed(logs):
        with st.expander(
            f"{log.timestamp.strftime('%H:%M:%S')} - {log.method}",
            expanded=False,
        ):
            st.markdown(f"**メソッド:** `{log.method}`")

            if log.tool_name:
                st.markdown(f"**ツール:** `{log.tool_name}`")

            if log.tool_arguments:
                st.markdown("**引数:**")
                st.json(log.tool_arguments)

            if log.llm_input:
                st.markdown("**Foundry Local 入力:**")
                if len(log.llm_input) > 500:
                    with st.expander("入力内容を表示", expanded=False):
                        st.code(log.llm_input, language=None)
                else:
                    st.code(log.llm_input, language=None)

            if log.llm_output:
                st.markdown("**Foundry Local 出力:**")
                if len(log.llm_output) > 800:
                    with st.expander("出力内容を表示", expanded=False):
                        st.code(log.llm_output, language=None)
                else:
                    st.code(log.llm_output, language=None)

            if log.response:
                st.markdown("**レスポンス:**")
                try:
                    st.json(log.response)
                except Exception:
                    st.code(str(log.response)[:500], language=None)

            if log.error:
                st.error(f"エラー: {log.error}")

            if log.duration_ms:
                st.caption(f"処理時間: {log.duration_ms:.1f}ms")


def main():
    st.set_page_config(page_title="Hybrid AI Agent", page_icon="🤖", layout="wide")
    st.title("🤖 Hybrid AI Agent System")
    st.caption("Local LLM (Foundry Local) + Cloud LLM (Azure AI) + MCP Server")

    # 医療診断エージェントのみ
    agent_type = "medical"

    # セッションステートの初期化
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # サイドバー
    with st.sidebar:
        st.header("設定")

        # MCPサーバー制御パネル（医療）
        render_mcp_server_panel()

        st.divider()
        st.markdown(
            """
        ### 使い方
        1. MCPサーバーを起動（ポート8081）
        2. Dev Tunnelを起動してURLを入力
        3. プロンプトを選択して読み込む
        4. 「実行」をクリック

        ### 表示内容
        - **MCPサーバー**: サーバー状態とログ
        - **ツール呼び出し**: Azure LLMがどのツールを呼び出したか
        - **Local LLM**: Foundry Localへの入出力
        - **Azure LLM**: クラウドからのストリーミング応答
        """
        )

    # メインエリア: タブで切り替え
    tab1, tab2 = st.tabs(["💬 チャット", "📋 MCPログ"])

    with tab1:
        # プロンプト選択
        prompt_files = get_prompts_for_agent(agent_type)

        if prompt_files:
            col_select, col_load = st.columns([3, 1])
            with col_select:
                selected_prompt = st.selectbox(
                    "プロンプトを選択",
                    options=prompt_files,
                    format_func=lambda p: f"{p.title} - {p.description}" if p.description else p.title,
                    key="prompt_select",
                )
            with col_load:
                st.write("")  # 位置調整用
                if st.button("📄 読み込む", use_container_width=True):
                    if selected_prompt:
                        st.session_state.message_input = selected_prompt.content
                        st.rerun()
        else:
            st.info(f"{agent_type}/prompts/ フォルダにプロンプトファイル(.md)がありません。")

        # メッセージ入力
        user_message = st.text_area(
            "メッセージを入力",
            value=st.session_state.user_input,
            height=200,
            key="message_input",
        )

        run_button = st.button("🚀 実行", type="primary", use_container_width=True)

        if run_button and user_message.strip():
            # 2カラムレイアウト
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔧 ツール呼び出し & MCP応答")
                tool_placeholder = st.container()

                # st.subheader("💻 Local LLM (Foundry Local)")
                local_placeholder = st.empty()

            with col2:
                st.subheader("☁️ Azure LLM レスポンス")
                azure_placeholder = st.empty()

            placeholders = {
                "tool_calls": tool_placeholder,
                "local_llm": local_placeholder,
                "azure_llm": azure_placeholder,
            }

            with st.spinner("処理中..."):
                try:
                    result = asyncio.run(
                        run_agent_stream(agent_type, user_message, placeholders)
                    )
                    st.success("✅ 完了!")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

        elif run_button:
            st.warning("メッセージを入力してください。")

    with tab2:
        # MCPリクエストログ表示
        render_mcp_logs()


if __name__ == "__main__":
    main()
