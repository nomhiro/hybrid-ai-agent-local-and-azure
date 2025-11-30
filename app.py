"""
Streamlit UI for Hybrid AI Agent System

ローカルLLM（Foundry Local）とクラウドLLM（Azure AI）を組み合わせた
ハイブリッドAIエージェントのUIを提供する。

実行方法:
    streamlit run app.py
"""

import asyncio
import json
from pathlib import Path

import streamlit as st
from azure.identity.aio import AzureCliCredential

from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient

from common.llm_logger import LLMLogEntry, llm_logger
from common.prompt_loader import get_prompts_for_agent
from finance.agent import (
    FINANCIAL_PLANNER_INSTRUCTIONS,
    analyze_financial_assets,
    analyze_life_plan,
)
from medical.agent import (
    SYMPTOM_CHECKER_INSTRUCTIONS,
    summarize_lab_report,
)


def append_runtime_data(message: str, agent_type: str) -> str:
    """
    ユーザーメッセージをそのまま返す。

    機密データ（金融資産、検査結果など）はローカルツール内で
    直接ファイルから読み込むため、ここでは追記しない。
    これにより、Azure AIエージェントには個人情報が送信されない。

    Args:
        message: ユーザーが入力したメッセージ
        agent_type: エージェントタイプ（"finance" または "medical"）

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

    # 実行時に機密データを追記（ユーザーには見せないデータを埋め込む）
    expanded_message = append_runtime_data(user_message, agent_type)

    # ロガーにコールバックを設定
    llm_logger.clear()
    llm_logger.set_callback(create_log_callback(placeholders["local_llm"]))

    # エージェント設定を選択
    if agent_type == "finance":
        instructions = FINANCIAL_PLANNER_INSTRUCTIONS
        tools = [analyze_financial_assets, analyze_life_plan]
        name = "hybrid-financial-planner"
    else:
        instructions = SYMPTOM_CHECKER_INSTRUCTIONS
        tools = [summarize_lab_report]
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

    async with (
        AzureCliCredential() as credential,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            instructions=instructions,
            tools=tools,
            name=name,
        ) as agent,
    ):
        full_text = ""
        tool_calls_displayed = []

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
                        try:
                            args = content.parse_arguments()
                            args_json = json.dumps(args, indent=2, ensure_ascii=False)
                        except Exception:
                            args_json = str(getattr(content, "arguments", ""))

                        placeholders["tool_calls"].info(
                            f"**Tool:** {content.name}\n\n"
                            f"**Arguments:**\n```json\n{args_json[:500]}\n```"
                        )

        # 最終結果（カーソルなし）
        placeholders["azure_llm"].markdown(base_display + "\n" + full_text)
        return full_text


def main():
    st.set_page_config(page_title="Hybrid AI Agent", page_icon="🤖", layout="wide")
    st.title("🤖 Hybrid AI Agent System")
    st.caption("Local LLM (Foundry Local) + Cloud LLM (Azure AI)")

    # セッションステートの初期化
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # サイドバー
    with st.sidebar:
        st.header("設定")
        agent_type = st.selectbox(
            "エージェントを選択",
            options=["finance", "medical"],
            format_func=lambda x: (
                "💰 ファイナンシャルプランナー" if x == "finance" else "🏥 医療トリアージ"
            ),
        )

        st.divider()
        st.markdown(
            """
        ### 使い方
        1. エージェントを選択
        2. プロンプトを選択して読み込む、またはメッセージを入力
        3. 「実行」をクリック

        ### 表示内容
        - **ツール呼び出し**: Azure LLMがどのツールを呼び出したか
        - **Local LLM**: Foundry Localへの入出力
        - **Azure LLM**: クラウドからのストリーミング応答
        """
        )

    # メインエリア: プロンプト選択
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
            st.subheader("🔧 ツール呼び出し")
            tool_placeholder = st.container()

            st.subheader("💻 Local LLM (Foundry Local)")
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


if __name__ == "__main__":
    main()
