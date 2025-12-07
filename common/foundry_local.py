"""
Foundry Local 接続クライアント

Microsoft Foundry Local (Phi-4-mini) へのHTTP接続を提供する共通モジュール。
OpenAI互換のChat Completions APIを使用。
"""

import json
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from .llm_logger import LLMLogEntry, llm_logger

# .env ファイルから環境変数を読み込み
load_dotenv()

# Foundry Local エンドポイント設定
# 環境変数 FOUNDRY_LOCAL_URL から読み取り（`foundry service status` で確認可能）
# 例: FOUNDRY_LOCAL_URL=http://127.0.0.1:53032
FOUNDRY_LOCAL_BASE = os.getenv("FOUNDRY_LOCAL_URL", "http://127.0.0.1:5273")
FOUNDRY_LOCAL_CHAT_URL = FOUNDRY_LOCAL_BASE + "/v1/chat/completions"

# 動作確認済みモデルID（`foundry model list`で確認）
# NPU版: phi-4-mini-instruct-vitis-npu:2
# CUDA版: Phi-4-mini-instruct-cuda-gpu:5
FOUNDRY_LOCAL_MODEL_ID = "phi-4-mini-instruct-vitis-npu:2"


def call_local_model(
    system_prompt: str,
    user_content: str,
    model_id: str = FOUNDRY_LOCAL_MODEL_ID,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout: int = 600,
    tool_name: str = "unknown",
) -> tuple[Dict[str, Any], Optional[LLMLogEntry]]:
    """
    Foundry Local を呼び出す共通関数。

    Args:
        system_prompt: システムプロンプト
        user_content: ユーザーメッセージ
        model_id: 使用するモデルID（デフォルト: Phi-4-mini）
        max_tokens: 最大トークン数
        temperature: 温度パラメータ（0.0-1.0）
        timeout: タイムアウト秒数
        tool_name: 呼び出し元のツール名（ログ表示用）

    Returns:
        (OpenAI互換形式のレスポンス辞書, ログエントリ) のタプル

    Raises:
        requests.HTTPError: API呼び出しに失敗した場合
    """
    # 入力をログに記録
    log_entry = llm_logger.log_request(
        tool_name=tool_name,
        system_prompt=system_prompt,
        user_content=user_content,
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Content-Type": "application/json",
    }

    # デバッグ: Foundry Local INPUT を標準出力に表示（JSON全体）
    print(f"\n{'='*60}")
    print(f"[Foundry Local] INPUT - Tool: {tool_name}")
    print(f"{'='*60}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"{'='*60}\n")

    resp = requests.post(
        FOUNDRY_LOCAL_CHAT_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
    )

    resp.raise_for_status()
    response_json = resp.json()

    # デバッグ: Foundry Local OUTPUT を標準出力に表示
    print(f"\n{'='*60}")
    print(f"[Foundry Local] OUTPUT - Tool: {tool_name}")
    print(f"{'='*60}")
    print(json.dumps(response_json, indent=2, ensure_ascii=False))
    print(f"{'='*60}\n")

    return response_json, log_entry


def extract_content(response: Dict[str, Any]) -> str:
    """
    Foundry Localのレスポンスからコンテンツテキストを抽出する。

    Args:
        response: OpenAI互換形式のレスポンス辞書

    Returns:
        抽出されたテキストコンテンツ
    """
    content = response["choices"][0]["message"]["content"]

    # コンテンツが文字列またはパーツリストの場合を処理
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content
