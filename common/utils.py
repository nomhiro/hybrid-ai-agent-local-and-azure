"""
共通ユーティリティ関数

JSON処理、テキスト処理などの汎用関数を提供。
"""

import json
from typing import Dict, Any


def strip_code_fences(text: str) -> str:
    """
    ```json ... ``` または ``` ... ``` のコードフェンスがあれば削除する。

    LLMがJSON出力にマークダウンのコードブロックを追加することがあるため、
    パース前にそれらを除去する。

    Args:
        text: 処理対象のテキスト

    Returns:
        コードフェンスを除去したテキスト
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # 先頭の ``` を削除
        stripped = stripped[3:].lstrip()
        # "json" などの言語タグがあれば削除
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
        # 末尾の ``` を削除
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


def parse_json_response(content: str) -> Dict[str, Any]:
    """
    LLMのレスポンスからJSONをパースする。

    コードフェンスの除去とJSONパースを一括で行う。

    Args:
        content: LLMからのレスポンステキスト

    Returns:
        パースされたJSON辞書

    Raises:
        json.JSONDecodeError: JSONパースに失敗した場合
    """
    cleaned = strip_code_fences(content)
    return json.loads(cleaned)
