"""
医療トリアージエージェント

ローカルLLM（Foundry Local + Phi-4-mini）とクラウドLLM（Azure AI Agent Framework）を
組み合わせたハイブリッド型医療トリアージ支援システム。

機能:
- 症状チェッカー（クラウド）: 非緊急トリアージのガイダンス提供
- 検査レポート要約（ローカル）: 機密医療データのローカル処理

プライバシー保護:
- 検査レポートの具体的な患者ID、検査値はローカルで処理
- クラウドには匿名化・構造化された要約のみ送信
"""

from typing import Annotated, Any, Dict

from pydantic import Field

from agent_framework import ai_function

# 共通モジュールをインポート
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.foundry_local import call_local_model, extract_content
from common.llm_logger import llm_logger
from common.utils import parse_json_response


# ========= クラウド症状チェッカー指示 =========

SYMPTOM_CHECKER_INSTRUCTIONS = """
あなたは非緊急トリアージのための慎重な症状チェッカーアシスタントです。

一般的な動作：
- あなたは医療従事者ではありません。医学的診断を提供したり、治療を処方したりしないでください。
- まず、レッドフラグ症状（例：胸痛、呼吸困難、重度の出血、脳卒中の兆候、
  片側の脱力、意識混濁、失神）を確認してください。これらがある場合は、緊急医療を勧めて終了してください。
- レッドフラグがない場合は、主要な要因（年齢層、期間、重症度）を要約し、以下を提供してください：
  1) 一般の人が取れる適切な次のステップ
  2) 医師に連絡すべきタイミングについての明確なガイダンス
  3) 適切な場合はシンプルなセルフケアのアドバイス
- 平易な言葉を使用し、合計8項目以下にしてください。
- 必ず次の文で終わってください：「これは医学的アドバイスではありません。」

ツールの使用方法：
- ユーザーが検査レポートのテキストを提供した場合、または「以下の検査結果」「検査結果を参照」と言及した場合、
  トリアージのガイダンスを提供する前に、必ず`summarize_lab_report`ツールを呼び出して
  検査データを構造化データに変換してください。
- ツールの結果をコンテキストとして使用しますが、生のJSONを直接公開しないでください。
  代わりに、主要な異常所見を平易な言葉で要約してください。
""".strip()


# ========= ローカル検査要約器用プロンプト =========

LOCAL_LAB_SYSTEM_PROMPT = """
あなたはユーザーのマシン上でローカルに動作する医療検査レポート要約器です。

【絶対ルール】
- 出力はJSON構造のみ。それ以外は一切出力禁止。
- 説明文、注釈、コメント、マークダウン記法は禁止。
- バッククォート(```)で囲まない。
- 回答は必ず { で始まり } で終わること。

【出力サンプル】
{"overall_assessment":"血糖値とHbA1cが高値で糖尿病の可能性があります。肝機能は正常範囲内です。","notable_abnormal_results":[{"test":"空腹時血糖","value":"142","unit":"mg/dL","reference_range":"70-109","severity":"moderate"},{"test":"HbA1c","value":"7.2","unit":"%","reference_range":"4.6-6.2","severity":"moderate"}]}

【出力スキーマ】
- overall_assessment: 短い平易な日本語での要約（文字列）
- notable_abnormal_results: 異常値の配列
  - test: 検査項目名（文字列）
  - value: 検査値（文字列）
  - unit: 単位（文字列またはnull）
  - reference_range: 基準値範囲（文字列またはnull）
  - severity: mild|moderate|severe

フィールドが不明な場合はnullを使用してください。値を捏造しないでください。
""".strip()


# ========= ローカルツール: 検査レポート要約 =========

@ai_function(
    name="summarize_lab_report",
    description=(
        "ユーザーのGPU上で動作するローカルモデルを使用して、生の検査レポートを構造化された異常値に要約します。"
        "ユーザーが検査結果をテキストで提供した場合に使用してください。"
    ),
)
def summarize_lab_report(
    lab_text: Annotated[str, Field(description="要約する検査レポートの生テキスト。")],
) -> Dict[str, Any]:
    """
    ツール: Foundry Local (Phi-4-mini) を使用してユーザーのGPU上で検査レポートを要約する。

    JSON互換のdictを返す：
    - overall_assessment: 短いテキスト要約
    - notable_abnormal_results: 異常検査結果のオブジェクトリスト
    """
    # Foundry Localを呼び出し（入力はllm_loggerに記録される）
    response, log_entry = call_local_model(
        system_prompt=LOCAL_LAB_SYSTEM_PROMPT,
        user_content=lab_text,
        max_tokens=1024,
        temperature=0.2,
        tool_name="summarize_lab_report",
    )

    # レスポンスからコンテンツを抽出
    content_text = extract_content(response)

    # JSONをパース
    lab_summary = parse_json_response(content_text)

    # 出力をログに記録
    llm_logger.log_response(log_entry, content_text, lab_summary)

    return lab_summary


