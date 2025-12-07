import asyncio
import json
import os
from typing import Optional, Dict, Any, Annotated

import requests
from dotenv import load_dotenv
from pydantic import Field

from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

# .env ファイルから環境変数を読み込み
load_dotenv()


# ========= クラウド症状チェッカー指示 =========

SYMPTOM_CHECKER_INSTRUCTIONS = """
あなたは非緊急診断のための慎重な症状チェッカーアシスタントです。

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
  診断のガイダンスを提供する前に、必ず`summarize_lab_report`ツールを呼び出して
  検査データを構造化データに変換してください。
- ツールの結果をコンテキストとして使用しますが、生のJSONを直接公開しないでください。
  代わりに、主要な異常所見を平易な言葉で要約してください。
""".strip()


# ========= ローカル検査要約器 (Foundry Local + Phi-4-mini) =========

# 環境変数 FOUNDRY_LOCAL_URL から読み取り（`foundry service status` で確認可能）
FOUNDRY_LOCAL_BASE = os.getenv("FOUNDRY_LOCAL_URL", "http://127.0.0.1:5273")
FOUNDRY_LOCAL_CHAT_URL = FOUNDRY_LOCAL_BASE + "/v1/chat/completions"

# 動作確認済みのモデルID（`foundry model list`で確認）：
FOUNDRY_LOCAL_MODEL_ID = "phi-4-mini-instruct-vitis-npu:2"


LOCAL_LAB_SYSTEM_PROMPT = """
あなたはユーザーのマシン上でローカルに動作する医療検査レポート要約器です。

必ず有効なJSONオブジェクト1つのみで応答してください。説明、バッククォート、
マークダウン、JSON以外のテキストを含めないでください。JSONは以下の形式である必要があります：

{
  "overall_assessment": "<短い平易な日本語での要約>",
  "notable_abnormal_results": [
    {
      "test": "文字列",
      "value": "文字列",
      "unit": "文字列またはnull",
      "reference_range": "文字列またはnull",
      "severity": "mild|moderate|severe"
    }
  ]
}

フィールドが不明な場合はnullを使用してください。値を捏造しないでください。
""".strip()


def _strip_code_fences(text: str) -> str:
    """
    ```json ... ``` または ``` ... ``` のフェンスがあれば削除する。
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

    payload = {
        "model": FOUNDRY_LOCAL_MODEL_ID,
        "messages": [
            {"role": "system", "content": LOCAL_LAB_SYSTEM_PROMPT},
            {"role": "user", "content": lab_text},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    headers = {
        "Content-Type": "application/json",
    }

    print(f"[ローカルツール] POST {FOUNDRY_LOCAL_CHAT_URL}")
    resp = requests.post(
        FOUNDRY_LOCAL_CHAT_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=120,
    )

    resp.raise_for_status()
    data = resp.json()

    # OpenAI互換形式: choices[0].message.content
    content = data["choices"][0]["message"]["content"]

    # 文字列またはパーツリストを処理
    if isinstance(content, list):
        content_text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    else:
        content_text = content

    print("[ローカルツール] モデルからの生コンテンツ:")
    print(content_text)

    # ```json フェンスがあれば削除してからJSONをパース
    cleaned = _strip_code_fences(content_text)
    lab_summary = json.loads(cleaned)
    print("[ローカルツール] パース済み検査要約JSON:")
    print(json.dumps(lab_summary, indent=2, ensure_ascii=False))

    # dictを返す – Agent Frameworkがツール結果としてシリアライズする
    return lab_summary


# ========= ハイブリッドメイン（エージェントがローカルツールを使用） =========

async def main():
    # サンプルの症例テキスト + エージェントがツールに送信できる生の検査テキスト
    case = (
        "30代の患者がめまいと動悸を訴えています。少し暑がりで、最近疲れやすいとのことです。"
    )

    lab_report_text = """
   -------------------------------------------
   東京内分泌臨床検査サービス
        〒100-0001 東京都千代田区千代田1-1-1
             千代田メディカルビル210号室
         TEL: 03-1234-5678  |  FAX: 03-1234-5679
    -------------------------------------------

    患者情報
    氏名:       佐藤 美咲
    生年月日:   1989年8月23日 (35歳)
    性別:       女性
    患者ID:     STO-558923
    住所:       〒150-0002 東京都渋谷区渋谷2-15-8

    依頼医師
    山本 誠一 医師
    医籍番号: 234567
    医療機関: 渋谷内分泌クリニック

    報告詳細
    受付番号:   25-TECS-224518
    採取日時:   2025/12/03 09:15
    受領日時:   2025/12/03 10:42
    報告日時:   2025/12/03 16:28
    検体:       全血（EDTA）、血清分離管

    ------------------------------------------------------
    甲状腺機能検査
    ------------------------------------------------------
    TSH（甲状腺刺激ホルモン）.... 0.05    µIU/mL     (0.35 – 4.94)  低値
    遊離T4 (FT4) ................ 2.8     ng/dL      (0.7 – 1.8)    高値
    遊離T3 (FT3) ................ 5.2     pg/mL      (2.3 – 4.3)    高値
    TSH受容体抗体 (TRAb) ........ 8.5     IU/L       (< 2.0)        高値

    ------------------------------------------------------
    血球計算（CBC）
    ------------------------------------------------------
    白血球数 (WBC) .......... 6.2      x10^3/µL      (4.0 – 10.0)
    赤血球数 (RBC) .......... 4.35     x10^6/µL      (3.80 – 5.00)
    ヘモグロビン ............ 12.8     g/dL          (11.5 – 15.0)
    ヘマトクリット .......... 38.5     %             (35.0 – 45.0)
    血小板数 ................ 245      x10^3/µL      (150 – 400)

    ------------------------------------------------------
    生化学検査（BMP）
    ------------------------------------------------------
    ナトリウム (Na) ......... 141   mmol/L       (136 – 145)
    カリウム (K) ............ 4.0   mmol/L       (3.5 – 5.1)
    クロール (Cl) ........... 102   mmol/L       (98 – 107)
    尿素窒素 (BUN) .......... 14    mg/dL        (7 – 20)
    クレアチニン ............ 0.70  mg/dL        (0.50 – 1.00)
    血糖（空腹時）........... 92    mg/dL        (70 – 99)

    ------------------------------------------------------
    肝機能検査
    ------------------------------------------------------
    AST ..................... 28  U/L          (10 – 35)
    ALT ..................... 24  U/L          (5 – 40)
    アルカリフォスファターゼ  95  U/L          (35 – 105)
    総ビリルビン ............ 0.6 mg/dL        (0.2 – 1.2)

    ------------------------------------------------------
    バイタルサイン（来院時）
    ------------------------------------------------------
    体温 .................... 37.2 ℃
    血圧 .................... 125/78 mmHg
    脈拍 .................... 108 bpm          高値（頻脈）

    ------------------------------------------------------
    所見
    ------------------------------------------------------
    TSH著明低値、FT4・FT3高値、TRAb陽性から甲状腺機能亢進症
    （バセドウ病）の再発が強く疑われます。

    ------------------------------------------------------
    報告終了
    TECS-検査機関ID: 05E6665084
    本報告書は情報提供のみを目的としており、診断ではありません。
------------------------------------------------------

    """

    # 症例と検査結果の両方を含む単一のユーザーメッセージ。
    # エージェントは検査結果があることを認識し、summarize_lab_report()をツールとして呼び出す。
    user_message = (
        "患者症例:\n"
        f"{case}\n\n"
        "以下は生テキストの検査結果です。必要に応じて、まず要約してください:\n"
        f"{lab_report_text}\n\n"
        "非緊急診断のガイダンスを提供してください。"
    )

    async with (
        AzureCliCredential() as credential,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            instructions=SYMPTOM_CHECKER_INSTRUCTIONS,
            # 👇 ツールがエージェントに登録された
            tools=[summarize_lab_report],
            name="hybrid-symptom-checker",
        ) as agent,
    ):
        result = await agent.run(user_message)

        print("\n=== 症状チェッカー（ハイブリッド: ローカルツール + クラウドエージェント） ===\n")
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())