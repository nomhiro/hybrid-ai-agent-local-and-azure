import asyncio
import json
from typing import Optional, Dict, Any, Annotated

import requests
from pydantic import Field

from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential


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


# ========= ローカル検査要約器 (Foundry Local + Phi-4-mini) =========

FOUNDRY_LOCAL_BASE = "http://127.0.0.1:56234"      # `foundry service status`から取得
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
        "10代の患者がひどい頭痛と嘔吐を訴えています。体温40度で、他の症状はありません。"
    )

    lab_report_text = """
   -------------------------------------------
   東京ファミリー臨床検査サービス
        〒100-0001 東京都千代田区千代田1-1-1
             千代田メディカルビル210号室
         TEL: 03-1234-5678  |  FAX: 03-1234-5679
    -------------------------------------------

    患者情報
    氏名:       山田 太郎
    生年月日:   2007年4月12日 (17歳)
    性別:       男性
    患者ID:     YMD-442871
    住所:       〒150-0001 東京都渋谷区神宮前1-2-3

    依頼医師
    田中 美咲 医師
    医籍番号: 123456
    医療機関: 渋谷小児科クリニック

    報告詳細
    受付番号:   24-TFLS-118392
    採取日時:   2025/11/14 14:32
    受領日時:   2025/11/14 16:06
    報告日時:   2025/11/14 20:54
    検体:       全血（EDTA）、血清分離管

    ------------------------------------------------------
    血球計算（CBC）
    ------------------------------------------------------
    白血球数 (WBC) .......... 14.5     x10^3/µL      (4.0 – 10.0)     高値
    赤血球数 (RBC) .......... 4.61     x10^6/µL      (4.50 – 5.90)
    ヘモグロビン ............ 13.2     g/dL          (13.0 – 17.5)    低正常
    ヘマトクリット .......... 39.8     %             (40.0 – 52.0)    低値
    平均赤血球容積 (MCV) .... 86.4     fL            (80 – 100)
    血小板数 ................ 210      x10^3/µL      (150 – 400)

    ------------------------------------------------------
    炎症マーカー
    ------------------------------------------------------
    C反応性蛋白 (CRP) ................ 60 mg/L       (< 5 mg/L)     高値
    赤血球沈降速度 (ESR) ............. 32 mm/hr      (0 – 15 mm/hr) 高値

    ------------------------------------------------------
    生化学検査（BMP）
    ------------------------------------------------------
    ナトリウム (Na) ......... 138   mmol/L       (135 – 145)
    カリウム (K) ............ 3.9   mmol/L       (3.5 – 5.1)
    クロール (Cl) ........... 102   mmol/L       (98 – 107)
    重炭酸イオン (CO2) ...... 23    mmol/L       (22 – 29)
    尿素窒素 (BUN) .......... 11    mg/dL        (7 – 20)
    クレアチニン ............ 0.74  mg/dL        (0.50 – 1.00)
    血糖（空腹時）........... 109   mg/dL        (70 – 99)        高値

    ------------------------------------------------------
    肝機能検査
    ------------------------------------------------------
    AST ..................... 28  U/L          (0 – 40)
    ALT ..................... 22  U/L          (0 – 44)
    アルカリフォスファターゼ  144 U/L          (65 – 260)
    総ビリルビン ............ 0.6 mg/dL        (0.1 – 1.2)

    ------------------------------------------------------
    所見
    ------------------------------------------------------
    軽度の白血球増多と炎症マーカー（CRP、ESR）の上昇は、急性感染症または
    炎症性疾患を示唆する可能性があります。血糖値はやや高値ですが、
    空腹時でなかった可能性があります。

    ------------------------------------------------------
    報告終了
    TFLS-検査機関ID: 05D5554973
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
        "非緊急トリアージのガイダンスを提供してください。"
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