"""
ファイナンシャルプランナーエージェント

ローカルLLM（Foundry Local + Phi-4-mini）とクラウドLLM（Azure AI Agent Framework）を
組み合わせたハイブリッド型ファイナンシャルプランニング支援システム。

機能:
- 資産分析（ローカル）: 金融資産データの匿名化・構造化
- ライフプラン分析（ローカル）: 家族構成と将来計画の匿名化・構造化
- ファイナンシャルプランニング（クラウド）: 一般的なアドバイスの生成

プライバシー保護:
- 具体的な資産金額、年齢、年収はローカルで処理
- クラウドには匿名化された割合・カテゴリ情報のみ送信
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


# ========= クラウドファイナンシャルプランナー指示 =========

FINANCIAL_PLANNER_INSTRUCTIONS = """
あなたは個人向けファイナンシャルプランニングをサポートするアシスタントです。

【重要な免責事項】
- あなたは金融アドバイザーの資格を持っていません
- 投資の推奨や具体的な金融商品の勧誘は行いません
- 提供する情報は一般的な教育目的のみです
- 重要な金融判断は必ず資格を持つファイナンシャルプランナーに相談してください

【一般的な動作】
1. ユーザーから資産情報や家族情報が提供された場合：
   - まず`analyze_financial_assets`ツールで資産データを分析
   - 次に`analyze_life_plan`ツールでライフプランを分析
   - 両方の分析結果を統合してアドバイスを生成

2. 分析結果に基づいて以下を提供：
   - 資産配分の現状評価（良い点・改善点）
   - ライフステージに応じた一般的な考慮事項
   - リスク分散の観点からの気づき
   - 情報収集のための次のステップ提案

3. 以下については言及を控える：
   - 具体的な投資商品名や銘柄
   - 将来のリターン予測
   - 「〜すべき」という断定的なアドバイス
   - 税務・法律に関する専門的な助言

【ツールの使用方法】
- ユーザーが資産情報をJSONまたはテキストで提供した場合：
  → `analyze_financial_assets`ツールを呼び出す
- ユーザーが家族構成や将来計画を提供した場合：
  → `analyze_life_plan`ツールを呼び出す
- 両方の情報がある場合は両方のツールを順に呼び出す

【出力形式】
- 平易な日本語で説明
- 箇条書きを活用し、8項目以下にまとめる
- 専門用語は簡単な説明を添える
- 必ず以下の文で終える：

「※この情報は一般的な参考情報であり、投資助言ではありません。
具体的な資産運用については、資格を持つファイナンシャルプランナーや
金融機関にご相談ください。」
""".strip()


# ========= ローカル資産分析プロンプト =========

LOCAL_ASSET_ANALYSIS_PROMPT = """
あなたはユーザーのマシン上でローカルに動作する金融資産アナライザーです。

【絶対ルール】
- 出力はJSON構造のみ。それ以外は一切出力禁止。
- 説明文、注釈、コメント、マークダウン記法は禁止。
- バッククォート(```)で囲まない。
- 回答は必ず { で始まり } で終わること。

【プライバシー保護】
具体的な金額を出力に含めないでください。割合と分類のみを出力します。

【出力サンプル】
{"total_assets_category":"1000万円〜2000万円","asset_allocation":{"cash_equivalents":{"percentage":35.7,"includes":["銀行預金"]},"investment_securities":{"percentage":40.0,"includes":["NISA成長投資枠","課税口座株式"]},"fixed_income":{"percentage":7.1,"includes":["国債"]},"insurance_products":{"percentage":5.7,"includes":["生命保険"]}},"risk_assessment":{"diversification_score":"moderate","liquidity_ratio":35.7,"tax_advantaged_ratio":25.7,"observations":["株式比率が高め","流動性は適切"]},"nisa_utilization":{"growth_quota_used":true,"reserve_quota_used":true,"annual_limit_status":"活用済み"}}

【出力スキーマ】
- total_assets_category: 500万円未満|500万円〜1000万円|1000万円〜2000万円|2000万円〜5000万円|5000万円以上
- asset_allocation: 各資産クラスのpercentage(数値)とincludes(文字列配列)
- risk_assessment: diversification_score(low|moderate|high), liquidity_ratio(数値), tax_advantaged_ratio(数値), observations(文字列配列)
- nisa_utilization: growth_quota_used(真偽値), reserve_quota_used(真偽値), annual_limit_status(文字列)

分析のポイント:
- 資産クラスは「現金同等物」「投資有価証券」「債券」「保険商品」に分類
- NISAの年間投資上限（成長枠240万円、積立枠120万円）を考慮
- リスク分散の観点から評価
""".strip()


# ========= ローカルライフプラン分析プロンプト =========

LOCAL_LIFE_PLAN_PROMPT = """
あなたはユーザーのマシン上でローカルに動作するライフプランアナライザーです。

【絶対ルール】
- 出力はJSON構造のみ。それ以外は一切出力禁止。
- 説明文、注釈、コメント、マークダウン記法は禁止。
- バッククォート(```)で囲まない。
- 回答は必ず { で始まり } で終わること。

【プライバシー保護】
具体的な年齢、名前、年収を出力に含めないでください。

【出力サンプル】
{"household_type":"子供2人の4人家族","income_structure":{"type":"共働き","primary_income_ratio":0.6,"stability":"安定雇用"},"life_stage_timeline":[{"period":"短期(1-3年)","events":["第一子大学進学"],"financial_priority":"教育資金確保"}],"key_financial_milestones":[{"milestone":"第一子大学進学","timing":"3年後","estimated_impact":"高"}],"risk_factors":["教育費集中","住宅ローン残債"]}

【出力スキーマ】
- household_type: 世帯タイプの説明
- income_structure: type(単身|片働き|共働き), primary_income_ratio(数値), stability(安定雇用|変動あり|自営業)
- life_stage_timeline: 配列[{period, events, financial_priority}]
- key_financial_milestones: 配列[{milestone, timing, estimated_impact}]
- risk_factors: 文字列配列

分析のポイント:
- 教育費は大学進学を基準に「高」影響として評価
- 退職までの期間で資産形成の緊急度を判断
- 住宅ローン完済は大きなキャッシュフロー改善要因
""".strip()


# ========= ローカルツール: 資産分析 =========

@ai_function(
    name="analyze_financial_assets",
    description=(
        "ユーザーのGPU上で動作するローカルモデルを使用して、"
        "金融資産データを匿名化・構造化します。"
        "具体的な金額は外部に送信せず、割合と分類のみを出力します。"
    ),
)
def analyze_financial_assets(
    assets_json: Annotated[str, Field(description="金融資産データのJSON文字列")],
) -> Dict[str, Any]:
    """
    ツール: Foundry Local (Phi-4-mini) を使用して資産データを分析する。

    JSON互換のdictを返す：
    - total_assets_category: 資産総額のカテゴリ
    - asset_allocation: 資産配分の割合
    - risk_assessment: リスク評価
    - nisa_utilization: NISA活用状況
    """
    # Foundry Localを呼び出し（入力はllm_loggerに記録される）
    response, log_entry = call_local_model(
        system_prompt=LOCAL_ASSET_ANALYSIS_PROMPT,
        user_content=assets_json,
        max_tokens=1024,
        temperature=0.2,
        tool_name="analyze_financial_assets",
    )

    # レスポンスからコンテンツを抽出
    content_text = extract_content(response)

    # JSONをパース
    asset_analysis = parse_json_response(content_text)

    # 出力をログに記録
    llm_logger.log_response(log_entry, content_text, asset_analysis)

    return asset_analysis


# ========= ローカルツール: ライフプラン分析 =========

@ai_function(
    name="analyze_life_plan",
    description=(
        "ユーザーのGPU上で動作するローカルモデルを使用して、"
        "家族構成と将来計画を匿名化・構造化します。"
        "具体的な年齢や名前は外部に送信せず、ライフステージと時期のみを出力します。"
    ),
)
def analyze_life_plan(
    family_json: Annotated[str, Field(description="家族構成と将来計画のJSON文字列")],
) -> Dict[str, Any]:
    """
    ツール: Foundry Local (Phi-4-mini) を使用してライフプランを分析する。

    JSON互換のdictを返す：
    - household_type: 世帯タイプ
    - income_structure: 収入構造
    - life_stage_timeline: ライフステージタイムライン
    - key_financial_milestones: 重要な財務マイルストーン
    - risk_factors: リスク要因
    """
    # Foundry Localを呼び出し（入力はllm_loggerに記録される）
    response, log_entry = call_local_model(
        system_prompt=LOCAL_LIFE_PLAN_PROMPT,
        user_content=family_json,
        max_tokens=1024,
        temperature=0.2,
        tool_name="analyze_life_plan",
    )

    # レスポンスからコンテンツを抽出
    content_text = extract_content(response)

    # JSONをパース
    life_plan_analysis = parse_json_response(content_text)

    # 出力をログに記録
    llm_logger.log_response(log_entry, content_text, life_plan_analysis)

    return life_plan_analysis


