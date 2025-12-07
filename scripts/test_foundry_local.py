"""
Foundry Local 接続テストスクリプト

NPUまたはGPU/CPUでのモデル動作確認を行う。

使用方法:
    python scripts/test_foundry_local.py
"""
import json
import os
import time

import requests
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込み
load_dotenv()

# Foundry Local 設定
# 環境変数 FOUNDRY_LOCAL_URL から読み取り（`foundry service status` で確認可能）
FOUNDRY_LOCAL_BASE = os.getenv("FOUNDRY_LOCAL_URL", "http://127.0.0.1:5273")
FOUNDRY_LOCAL_CHAT_URL = FOUNDRY_LOCAL_BASE + "/v1/chat/completions"
FOUNDRY_LOCAL_MODELS_URL = FOUNDRY_LOCAL_BASE + "/v1/models"

# テストするモデルID
MODEL_ID = "phi-4-mini-instruct-vitis-npu:2"


def test_foundry_local():
    """Foundry Local の接続テストを実行"""
    print("=" * 60)
    print("Foundry Local 接続テスト")
    print("=" * 60)

    # 1. ヘルスチェック（モデル一覧取得）
    print("\n1. ヘルスチェック（/v1/models）...")
    try:
        resp = requests.get(FOUNDRY_LOCAL_MODELS_URL, timeout=5)
        print(f"   ステータス: {resp.status_code}")
        if resp.status_code == 200:
            models = resp.json()
            print(f"   利用可能なモデル:")
            if "data" in models:
                for model in models["data"]:
                    model_id = model.get("id", "unknown")
                    print(f"     - {model_id}")
            else:
                print(f"   レスポンス: {json.dumps(models, indent=2)}")
        else:
            print(f"   エラーレスポンス: {resp.text[:500]}")
    except requests.ConnectionError:
        print("   エラー: Foundry Local に接続できません")
        print("   → foundry service status で確認してください")
        return
    except Exception as e:
        print(f"   エラー: {e}")
        return

    # 2. 簡単な推論テスト
    print(f"\n2. 推論テスト (モデル: {MODEL_ID})...")
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Hello, what is 2+2? Reply with just the number."}
        ],
        "max_tokens": 50,
        "temperature": 0.1,
    }

    start_time = time.time()
    try:
        resp = requests.post(FOUNDRY_LOCAL_CHAT_URL, json=payload, timeout=120)
        elapsed = time.time() - start_time
        print(f"   ステータス: {resp.status_code}")
        print(f"   処理時間: {elapsed:.2f}秒")

        if resp.status_code == 200:
            data = resp.json()
            print(f"   使用モデル: {data.get('model', 'N/A')}")
            usage = data.get('usage', {})
            print(f"   トークン: prompt={usage.get('prompt_tokens', 'N/A')}, completion={usage.get('completion_tokens', 'N/A')}")

            content = data['choices'][0]['message']['content']
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            print(f"   出力: {content}")

            # NPU使用の推測
            if elapsed > 5:
                print("\n   [注意] 処理時間が5秒以上かかりました。")
                print("   → NPUが正しくロードされていない可能性があります。")
                print("   → foundry model list でモデル確認してください。")
        else:
            print(f"   エラーレスポンス: {resp.text[:500]}")
            if resp.status_code == 404:
                print("\n   [注意] モデルが見つかりません。")
                print(f"   → モデルID '{MODEL_ID}' が正しいか確認してください。")
                print("   → foundry model list で利用可能なモデルを確認してください。")

    except requests.Timeout:
        print("   タイムアウト (120秒)")
        print("   → モデルのロードに時間がかかっている可能性があります。")
    except Exception as e:
        print(f"   エラー: {e}")

    # 3. 長めのプロンプトテスト
    print(f"\n3. 長めのプロンプトテスト...")
    long_prompt = """
    以下の患者情報を読んで、匿名化されたサマリーをJSONで出力してください。

    【患者情報】
    - 年齢: 35歳
    - 症状: 頭痛、倦怠感
    - 既往歴: なし
    - アレルギー: ペニシリン

    JSONのみで回答してください。
    """

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "あなたは医療情報を匿名化するアシスタントです。出力はJSONのみ。"},
            {"role": "user", "content": long_prompt}
        ],
        "max_tokens": 256,
        "temperature": 0.2,
    }

    start_time = time.time()
    try:
        resp = requests.post(FOUNDRY_LOCAL_CHAT_URL, json=payload, timeout=120)
        elapsed = time.time() - start_time
        print(f"   ステータス: {resp.status_code}")
        print(f"   処理時間: {elapsed:.2f}秒")

        if resp.status_code == 200:
            data = resp.json()
            usage = data.get('usage', {})
            print(f"   トークン: prompt={usage.get('prompt_tokens', 'N/A')}, completion={usage.get('completion_tokens', 'N/A')}")

            content = data['choices'][0]['message']['content']
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            # 出力を整形して表示
            print(f"   出力:")
            for line in content.split('\n')[:10]:  # 最初の10行
                print(f"     {line}")
            if content.count('\n') > 10:
                print("     ...")

    except Exception as e:
        print(f"   エラー: {e}")

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    print("\n[次のステップ]")
    print("1. NPUが使われていない場合:")
    print("   - foundry model list でモデル一覧を確認")
    print("   - foundry service status でサービス状態を確認")
    print("   - タスクマネージャー → パフォーマンス → NPU を確認")
    print("")
    print("2. モデルが見つからない場合:")
    print("   - foundry model download <model-id> でモデルをダウンロード")
    print("   - foundry model run <model-id> でモデルをロード")


if __name__ == "__main__":
    test_foundry_local()
