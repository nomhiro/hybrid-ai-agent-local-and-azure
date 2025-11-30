# ハイブリッドAI: ローカルLLMとクラウドLLMの融合

**Azure AI Foundry + Foundry Local + Agent Framework によるハイブリッドAIアーキテクチャ**

---

## はじめに

本プロジェクトは、**ローカルで動作するLLM（Foundry Local）** と **クラウドベースのLLM（Azure AI Foundry）** を組み合わせた「ハイブリッドAI」アーキテクチャの実装例です。

従来のAIアプリケーションでは、すべてのデータをクラウドに送信して処理する必要がありました。しかし、医療記録や財務情報などの機密データを扱う場合、プライバシーや規制の観点から課題がありました。

ハイブリッドAIは、この課題を解決します：

> **「機密データはユーザーのマシン上でローカル処理し、高度な推論のみクラウドで実行する」**

---

## ユースケース

本プロジェクトでは、2つのユースケースを実装しています：

| フォルダ | ユースケース | 説明 |
|---------|-------------|------|
| [medical/](./medical/) | 医療トリアージ支援 | 検査レポートをローカルで構造化し、クラウドでトリアージガイダンスを生成 |
| [finance/](./finance/) | ファイナンシャルプランニング | 資産情報と家族構成をローカルで匿名化し、クラウドでアドバイスを生成 |

### 実行方法

```powershell
# 医療ユースケース
python -m medical.agent

# 金融ユースケース
python -m finance.agent
```

---

## ハイブリッドAI構成の5つのメリット

### 1. プライバシー保護

```
┌─────────────────┐          ┌─────────────────┐
│   機密データ     │    ✗    │    クラウド      │
│  （医療・財務）   │ ──────→ │                 │
└─────────────────┘          └─────────────────┘
         │
         ▼ ローカル処理
┌─────────────────┐
│  構造化された     │          ┌─────────────────┐
│  匿名データのみ   │ ──────→ │    クラウド      │
└─────────────────┘    ✓     └─────────────────┘
```

- 個人識別情報（PII）がデバイスを離れない
- GDPR、HIPAA等のデータ保護規制に準拠
- 医療記録、財務データの安全な処理

### 2. コスト最適化

| 処理タイプ | 実行場所 | コスト |
|-----------|---------|--------|
| 基本的なデータ変換・抽出 | ローカル | 無料 |
| 複雑な推論・判断 | クラウド | 従量課金 |

### 3. 低遅延・高速応答

```
従来: ユーザー → クラウド → ユーザー (数秒)
ハイブリッド: ユーザー → ローカル → ユーザー (ミリ秒)
                    ↓
              必要時のみクラウド
```

### 4. 柔軟なスケーラビリティ

- ユーザー数増加時もローカル処理でクラウド負荷を分散
- エッジデバイスでの分散処理が可能

### 5. 規制準拠の容易さ

- データの地理的制限（データレジデンシー）への対応
- 業界固有の規制要件を満たす設計

---

## アーキテクチャ概要

### システム構成図

```
┌────────────────────────────────────────────────────────────────────┐
│                         ユーザー入力                                │
│              （医療データ / 金融データ）                             │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Azure AI Agent Framework                         │
│                      （オーケストレーション層）                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ChatAgent                                                    │  │
│  │  - ユーザー入力の解析                                          │  │
│  │  - ツール呼び出し判断                                          │  │
│  │  - 最終応答の生成                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
          │                                        │
          │ ローカルツール呼び出し                    │ 最終推論
          ▼                                        ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│    Foundry Local          │          │   Azure AI Foundry        │
│   （ローカルGPU処理）      │          │   （クラウド処理）         │
│                           │          │                           │
│  ┌─────────────────────┐ │          │  ┌─────────────────────┐  │
│  │ Phi-4-mini          │ │          │  │ GPT-4o              │  │
│  │ - データ構造化       │ │          │  │ - 高度な推論         │  │
│  │ - 匿名化処理        │ │          │  │ - ガイダンス生成     │  │
│  └─────────────────────┘ │          │  └─────────────────────┘  │
│                           │          │                           │
│  http://localhost:52403   │          │  Azure Endpoint           │
└──────────────────────────┘          └──────────────────────────┘
```

---

## セットアップ

### 前提条件

1. **Windows 11** + **NVIDIA GPU**（CUDA対応）または **NPU**
2. **Python 3.10以上**
3. **Azure サブスクリプション**

### インストール手順

#### 1. Foundry Localのセットアップ

```powershell
# インストール
winget install Microsoft.FoundryLocal

# モデルのダウンロード
foundry model download phi-4-mini

# モデルのロード
foundry model load phi-4-mini

# サービス状態確認（ポート番号を確認）
foundry service status
```

#### 2. Azure CLI認証

```powershell
az login
```

#### 3. 依存パッケージのインストール

```powershell
pip install -r requirements.txt
```

---

## 参考リンク

- [Hybrid AI using Foundry Local, Microsoft Foundry and the Agent Framework - Part 1](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/hybrid-ai-using-foundry-local-microsoft-foundry-and-the-agent-framework---part-1/4470813)
- [Hybrid AI using Foundry Local, Microsoft Foundry and the Agent Framework - Part 2](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/hybrid-ai-using-foundry-local-microsoft-foundry-and-the-agent-framework---part-2/4471983)
- [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/)
- [Azure AI Agent Framework](https://learn.microsoft.com/azure/ai-foundry/agent-framework/)

---

## ファイル構成

```
hybrid-ai-foundry-local-and-microsoft-foundry/
├── common/                # 共通モジュール
│   ├── __init__.py
│   ├── foundry_local.py   # Foundry Local接続クライアント
│   └── utils.py           # ユーティリティ関数
│
├── medical/               # 医療トリアージエージェント
│   ├── __init__.py
│   ├── agent.py           # メインエージェント
│   ├── DESIGN.md          # 設計書
│   └── README.md          # ユースケース説明
│
├── finance/               # ファイナンシャルプランナーエージェント
│   ├── __init__.py
│   ├── agent.py           # メインエージェント
│   ├── DESIGN.md          # 設計書
│   └── README.md          # ユースケース説明
│
├── requirements.txt       # 依存パッケージ
├── README.md              # 本ドキュメント
├── work-log.md            # 開発ログ
└── .env                   # 環境変数（Azure認証情報）
```

---

## ライセンス

MIT License

---

## 免責事項

本プロジェクトは教育・デモンストレーション目的で作成されています。医療トリアージの例は、実際の医療判断に使用することを意図していません。実際の医療に関する判断は、必ず医療専門家にご相談ください。
