# ハイブリッドAI: ローカルLLMとクラウドLLMの融合

**Azure AI Foundry + Foundry Local + Agent Framework によるハイブリッドAIアーキテクチャ**

---

## はじめに

本プロジェクトは、**ローカルで動作するLLM（Foundry Local）** と **クラウドベースのLLM（Azure AI Foundry）** を組み合わせた「ハイブリッドAI」アーキテクチャの実装例です。

従来のAIアプリケーションでは、すべてのデータをクラウドに送信して処理する必要がありました。しかし、医療記録などの機密データを扱う場合、プライバシーや規制の観点から課題がありました。

ハイブリッドAIは、この課題を解決します：

> **「機密データはユーザーのマシン上でローカル処理し、高度な推論のみクラウドで実行する」**

---

## ユースケース

本プロジェクトでは、医療診断支援のユースケースを実装しています：

| フォルダ | ユースケース | 説明 |
|---------|-------------|------|
| [medical/](./medical/) | 医療診断支援 | 検査レポートをローカルで構造化し、クラウドで診断ガイダンスを生成 |

### 実行方法

```powershell
# Streamlit UIを起動（推奨）
streamlit run app.py

# MCPサーバーを単独で起動
python -m mcp_server.mcp_medical_server
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
- 医療記録の安全な処理

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

### システム構成図（MCPサーバー版）

医療診断はMCPサーバーとして動作し、Azure AI Foundry AgentからDev Tunnel経由でアクセスできます。

**重要なプライバシー保護ポイント**:
- ユーザーの症状のみがクラウドに送信される（個人情報なし）
- 患者データ（アレルギー、既往歴、服用薬）はローカルMCPサーバー内でのみ処理
- クラウドには匿名化されたサマリーのみ送信

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Streamlit UI (app.py)                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ MCPサーバー状態  │  │ Azure LLM表示   │  │ MCPリクエストログ       │   │
│  │ (起動/停止/URL)  │  │ (リクエスト/応答)│  │ (ツール呼び出し状況)    │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │                              ↑
          │ 1. ユーザー入力（症状のみ）  │ 6. 最終回答
          ▼                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│              Azure AI Foundry Agent (クラウド)                              │
│  - MCPStreamableHTTPTool経由でローカルMCPサーバーに接続                      │
│  - 症状チェッカーとして診断ガイダンス生成                                    │
└─────────────────────────────────────────────────────────────────────────────┘
          │ 2. MCPツール呼び出し         ↑ 5. 匿名化サマリー
          │    (症状を引数として)        │    （個人情報なし）
          ▼                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Dev Tunnel (HTTPS公開)                             │
│                    https://<tunnel-id>.devtunnels.ms                        │
└─────────────────────────────────────────────────────────────────────────────┘
          │                              ↑
          ▼                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│              MCPサーバー (mcp_server/mcp_medical_server.py) - ポート8081    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ HTTP JSON-RPC Handler                                                │   │
│  │ - initialize: サーバー情報返却                                        │   │
│  │ - tools/list: get_patient_background ツール公開                       │   │
│  │ - tools/call: ツール実行                                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│          │ 3. 患者データ読み込み       ↑ 4. 匿名化サマリー生成             │
│          ▼   （ローカルファイル）      │                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Foundry Local (Phi-4-mini) - http://127.0.0.1:56234                  │   │
│  │ - 患者データをローカルで処理（機密情報はここで完結）                   │   │
│  │ - 匿名化されたJSONサマリーを生成                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### データフロー詳細

1. **ユーザー入力**: 症状のみをStreamlit UIに入力（個人情報なし）
2. **Azure AI呼び出し**: 症状がAzure AI Foundry Agentに送信される
3. **MCPツール呼び出し**: Agentが`get_patient_background`ツールをDev Tunnel経由で呼び出し
4. **ローカル処理**: MCPサーバーが患者データを読み込み、Foundry Localで処理
5. **匿名化サマリー**: 個人情報を含まない構造化サマリーをクラウドに返却
6. **最終回答**: Agentが患者背景を考慮した診断ガイダンスを生成

---

## セットアップ

### 前提条件

1. **Windows 11** + **NVIDIA GPU**（CUDA対応）または **NPU**
2. **Python 3.10以上**
3. **Azure サブスクリプション**
4. **Dev Tunnel CLI**（MCPサーバーを外部公開する場合）

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

#### 4. Dev Tunnelのセットアップ（オプション）

MCPサーバーをAzure AI Foundry Agentから呼び出す場合は、Dev Tunnelが必要です。

```powershell
# Dev Tunnel CLIのインストール
winget install Microsoft.devtunnel

# ログイン
devtunnel user login

# トンネルの起動（MCPサーバー起動後）
devtunnel host -p 8081 --allow-anonymous
```

詳細は [Dev Tunnelセットアップ手順](./docs/dev-tunnel-setup.md) を参照してください。

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
├── app.py                 # Streamlit UIアプリケーション
│
├── mcp_server/            # MCPサーバー（医療診断用）
│   ├── __init__.py
│   ├── mcp_medical_server.py  # HTTP JSON-RPC MCPサーバー
│   └── mcp_state.py       # サーバー状態管理
│
├── common/                # 共通モジュール
│   ├── __init__.py
│   ├── foundry_local.py   # Foundry Local接続クライアント
│   ├── llm_logger.py      # LLM入出力ログ管理
│   └── utils.py           # ユーティリティ関数
│
├── medical/               # 医療診断エージェント
│   ├── __init__.py
│   ├── agent.py           # エージェント
│   ├── data/              # 患者データ（機密・ローカル保持）
│   ├── DESIGN.md          # 設計書
│   └── README.md          # ユースケース説明
│
├── docs/                  # ドキュメント
│   └── dev-tunnel-setup.md  # Dev Tunnelセットアップ手順
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

本プロジェクトは教育・デモンストレーション目的で作成されています。医療診断の例は、実際の医療判断に使用することを意図していません。実際の医療に関する判断は、必ず医療専門家にご相談ください。
