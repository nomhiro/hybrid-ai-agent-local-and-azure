# Dev Tunnel セットアップ手順

Azure Dev Tunnelを使用して、ローカルのMCPサーバーをインターネットに公開し、
Azure AI Foundry Agentからアクセスできるようにします。

## 前提条件

- Windows 10/11
- Microsoft / Azure AD / GitHub アカウント（ログイン用）
- MCPサーバーがローカルで起動していること（ポート8081）

## 1. Dev Tunnel CLIのインストール

### Windows Package Manager (winget) を使用

```powershell
winget install Microsoft.devtunnel
```

### 手動インストール

```powershell
Invoke-WebRequest -Uri https://aka.ms/TunnelsCliDownload/win-x64 -OutFile devtunnel.exe
```

ダウンロード後、PATHを通すか、実行ファイルのあるディレクトリで作業してください。

## 2. ログイン

Dev Tunnelを使用するにはログインが必要です。

```powershell
devtunnel user login
```

Microsoft、Azure AD、または GitHub アカウントでログインできます。

## 3. トンネルの作成と起動

### 方法A: 一時的なトンネル（推奨：テスト用）

```powershell
# MCPサーバーのポート8081をトンネル
devtunnel host -p 8081 --allow-anonymous
```

`--allow-anonymous` フラグにより、認証なしでトンネルにアクセスできます。
これはAzure AI Foundryからのアクセスに必要です。

### 方法B: 永続的なトンネル（推奨：繰り返し使用する場合）

```powershell
# トンネルを作成
devtunnel create --allow-anonymous

# ポートを追加
devtunnel port create --port-number 8081 --protocol http

# トンネルを起動
devtunnel host
```

永続的なトンネルは、再起動してもURLが変わりません。

## 4. トンネルURLの確認

トンネル起動後、以下のような出力が表示されます：

```
Hosting port 8081 at https://<tunnel-id>.usw2.devtunnels.ms:8081/
```

このURLをコピーして、次のステップで使用します。

## 5. Streamlit UIでのURL設定

1. Streamlit UIを起動: `streamlit run app.py`
2. サイドバーの「MCPサーバー」セクションで「▶️ 起動」をクリック
3. 「Dev Tunnel URL」欄に、上記でコピーしたURLを入力

## 6. Azure AI Foundryでのツール登録

### Azure AI Foundry Portalでの設定

1. [Azure AI Foundry](https://ai.azure.com/) にアクセス
2. プロジェクトを選択
3. 「Agents」セクションに移動
4. 新しいエージェントを作成、または既存のエージェントを編集
5. 「Tools」タブを選択
6. 「Add Tool」→「MCP」を選択
7. 以下の情報を入力:
   - **Name**: `get_patient_background`
   - **MCP Server URL**: `https://<tunnel-id>.devtunnels.ms`
   - **Authentication**: None（匿名アクセス）
8. 「Save」をクリック

### システムプロンプトの設定

エージェントのシステムプロンプトに以下を追加:

```
患者の医療背景情報が必要な場合は、`get_patient_background` ツールを使用してください。
このツールはローカルで機密データを処理し、匿名化されたサマリーのみを返します。
```

## 7. 動作確認

### ローカルでのテスト

```powershell
# GETリクエスト（ヘルスチェック）
curl http://localhost:8081

# POSTリクエスト（tools/list）
curl -X POST http://localhost:8081 -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### トンネル経由でのテスト

```powershell
# トンネルURL経由でアクセス
curl https://<tunnel-id>.devtunnels.ms

curl -X POST https://<tunnel-id>.devtunnels.ms -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

## トラブルシューティング

### トンネルに接続できない

1. MCPサーバーがローカルで起動していることを確認
2. ファイアウォールでポート8081が開いていることを確認
3. `devtunnel host` コマンドが実行中であることを確認

### 認証エラー

Azure AI Foundryからのアクセスに認証エラーが発生する場合:
- トンネル作成時に `--allow-anonymous` フラグを使用していることを確認
- 既存のトンネルを削除して再作成

```powershell
devtunnel list
devtunnel delete <tunnel-id>
devtunnel create --allow-anonymous
```

### タイムアウト

Foundry Local（Phi-4-mini）の推論に時間がかかる場合、タイムアウトが発生することがあります。
MCPサーバーのタイムアウト設定（600秒）で対応しています。

## 参考リンク

- [Dev Tunnels ドキュメント](https://learn.microsoft.com/en-us/azure/developer/dev-tunnels/)
- [Dev Tunnels CLIリファレンス](https://learn.microsoft.com/en-us/azure/developer/dev-tunnels/cli-commands)
- [Hybrid AI ブログ Part 2](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/hybrid-ai-using-foundry-local-microsoft-foundry-and-the-agent-framework---part-2/4471983)
