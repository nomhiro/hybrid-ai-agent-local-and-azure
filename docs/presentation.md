ご提示いただいたPDF資料『Agent Frameworkを利用した、Foundry LocalとMicrosoft Foundryのハイブリッドエージェント』の内容を基に、Markdown形式の解説資料を作成しました。

-----

# Agent Frameworkを利用したハイブリッドAIエージェント解説資料

[cite_start]**発表者**: Hiroki Nomura (しろくま) [cite: 4, 6]
[cite_start]**日付**: 2025年12月6日 [cite: 3]
[cite_start]**イベント**: なごあずの集い \#7 [cite: 3]

## 1\. はじめに

[cite_start]本資料は、ローカル環境（Foundry Local）とクラウド環境（Microsoft Foundry）を組み合わせた「ハイブリッドAIエージェント」のアーキテクチャとその実装についての解説です。機密性の高いデータを保護しつつ、クラウドの高度な推論能力を活用する手法が提案されています [cite: 1, 2, 54]。

## 2\. 背景とモチベーション

[cite_start]ローカルマシンのNPU（Neural Processing Unit）を活用し、推論処理をローカル環境に留めることには以下の重要な動機があります [cite: 34, 35, 36]。

  * [cite_start]**プライバシー保護**: 個人情報や機密データを外部に出さない [cite: 37]。
  * [cite_start]**通信環境への対応**: ネットワークが不安定な場所での利用 [cite: 38]。
  * [cite_start]**コスト最適化**: クラウドへのトークン送信量を減らしコストを削減 [cite: 39]。
  * [cite_start]**ローカルマシンに対する動作**: ローカル環境特有の操作 [cite: 46]。

## 3\. 従来の課題と解決策

### 3.1 従来のクラウドLLMの問題点

[cite_start]クラウドLLMのみを利用する場合、以下のような機密データがそのままインターネット経由で送信されるリスクがあります [cite: 47, 51]。

  * [cite_start]**機密データの例**: 患者ID、年収、社内文書など [cite: 50]。
  * [cite_start]**リスク**: データ漏洩、規制（GDPR/HIPAA）への抵触、信頼の喪失、コスト増大 [cite: 63, 64, 65]。

### 3.2 解決策：ハイブリッドAIアーキテクチャ

[cite_start]\*\*「機密データはローカルで処理し、高度な推論のみクラウドで実行する」\*\*というアプローチをとります [cite: 55, 56]。

  * [cite_start]**ローカル処理 (Foundry Local)**: データの「匿名化」と「構造化」を行う [cite: 67, 68]。
      * [cite_start]例：具体的な金額などは削除し、比率や重症度（moderateなど）に変換する [cite: 69, 71]。
  * [cite_start]**クラウド処理**: 匿名化されたデータのみを受け取り、高度な推論やガイダンス生成を行う [cite: 74, 76, 79]。

## 4\. アーキテクチャ詳細

[cite_start]このシステムは、ユーザーの入力を解析し、必要に応じてローカルのツール（LLM）を呼び出し、安全なデータのみをクラウドの高性能モデルに渡す構成になっています [cite: 119]。

### 4.1 コンポーネント構成

| 役割 | 担当環境 | 使用モデル/ツール | 主な機能 |
| :--- | :--- | :--- | :--- |
| **Orchestrator** | Agent Framework | ChatAgent | [cite_start]ユーザー入力の解析、ツール呼び出しの判断 [cite: 126, 127] |
| **Local Tool** | Foundry Local | **Phi-4-mini** | [cite_start]データ構造化、匿名化処理、異常値抽出 [cite: 130, 131, 132, 133] |
| **Cloud Model** | Microsoft Foundry | **gpt-5.1** | [cite_start]高度な推論、複雑な判断、ガイダンス生成 [cite: 136, 137, 138] |

> [cite_start]**注記**: ローカルLLM（Phi-4-mini）はMCP (Model Context Protocol) サーバとして公開され、エージェントから呼び出されます [cite: 95, 128]。

### 4.2 データ処理の違い

#### クラウドLLMのみの場合（危険）

生データがそのまま送信されます。

  * [cite_start]患者ID: 12345 [cite: 83]
  * [cite_start]白血球数: $14.5 \times 10^3/\mu L$ [cite: 85]
  * [cite_start]CRP: $60~mg/L$ [cite: 86]

#### ハイブリッドアプローチの場合（安全）

[cite_start]ローカルで以下のような**匿名化JSON**に変換してから送信されます [cite: 99]。

```json
{
  "overall_assessment": "...",
  "notable_abnormal_results": [
    {
      "test": "WBC",
      "severity": "moderate"
    },
    {
      "test": "CRP",
      "severity": "severe"
    }
  ]
}
```

[cite_start]※ 具体的な数値やIDは削除され、`severity`（重症度）のような抽象化された情報のみが含まれます [cite: 106, 112]。

## 5\. 処理フロー（シーケンス）

1.  [cite_start]**ユーザー入力**: 医療データや金融データなどを入力 [cite: 121]。
2.  [cite_start]**解析**: Microsoft Foundryのエージェントが内容を解析し、機密データが含まれると判断した場合、ローカルツールを呼び出します [cite: 127]。
3.  [cite_start]**ローカル処理**: ローカルLLM（Phi-4-mini）がデータを参照し、匿名化・構造化を行います [cite: 130, 158][cite_start]。**機密データはここで止まります** [cite: 134]。
4.  [cite_start]**クラウド送信**: 生成された「匿名化データ」のみがクラウド（gpt-5.1）に送信されます [cite: 144, 159]。
5.  [cite_start]**最終回答**: クラウド側で高度な推論を行い、最終的な回答やガイダンスを生成してユーザーに返します [cite: 160]。

## 6\. まとめ

  * [cite_start]**データ保護**: 機密データはローカル環境から出ることなく処理されます [cite: 143]。
  * [cite_start]**自動化**: Agent Frameworkが自動的にツール呼び出し（ローカルかクラウドか）を判断します [cite: 145]。
  * [cite_start]**実用性**: ローカルのNPUを活用することで、プライバシー、コスト、通信の課題を解決しつつ、クラウドの知能を活用可能です [cite: 35, 54]。

-----

**次のステップとして、このアーキテクチャで具体的に使用されている「MCPサーバ」の設定方法や、「Phi-4-mini」の具体的なコード実装例について知りたいですか？**