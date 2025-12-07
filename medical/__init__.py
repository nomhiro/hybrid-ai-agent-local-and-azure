# medical - 医療診断エージェント
# Azure AI Foundry Agent + ローカルMCPサーバーのハイブリッド構成
#
# MCPツール (get_patient_background) はmcp_server/で実装

from .agent import SYMPTOM_CHECKER_INSTRUCTIONS

__all__ = [
    "SYMPTOM_CHECKER_INSTRUCTIONS",
]
