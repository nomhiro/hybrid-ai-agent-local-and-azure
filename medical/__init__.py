# medical - 医療トリアージエージェント
# ローカルLLM（Foundry Local）とクラウドLLM（Azure AI）のハイブリッド構成

from .agent import (
    SYMPTOM_CHECKER_INSTRUCTIONS,
    summarize_lab_report,
)

__all__ = [
    "SYMPTOM_CHECKER_INSTRUCTIONS",
    "summarize_lab_report",
]
