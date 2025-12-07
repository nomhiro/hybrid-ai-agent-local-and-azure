"""
MCP (Model Context Protocol) サーバーパッケージ

医療トリアージエージェントをHTTP MCPサーバーとして公開する。
Azure AI Foundry Agent からDev Tunnel経由でアクセス可能。
"""

from .mcp_state import MCPState, MCPRequestLog, mcp_state, TunnelStatus
from .mcp_medical_server import run_mcp_server, stop_mcp_server, MCP_SERVER_PORT
from .dev_tunnel import DevTunnelManager, TunnelResult

__all__ = [
    "MCPState",
    "MCPRequestLog",
    "mcp_state",
    "TunnelStatus",
    "run_mcp_server",
    "stop_mcp_server",
    "MCP_SERVER_PORT",
    "DevTunnelManager",
    "TunnelResult",
]
