"""
MCP chỉ là adapter/transport.
Business tools được discover qua tool_registry.py.
"""

from __future__ import annotations
import logging
from mcp.server.fastmcp import FastMCP
from tool_registry import REGISTERED_TOOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

mcp = FastMCP(
    "dmp-coffee-tools",
    instructions=(
        "Business tools của quán cà phê DMP. "
        "Tool definitions được discover từ tool registry."
    ),
)

def _register_tools() -> None:
    for tool in REGISTERED_TOOLS:
        function = getattr(tool, "func", None)
        if function is None or not callable(function):
            raise TypeError(f"Tool '{tool.name}' không có function gốc hợp lệ.")
                
        mcp.add_tool(function, name=tool.name, description=tool.description)
        logger.info("Registered MCP tool: %s", tool.name)

_register_tools()

if __name__ == "__main__":
    mcp.run(transport="stdio")
