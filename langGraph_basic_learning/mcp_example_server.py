"""
本文件用于写一个MCP server用于示例演示
"""
"""
python绘图的mcp
"""
import sys
sys.path.append("/root/wangshihang/langGraph_agent/langGraph_basic_learning")
from mcp.server import FastMCP
# %%
mcp = FastMCP(name='check_at_home_mcp_example', instructions='check_at_home MCP', host="0.0.0.0", port=9007)
# %%
@mcp.tool()
async def check_at_home(user_name: str):
    """
    检查用户是否在家
    :param user_name:用户名，如：张三,里斯
    :return: [在家,不在家]
    """
    if user_name == "张三":
        return '在家'
    else:
        return '不在家'
# %%
if __name__ == "__main__":
    # 以标准 sse方式运行 MCP 服务器
    mcp.run(transport='streamable-http')
# nohup uv run mcp_example_server.py &

# %%
