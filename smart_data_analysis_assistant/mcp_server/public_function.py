import sys
from pathlib import Path

# Windows / 本地运行：把项目根目录加入 sys.path，避免写死 Linux 路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from mcp.server import FastMCP
from dotenv import load_dotenv
import os
import asyncio
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
#加载.env文件中的环境变量
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
load_dotenv(dotenv_path=project_root / ".env", override=False)
#定义大模型调用函数，用于处理文本类模型生成功能
async def LLM_replay(messages):
    """
    prompt_template:大模型调用的提示词模板
    message:大模型调用的用户输入
    """
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未找到 DASHSCOPE_API_KEY/QWEN_API_KEY，无法调用通义模型")

    model = os.getenv("QWEN_MODEL", "qwen-plus")
    llm = ChatTongyi(model=model, temperature=0, dashscope_api_key=api_key)
    # ChatTongyi 支持异步调用
    resp = await llm.ainvoke([HumanMessage(content=messages)])
    return getattr(resp, "content", str(resp))

# result=await LLM_replay(messages="你好")
# print(result)