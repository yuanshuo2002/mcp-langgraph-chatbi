from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from pathlib import Path
from dotenv import load_dotenv
import os

# 允许在不同机器上运行：优先加载当前目录的 `.env`
script_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=script_dir / ".env", override=False)

# 本项目需要一个可用的聊天模型。
# 你的场景是“只有百炼 Key（DASHSCOPE_API_KEY）且不支持 OpenAI 兼容 HTTP 调用”，
# 因此默认优先使用 `ChatTongyi`（走 DashScope 原生接口），再回退到 OpenAI 兼容的方式。
qwen_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")

prefer_tongyi = os.getenv("PREFER_TONGYI", "1") == "1"

if qwen_key and prefer_tongyi:
    # DashScope 原生（推荐）
    llm = ChatTongyi(
        model=os.getenv("QWEN_MODEL", "qwen-plus"),
        temperature=0,
        dashscope_api_key=qwen_key,
    )
elif qwen_key:
    # OpenAI 兼容模式（只有当你的 Key 支持 http call 时才可用）
    llm = ChatOpenAI(
        temperature=0,
        model=os.getenv("QWEN_MODEL", "qwen-plus"),
        api_key=qwen_key,
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
else:
    llm = ChatOpenAI(
        temperature=0,
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=deepseek_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    )

# 通义千问LLM
# llm = ChatOpenAI(
#     temperature=0,
#     model="qwen-plus-latest", #qwen-plus
#     openai_api_key=os.getenv("QWEN_API_KEY"),
#     openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     extra_body={"chat_template_kwargs": {"enable_thinking": False},"parallel_tool_calls":True},
#     parallel_tool_calls=True
# )
#,"parallel_tool_calls":True
