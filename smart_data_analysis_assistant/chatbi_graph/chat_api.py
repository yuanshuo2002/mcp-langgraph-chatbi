"""
本文件用于对项目进行API封装部署
"""
import sys
from pathlib import Path

# Windows / 本地运行：把项目根目录加入 sys.path，避免写死 Linux 路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# 如无 LangSmith 需求，可不设置；这里保持原行为但不强依赖
import os
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
import asyncio
from smart_data_analysis_assistant.chatbi_graph.build_graph import make_graph
import time
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
import hashlib
import time
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from fastapi import WebSocket, WebSocketDisconnect
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
import sys
import datetime
from collections import defaultdict
import json
from fastapi import WebSocket, WebSocketDisconnect, Query
from async_timeout import timeout  # 确保安装了 async-timeout 库

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
#定义消息体
class UserInput(BaseModel):
    user_id: str
    message: str
    history: list[dict]
@app.post("/chatbi_service")
async def chatbi_server(user_input: UserInput):
    print("user_input:",user_input)
    user_id=user_input.user_id
    user_message=user_input.message
    history=user_input.history
    history.append({"role": "user", "content": user_message})
    print(f"用户Id:{user_id},本轮输入:{user_message},历史记录:{history}")
    # thread_config = {"configurable": {"thread_id": user_id}}

    async with make_graph() as graph:
        print("创建图成功")
        async for event in graph.astream({"messages":history},
                                         stream_mode="values"):  # 保持同一个用户的对话的连续记忆 , config=thread_config
            print("event:",event)
            # Emit all values in the state after each step, including interrupts.When used with functional API, values are emitted once at the end of the workflow.
            messages = event.get('messages')
            event["messages"][-1].pretty_print()
            print("messages:xxxxxxxxxxx",messages)
            if messages:
                if isinstance(messages, list):
                    message = messages[-1]  # 如果消息是列表，则取最后一个
                if message.__class__.__name__ == 'AIMessage':
                    if message.content and not message.tool_calls: #是AIMessage且不是工具调用类消息(是正常回复类的消息)
                        result = message.content  # 需要回传消息
                        print("本轮回复:",result)
                        return {"message": result}
                else:
                    print("中间处理消息:",message)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9008)

# nohup uvicorn chat_api:app --host 0.0.0.0 --port 9008 --workers 1 &
