"""
本文件用于演示LangGraph基本功能
langSmith注册APIKEY页面：https://smith.langchain.com/settings
"""
#%%
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
from langgraph.graph import START,StateGraph,END
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGSMITH_API_KEY"] = "your_langsmith_api_key"
# os.environ["LANGCHAIN_PROJECT"] = "LangGraph_Practise" #项目名称
import sys
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
# 加载环境变量
load_dotenv()
sys.path.append("/root/wangshihang/langGraph_agent/smart_data_analysis_assistant")
#创建state
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"]=os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "test_project"
#通过python LangSmith SDK的方式创建project
# from langsmith.async_client import AsyncClient
# # llm = ChatOpenAI(
# #     temperature=0,
# #     model='deepseek-chat',
# #     api_key=os.getenv("DEEPSEEK_API_KEY"),
# #     base_url="https://api.deepseek.com")
# # llm.invoke("Hello, world!")
# client=AsyncClient(
#     api_url="https://api.smith.langchain.com",
#     api_key="<your-langsmith-api-key>",
#     timeout_ms=20000
# )
# await client.create_project(project_name="test_project") #通过代码创建项目,如果没有指定项目则统一存储到default下面

# os.environ["OPENAI_API_KEY"]="<your-openai-api-key>"
def my_node(state):
    return {"x":state["x"]+1,"y":state["y"]+2}
#创建一个状态图构建器builder（graph对象）,适用字典类型作为状态类型
builder=StateGraph(dict) #这里支持各种类型的数据形式的传递
#向构建器中(graph实例化对象)中增加节点
builder.add_node("my_node",my_node) #等价于 builder.add_node("my_node",my_node)
#添加一条边连接两个节点名称
builder.add_edge(START,"my_node")
builder.add_edge("my_node",END)
#编译状态图,生成可执行图
graph=builder.compile(debug=True)
print(graph)
graph_image=graph.get_graph().draw_mermaid_png() #带虚线的是条件边，实现的是单线的普通边
img = mpimg.imread(BytesIO(graph_image), format='PNG')
plt.imshow(img)
plt.axis('off')
plt.show()
graph.invoke(input={"x":7,"y":8})
#%%

#%%
#langchain和langSmith平台的交互
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
# 定义一个简单工具
def calculator(query: str) -> str:
    return str(eval(query))
tools = [Tool(name="Calculator", func=calculator, description="用于数学计算")]
llm = ChatOpenAI(
    temperature=0,
    model='deepseek-chat',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com")
# 初始化 LLM 和 Agent
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

# 运行 Agent
result = agent.run("计算 2 + 3 * 4")
print(result)
#%%

