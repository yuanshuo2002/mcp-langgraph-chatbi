#%%
"""
LangGraph 专为希望构建强大、适应性强的 AI 智能体的开发者而设计。开发者选择 LangGraph 的原因是：
可靠性和可控性。通过审核检查和人工干预审批来指导智能体行为。LangGraph 可为长时间运行的工作流持久化上下文，使您的智能体保持正常运行。
低层级和可扩展性。使用完全描述性的低层级原语构建自定义智能体，不受限制自定义的僵化抽象约束。设计可扩展的多智能体系统，其中每个智能体都为您的用例量身定制特定角色。
一流的流式传输支持。通过逐令牌流式传输和中间步骤流式传输，LangGraph 让用户实时清晰地了解智能体的推理和行动过程。
pip install -U langgraph langsmith
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
import sys
from dotenv import load_dotenv
sys.path.append("/root/wangshihang/langGraph_agent/smart_data_analysis_assistant")
load_dotenv()
"""
现在您可以使用 LangGraph 创建一个基本聊天机器人。这个聊天机器人将直接回复用户的消息。
1.首先创建一个 StateGraph。一个 StateGraph 对象将我们的聊天机器人结构定义为“状态机”。
我们将添加 节点 来表示 LLM 和聊天机器人可以调用的函数，并添加 边 来指定机器人应如何在这些函数之间进行转换。
定义图时，第一步是定义其状态。状态 包括图的模式和处理状态更新的 reducer函数。状态是一个具有一个键：messages的TypedDict。 
add_messages reducer函数用于将新消息追加到列表中，而不是覆盖它。没有 reducer 注解的键将覆盖先前的值。
每个节点都可以接收当前状态作为输入，并输出状态的更新。对消息的更新将追加到现有列表而不是覆盖它，这得益于与 Annotated 语法一起使用的预构建 
的add_messages 函数。
"""
class State(TypedDict):
    messages: Annotated[list, add_messages] #{"messages":[{xxxx},{xxxxx}]}
#初始化一张空图，使用定义的State作为状态传递
graph_builder = StateGraph(State)
print(graph_builder)
#%%
"""
添加一个节点¶
接下来，添加一个“chatbot”节点。 节点 表示工作单元，通常是普通的 Python 函数。
pip install -U "langchain[openai]"
"""
import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    temperature=0,
    model='deepseek-chat',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com")
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]} #State['messages']+[{"messages": [llm.invoke(state["messages"])]}]

#将chatbot节点添加到图中
"""
注意 `chatbot` 节点函数将当前状态作为输入，并返回一个包含更新的消息列表的字典，键为“messages”。这是所有 LangGraph 节点函数的基本模式。
我们状态中的add_messages函数会将LLM的响应消息追加到状态中已有的消息之后。
"""
graph_builder.add_node("chatbot", chatbot)
#%%
"""
#添加一个入口点START，建立Start->chatbot的边edge
# 编译图¶
# 在运行图之前，我们需要对其进行编译。我们可以通过在图构建器上调用 compile() 来完成。这将创建一个 CompiledGraph，我们可以在我们的状态上调用它。
"""
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot",END)
graph = graph_builder.compile() #编译图
print(graph)
#%%
"""
可视化图
pip install matplotlib==3.9.1
"""
import matplotlib

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from io import BytesIO
print(graph)

graph_png = graph.get_graph().draw_mermaid_png()
img = mpimg.imread(BytesIO(graph_png), format='PNG')
plt.imshow(img)
plt.axis('off')
plt.show()
# png_path = '/root/wangshihang/langGraph_agent/smart_data_analysis_assistant/graph_image.png'
# plt.imsave(png_path, img)
plt.axis('off')
plt.show()

#%%
"""
运行图，注意：此时的LangGraph是没有记忆的，只能处理当前用户输入的user_input交给LLM进行会话
"""
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        print("event:",event)
        # event: {'chatbot': {'messages': [
        #     AIMessage(content='你好！😊 有什么可以帮你的吗？', additional_kwargs={'refusal': None}, response_metadata={
        #         'token_usage': {'completion_tokens': 11, 'prompt_tokens': 6, 'total_tokens': 17,
        #                         'completion_tokens_details': None,
        #                         'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0},
        #                         'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 6},
        #         'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_8802369eaa_prod0623_fp8_kvcache',
        #         'id': '71e94679-b32e-48cd-adfe-8e461b07322c', 'service_tier': None, 'finish_reason': 'stop',
        #         'logprobs': None}, id='run--18d42d62-dba2-4181-9dbb-edf5f5cd8245-0',
        #               usage_metadata={'input_tokens': 6, 'output_tokens': 11, 'total_tokens': 17,
        #                               'input_token_details': {'cache_read': 0}, 'output_token_details': {}})]}}
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "fallback"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

#%%
"""LangGraph实现工具调用
（1）工具节点接收一个AIMessage,其子类型为tool_calls类型，并必须包含name、args、id，返回工具调用函数执行结果
注意:工具节点执行的工具实现逻辑而非工具调用参数获取逻辑
返回一个ToolMessage类型
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage
#定义一个自定义的工具函数,并将其注册为Langgraph的工具
@tool
def check_at_home(user_name: str)->str:
    """
    检查用户是否在家
    :param user_name:用户名，如：张三,里斯
    :return: [在家,不在家]
    """
    if user_name=="张三":
        return '在家'
    else:
        return '不在家'

tools=[check_at_home]
tool_executor=ToolNode(tools) #为工具节点绑定参数
ai_message = AIMessage(
    content="",
    tool_calls=[{
        "name": "check_at_home",
        "args": {"user_name": "张三"},
        "id": "call_123"  # A unique ID for the tool call
    }]
)
response = tool_executor.invoke({"messages": [ai_message]})
print(response)
# {'messages': [ToolMessage(content='在家', name='check_at_home', tool_call_id='call_123')]}

#%%
"""(2)那么在langGraph工具调用的AIMessage是从哪里来的呢？
答案是从LLM FunctionCalling处获取参数"""
#定义一个LLM执行节点,用于产生调用check_at_home函数实现的工具调用参数
def call_check_at_home(state):
    llm_with_tools = llm.bind_tools([check_at_home])
    print("state['messages']:",state['messages'])
    # ToolNode 通过图形状态和消息列表进行操作。它期望消息列表中的最后一条消息为 AIMessage 类型，并具有 tool_calls 参数
    get_at_home_params = llm_with_tools.invoke(state['messages'])  # 返回的是一个AI_MESSAGE对象，有可能是工具调用(判断下游业务转向)，有可能不走工具调用(自由回复)
    print("提取用户在家工具调用入参:", get_at_home_params)
    return {"messages": [get_at_home_params]}

# 创建状态图
class State(TypedDict):
    messages: Annotated[list, add_messages] #定义消息传递的key
#初始化一张空图，使用定义的State作为状态传递
workflow = StateGraph(State)
# 创建工具节点
check_at_home_tool_node = ToolNode(tools=[check_at_home])

# 添加工具节点添加到工作流
workflow.add_node("call_check_at_home", call_check_at_home) #执行节点
workflow.add_node("check_at_home_tool_node",check_at_home_tool_node) #工具节点
# 设置入口和出口
workflow.set_entry_point("call_check_at_home") #workflow.add_edge(START,"call_check_at_home")

workflow.add_edge("call_check_at_home","check_at_home_tool_node")

workflow.set_finish_point("check_at_home_tool_node") #workflow.add_edge("check_at_home_tool_node",END)
import matplotlib

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from io import BytesIO
# 编译工作流
app = workflow.compile()
graph_image=app.get_graph().draw_mermaid_png() #带虚线的是条件边，实现的是单线的普通边
img = mpimg.imread(BytesIO(graph_image), format='PNG')
plt.imshow(img)
plt.axis('off')
plt.show()
# 调用工具节点
from langchain_core.messages import AIMessage, HumanMessage
node_result = app.invoke({"messages":[HumanMessage(content="张三在家吗")]})

print("node_result:", node_result)


#%%
"""条件边与多工具选择性调用"""
@tool
def check_at_sleep(user_name:str)->str:
    """
    检查用户是否在睡觉
    :param user_name:用户名，如：张三,里斯
    :return: [睡觉,未睡觉]
    """
    if user_name == "张三":
        return '睡觉'
    else:
        return '未睡觉'
@tool
def check_at_home(user_name: str)->str:
    """
    检查用户是否在家
    :param user_name:用户名，如：张三,里斯
    :return: [在家,不在家]
    """
    if user_name=="张三":
        return '在家'
    else:
        return '不在家'

tools=[check_at_home,check_at_sleep]
# tool_executor=ToolNode(tools) #为工具节点绑定参数
# ai_message = AIMessage(
#     content="",
#     tool_calls=[{
#         "name": "check_at_sleep",
#         "args": {"user_name": "张三"},
#         "id": "call_123"  # A unique ID for the tool call
#     }]
# )
# response = tool_executor.invoke({"messages": [ai_message]})
# print("在睡觉ToolNode模拟执行:",response) #返回一个ToolMessage
from typing_extensions import Literal
#根据调用了什么工具，通过条件性选择边路由到不同的工具执行ToolNode进行功能函数的执行
def call_check_selection(state):
    llm_with_tools = llm.bind_tools([check_at_home,check_at_sleep],tool_choice="auto") #绑定两个工具+同时支持不选择工具,自动选择工具的调用模式
    print("state['messages']1:",state['messages'])
    # ToolNode 通过图形状态和消息列表进行操作。它期望消息列表中的最后一条消息为 AIMessage 类型，并具有 tool_calls 参数
    check_params_result = llm_with_tools.invoke(state['messages'])  # 返回的是一个AI_MESSAGE对象，有可能是工具调用(判断下游业务转向)，有可能不走工具调用(自由回复)
    print("提取用户在家工具调用入参:", check_params_result)
    return {"messages": [check_params_result]}

#定义转向路由条件函数
def select_router(state:State)-> Literal[END,"check_at_home_tool_node", "check_at_sleep_tool_node"]:
    print("state['messages']2:", state['messages'])
    if not state['messages'][-1].content:
        print("state['messages'][-1].tool_calls[0]：", state['messages'][-1].tool_calls[0])
        if state['messages'][-1].tool_calls[0].get('name')=="check_at_home":
            print("进入check_at_home_tool_node分支")
            return "check_at_home_tool_node" #路由条选择返回的是节点名
        else:
            print("进入check_at_sleep分支")
            return "check_at_sleep_tool_node"
    else:
        return END

from langchain_core.messages import AnyMessage
# 创建状态图
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] #定义消息传递的key为message,消息体的结构为 messages:{[AIMessage,HumanMessage,ToolMessage]}
#初始化一张空图，使用定义的State作为状态传递
workflow = StateGraph(State)
# 创建2个工具执行节点
check_at_home_tool_node = ToolNode(tools=[check_at_home])
check_at_sleep_tool_node = ToolNode(tools=[check_at_sleep])
# 创建普通执行节点
workflow.add_node("call_check_selection", call_check_selection) #执行节点
workflow.add_node("check_at_home_tool_node",check_at_home_tool_node) #工具节点
workflow.add_node("check_at_sleep_tool_node",check_at_sleep_tool_node) #工具节点
#将节点通过边连接起来组成langGraph图
workflow.add_edge(START,"call_check_selection") #开始->选择工具节点
workflow.add_conditional_edges("call_check_selection",select_router) #根据工具调用的工具名称选择下游的转向条件
#思考这里为什么没有workflow.add_edge("",END)连接到结束节点？？
tool_selection_graph=workflow.compile()
graph_image=tool_selection_graph.get_graph().draw_mermaid_png() #带虚线的是条件边，实现的是单线的普通边
img = mpimg.imread(BytesIO(graph_image), format='PNG')
plt.imshow(img)
plt.axis('off')
plt.show()
# while True:
#     print("请输出消息:")
#     input_message=input()
#     tool_selection_result=tool_selection_graph.invoke(input={"messages":input_message}) #王大爷在家吗 王大爷在睡觉吗
#     print("tool_selection_result:",tool_selection_result)

#%%
"""集成MCP工具调用 pip install langchain_mcp_adapters"""
import asyncio
from contextlib import asynccontextmanager
from typing import Literal
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from io import BytesIO
from langgraph.graph import StateGraph, START, END
import sys
sys.path.append("/root/wangshihang/langGraph_agent/smart_data_analysis_assistant/chatbi_graph")
# 加载环境变量
load_dotenv(dotenv_path="/root/wangshihang/langGraph_agent/smart_data_analysis_assistant/chatbi_graph/.env")
server_url=os.getenv("server_url") #mcp 服务器ip或者域名
print(server_url)
mcp_server_config = {
    "my_mcp":{
    "url": f"http://{server_url}:9007/mcp", #注意streamable-http协议的路由固定为/mcp
    "transport": "streamable_http",
    "timeout": 20000,  # 增加超时时间
    "sse_read_timeout": 20000
}}

from langchain_openai import ChatOpenAI
# 创建状态图
#定义消息传递的key为message,消息体的结构为 messages:{[AIMessage,HumanMessage,ToolMessage]}
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

#将mcp server_name作为session_name，
@asynccontextmanager  # 作用：用于快速创建异步上下文管理器。它使得异步资源的获取和释放可以像同步代码一样通过 async with 语法优雅地管理。=
async def make_mcp_graph():
    client = MultiServerMCPClient(mcp_server_config)  # 接收一个MCP服务器组对象
    llm = ChatOpenAI(
        temperature=0,
        model='deepseek-chat',
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com")
    #MCP运行依赖于会话状态保持，持久化会话session,使得我clinet与sever之间能持久的会话不断开
    async with client.session("my_mcp") as check_home_session:
        check_at_home_server_tools = await load_mcp_tools(check_home_session)  #
        print(check_at_home_server_tools)
        tools=[] #所有工具
        # 解析tool获取工具变量
        for one_tool in check_at_home_server_tools:
            print("one_tool:",one_tool)
            if one_tool.name=="check_at_home":
                check_at_home_mcp_tool=one_tool
                tools.append(check_at_home_mcp_tool)
        #定义函数调用节点
        def call_check_at_home(state:State):
            print("state['messages']:", state['messages'])
            llm_with_tools = llm.bind_tools(tools)
            # ToolNode 通过图形状态和消息列表进行操作。它期望消息列表中的最后一条消息为 AIMessage 类型，并具有 tool_calls 参数
            get_at_home_params = llm_with_tools.invoke(state['messages'])  # 返回的是一个AI_MESSAGE对象，有可能是工具调用(判断下游业务转向)，有可能不走工具调用(自由回复)
            print("提取用户在家工具调用入参:", get_at_home_params)
            return {"messages": [get_at_home_params]}

        #使用一个ToolNode来执行一个FunctionCalling提取到的AIMessage参数
        check_at_home_tool_node = ToolNode(tools, name="check_at_home_tool_node")

        workflow = StateGraph(State)
        workflow.add_node(call_check_at_home)
        workflow.add_node(check_at_home_tool_node)
        workflow.add_edge(START,"call_check_at_home")
        workflow.add_edge("call_check_at_home","check_at_home_tool_node")
        workflow.add_edge("check_at_home_tool_node",END)
        try:
            graph = workflow.compile() #
        except Exception as e:
            print("创建图出现错误:",e)
        #绘制langGraph流程图并保存到本地
        graph_png = graph.get_graph().draw_mermaid_png()
        img = mpimg.imread(BytesIO(graph_png), format='PNG')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        yield graph

async def run_graph():
    """执行该 工作流"""
    async with make_mcp_graph() as graph:
        while True:
            print("用户输入:")
            user_input = input("用户：")
            if user_input.lower() in ['q', 'exit', 'quit']:
                print('对话结束，拜拜！')
                break
            else:
                async for event in graph.astream({"messages": [{"role": "user", "content": user_input}]},
                                                 stream_mode="values"):
                    #Emit all values in the state after each step, including interrupts.When used with functional API, values are emitted once at the end of the workflow.
                    event["messages"][-1].pretty_print()


asyncio.run(run_graph())


#%%
"""memory机制"""
from langgraph.checkpoint.memory import MemorySaver
thread_config={"configurable":{"thread_id":"session_3"}} #session配置
memory = MemorySaver() #实例化,会存储记忆历史记录至checkpoint.db数据库中
tool_selection_graph_with_memory=graph_builder.compile(memory)
while True:
    print("请输出消息:")
    input_message=input()
    tool_selection_result=tool_selection_graph_with_memory.invoke(input={"messages":input_message},config=thread_config) #王大爷在家吗 王大爷在睡觉吗
    print("tool_selection_result:",tool_selection_result.get('messages')[-1].content) #最后一条消息就是本轮回复的消息
#%%
"""
#消息输出形式:普通输出 vs 流式输出
同步阻塞式调用:
(1)等待整个工作流完全执行完毕
(2)一次性返回完整结果
(3)适用于不需要实时交互的简单场景
(4)内存中保存完整结果后才返回
流式输出（异步/实时）
异步非阻塞式调用：
(1)实时生成和返回部分结果（通过async for逐项处理）
(2)特别适合需要逐步显示结果的交互式场景（如聊天应用）
(3)可以更早开始处理部分结果，降低延迟感
stream_mode 参数控制流式输出的内容：
- "values"：只输出状态值的变化
- "updates"：输出每个步骤的增量更新
- "debug"：包含完整调试信息
"""
while True:
    print("请输入本轮会话内容:")
    user_input=input()
    async for event in tool_selection_graph_with_memory.astream({"messages": [{"role": "user", "content": user_input}]},
                                     stream_mode="values", config=thread_config):  # 保持同一个用户的对话的连续记忆
        # Emit all values in the state after each step, including interrupts.When used with functional API, values are emitted once at the end of the workflow.
        event["messages"][-1].pretty_print()
        print("========")
#%%
"""
langGraph预构建智能体:LangGraph最高层级的封装
"""
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from io import BytesIO
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    temperature=0,
    model='deepseek-chat',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com")
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
memory=MemorySaver() #初始化内存记录
system_prompt="Respond in Chinese" #添加系统提示词,通过创建reAct Agent时引入state_modifier(老版本)/prompt(新版本)参数进行指定

@tool
def check_at_home(user_name: str)->str:
    """
    检查用户是否在家
    :param user_name:用户名，如：张三,里斯
    :return: [在家,不在家]
    """
    if user_name=="张三":
        return '在家'
    else:
        return '不在家'
tools=[check_at_home]
#%%
#工作中用的更多的是create_react_agent来创建图
thread_config={"configurable":{"thread_id":"session_5"}}
agent_compiled_builder=create_react_agent(model=llm,tools=tools,checkpointer=memory,prompt=system_prompt)
graph_image=agent_compiled_builder.get_graph().draw_mermaid_png() #带虚线的是条件边，实现的是单线的普通边
img = mpimg.imread(BytesIO(graph_image), format='PNG')
plt.imshow(img)
plt.axis('off')
plt.show()
result=agent_compiled_builder.invoke({"messages":"我刚问了什么问题"},config=thread_config)
print("返回的结果:",result.get("messages")[-1].content)