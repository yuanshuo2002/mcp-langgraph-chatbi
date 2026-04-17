import sys
from pathlib import Path

# Windows / 本地运行：把项目根目录加入 sys.path，避免写死 Linux 路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from contextlib import asynccontextmanager
from typing import Literal
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from smart_data_analysis_assistant.chatbi_graph.my_llm import llm
from smart_data_analysis_assistant.chatbi_graph.my_state import BIState
from smart_data_analysis_assistant.chatbi_graph.tools_node import (
    generate_query_system_prompt,
    query_check_system,
    call_get_schema,
    select_deep_data_analysis_system_prompt,
    get_schema_tool,
    get_schema_node,
)
from langchain_core.messages import SystemMessage, HumanMessage,ToolMessage
from langgraph.checkpoint.memory import MemorySaver
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from io import BytesIO
import os
from dotenv import load_dotenv

# 加载环境变量：优先 chatbi_graph/.env，其次项目根目录 .env
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
load_dotenv(dotenv_path=project_root / ".env", override=False)

# 本地复现：默认 MCP 都在本机
server_url = os.getenv("SERVER_URL") or os.getenv("server_url") or "127.0.0.1"
#数据库查询MCP
mcp_server_config = {
    "search_db_mcp":{
    "url": f"http://{server_url}:9004/sse",
    "transport": "sse",
    "timeout": 20000,  # 增加超时时间
    "sse_read_timeout": 20000
},
#机器学习MCP
"machine_learning_mcp":{
    "url": f"http://{server_url}:9003/mcp",
    "transport": "streamable_http",
    "timeout": 20000,  # 机器学习时间需要久一些
    "sse_read_timeout": 20000
},
#生成python代码，执行python程序的MCP
"python_chart_mcp":{
    "url": f"http://{server_url}:9002/mcp",
    "transport": "streamable_http",
    "timeout": 20000,  # 机器学习时间需要久一些
    "sse_read_timeout": 20000
},
#业务分流MCP
"ywfl_mcp":{
    "url": f"http://{server_url}:9005/mcp",
    "transport": "streamable_http",
    "timeout": 20000.0,  # 机器学习时间需要久一些
    "sse_read_timeout": 20000.0
},
}
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
class PythonState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] #消息追加的模式增加消息

#业务分流路由条件，
def should_continue_ywfl(state: BIState) -> Literal[END,"call_python_coder", "call_list_tables"]:
    """条件路由的，动态边"""
    messages = state["messages"]
    last_message = messages[-1]
    print("ywfl_last_message:",last_message)
    # print("ywfl_last_message.tool_calls:",last_message.tool_calls)
    #兼容 ToolMessage content 为列表或字符串的情况
    content = last_message.content
    if isinstance(content, list):
        # MCP ToolMessage 可能返回 [{'type': 'text', 'text': '...'}] 格式
        content = " ".join(item.get("text", str(item)) for item in content if isinstance(item, dict))
    content = str(content).strip()
    #最后一个message不是functionCall或者是Functioncall但是没有SQL都走到END节点
    if content=="纯python编码": #如果有工具调用则需要根据，业务分流的结果决策向哪里走
        return "call_python_coder"
    elif content=="业务数据查询分析":
        return "call_list_tables"
    else: #如果是正常回复，则结束当前流程->END节点(正常回复)
        return END

#路由逻辑->下游可能的节点
def should_continue(state: BIState) -> Literal[END,"call_select_deep_data_analysis"]:
    """条件路由的，动态边"""
    messages = state["messages"]
    last_message = messages[-1]
    print("should_continue last_message1:",last_message)
    print("should_continue last_message.tool_calls:",last_message.tool_calls)
    #最后一个message不是functionCall或者是Functioncall但是没有SQL都走到END节点
    if not last_message.tool_calls:
        return END
    else:
        if "query" not in last_message.tool_calls[0]["args"]:
            print("触发了MCP但是没有成功返回SQL")
            return END
        return "call_select_deep_data_analysis"

#select_deep_data_analysis节点是否需要进一步调用工具完成数据分析
# def should_analysis_continue(state: BIState) -> Literal[END, "deep_data_analysis_tool"]:
#     """条件路由的，动态边"""
#     messages = state["messages"]
#     last_message = messages[-1]
#     print("should_analysis_continue last_message:",last_message)
#     # print("should_analysis_continue last_message.tool_calls:",last_message.tool_calls)
#     # 还有数据分析、机器学习类的工具调用，因此需要继续走LangGraph的数据分析流程
#     if not last_message.content: #content='' tool_cals:[{args:{}}]
#         return "deep_data_analysis_tool"
#     else: #最后一个message不是functionCall说明run sql已经得到答案了
#         return END



#%%
@asynccontextmanager  # 作用：用于快速创建异步上下文管理器。它使得异步资源的获取和释放可以像同步代码一样通过 async with 语法优雅地管理。
async def make_graph():
    """定义，并且编译工作流"""
    client = MultiServerMCPClient(mcp_server_config) #接收一个MCP服务器组对象
    print(client)
    tools=[] #所有工具
    #并行同时启动4个会话session,session的作用是持久化会话,保持MCP server和client之间的不断
    async with client.session("python_chart_mcp") as python_chart_session, client.session("ywfl_mcp") as ywfl_session, \
        client.session("machine_learning_mcp") as machine_learning_session, client.session("search_db_mcp") as search_db_session:
        python_chart_server_tools = await load_mcp_tools(python_chart_session) #加载所有MCP，并显示所有的工具 list_tools()
        ywfl_server_tools = await load_mcp_tools(ywfl_session)  #
        machine_learning_server_tools = await load_mcp_tools(machine_learning_session)  #
        search_db_server_tools = await load_mcp_tools(search_db_session)  #
        tools.extend(ywfl_server_tools)
        tools.extend(machine_learning_server_tools)
        tools.extend(python_chart_server_tools)
        tools.extend(search_db_server_tools)
        print("所有tools列表:",tools)
        # 解析tool获取工具变量
        for one_tool in tools:
            print("one_tool:",one_tool)
            if one_tool.name=="list_tables_tool":
                list_tables_tool=one_tool
            elif one_tool.name=="db_sql_tool":
                db_sql_tool=one_tool
                print("数据库查询工具db_sql_tool:", db_sql_tool)
            elif one_tool.name == "ywfl_tool":
                ywfl_tool=one_tool
            elif one_tool.name == "run_python_script_tool":
                run_python_script_tool=one_tool
            elif one_tool.name =="reviews_stars_correlation_test_tool":
                reviews_stars_correlation_test_tool=one_tool
            elif one_tool.name =="analysis_product_reviews_tool":
                analysis_product_reviews_tool=one_tool
            elif one_tool.name =="sales_predict_tool":
                sales_predict_tool=one_tool
            elif one_tool.name=="translate_to_python_plot_script": #写python代码绘图的工具
                translate_to_python_plot_script_tool=one_tool
            else:
                print(f"遇到了其它tools:{one_tool.name}")

        def call_identify_intention(state: BIState):
            """业务分流节点"""
            #每一轮的history应当是HumanMessage与最终执行完整个LangGraph流程得到了AI Message回复+本轮消息
            call_identify_system_message=[SystemMessage(content="""你是一个对话智能助手,你具有语言技能和工具调用技能。
            若用户希望做数据分析、查询业务数据、机器学习建模、做计算、绘制统计图表、写代码等工作，你需要调用`ywfl_tool`工具来实现下游任务分流，
            若用户做纯咨询则使用对话技能完成自由对话""")]
            print("多轮对话开始节点BISTATE消息列表检查点:",state['messages'])
            # 不强制ywfl_tool的调用，允许模型在获得解决方案时自然响应，如正常回复。只对写代码和数据分析响应工具调用
            llm_with_tools = llm.bind_tools([ywfl_tool])
            #ToolNode 通过图形状态和消息列表进行操作。它期望消息列表中的最后一条消息为 AIMessage 类型，并具有 tool_calls 参数
            ywfl_result = llm_with_tools.invoke(call_identify_system_message+state['messages']) #返回的是一个AI_MESSAGE对象，有可能是工具调用(判断下游业务转向)，有可能不走工具调用(自由回复)
            print("业务分流模块参数提取:", ywfl_result)
            return  {"messages":[ywfl_result]}
        identify_intention_tool_node = ToolNode([ywfl_tool], name="identify_intention_tool_node")
        def call_python_coder(state: BIState):
            """PYTHON直接写程序+自动执行节点"""
            llm_with_tools = llm.bind_tools([run_python_script_tool])
            python_coder_result = llm_with_tools.invoke(state['messages'],parallel_tool_calls=True) #state['messages'][-2:] whole_message_list
            print("python写程序+执行结果:",python_coder_result)
            return {'messages': [python_coder_result]}
        #执行python程序的工具节点(绘图+编码执行结果)
        python_run_tool_node = ToolNode([run_python_script_tool], name="python_run_tool_node")
        def call_list_tables(state: BIState):
            """获取数据库信息节点"""
            tool_call = {
                "name": "list_tables_tool", #关系型数据库的内置方法
                "args": {},
                "id": "tool1",
                "type": "tool_call",
            }
            print("call_list_tables state['messages']:",state['messages'])
            tool_call_message = AIMessage(content="", tool_calls=[tool_call])
            return {"messages": [tool_call_message]}

         # 第二个节点
        list_tables_tool = ToolNode([list_tables_tool], name="list_tables_tool")

        #数据分析和机器学习、绘图的一个runnable编译完了的子图
        data_analysis_agent = create_react_agent(model=llm,
                                                 tools=[db_sql_tool,
                                                        run_python_script_tool,
                                                        reviews_stars_correlation_test_tool,
                                                        translate_to_python_plot_script_tool,
                                                        analysis_product_reviews_tool,
                                                        sales_predict_tool],
                                                 prompt=generate_query_system_prompt,name="data_analysis_agent",debug=True)
        workflow = StateGraph(BIState)
        workflow.add_node(call_python_coder)
        workflow.add_node(python_run_tool_node)
        workflow.add_node(call_identify_intention)
        workflow.add_node(identify_intention_tool_node)
        workflow.add_node(call_list_tables)
        workflow.add_node(list_tables_tool)
        workflow.add_node(data_analysis_agent) #generate_sql
        workflow.add_edge(START, "call_identify_intention")
        workflow.add_edge("call_identify_intention", "identify_intention_tool_node")
        workflow.add_conditional_edges("identify_intention_tool_node", should_continue_ywfl) #    {"tools": "tools", END: END},
        workflow.add_edge("call_python_coder", "python_run_tool_node")
        workflow.add_edge("python_run_tool_node",END) #执行python程序
        workflow.add_edge("call_list_tables", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "data_analysis_agent")
        workflow.add_edge('data_analysis_agent', END) #generate_sql 工具调用了代表生成了sql，没有工具调用代表出了问题走到END结束节点;如果没有继续走就从should_continue中走到END
        #构建带有MemorySaver的图结构
        # memory=MemorySaver()
        print("正在创建LangGraph图....")
        try:
            graph = workflow.compile() # checkpointer=memory
        except Exception as e:
            print("创建图出现错误:",e)
        # 绘制 LangGraph 流程图并保存到本地
        # Windows/服务端环境下 `plt.show()` 可能阻塞，因此默认仅保存文件，不弹窗。
        graph_png = graph.get_graph().draw_mermaid_png()
        with open("./build_graph.png", "wb") as f:
            f.write(graph_png)
        if os.getenv("SHOW_GRAPH") == "1":
            img = mpimg.imread(BytesIO(graph_png), format="PNG")
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        yield graph


