# MCP + LangGraph 智能数据分析助手 (ChatBI)

基于 **MCP (Model Context Protocol)** 和 **LangGraph** 构建的企业级数据分析智能助手。通过自然语言对话，实现数据库查询、数据可视化、用户画像分析、评论情感挖掘、销量预测等复杂分析任务。

## 项目架构

```
用户自然语言输入
       │
       ▼
┌──────────────────────────────────────────────────────┐
│              LangGraph 主图 (execute_graph.py)         │
│                                                      │
│  ┌─────────────┐     ┌──────────────────────────┐    │
│  │ 业务分流节点  │────▶│  条件路由                  │    │
│  │ (ywfl_tool)  │     │  "业务数据查询分析" 或      │    │
│  └─────────────┘     │  "纯python编码"           │    │
│                      └──────┬──────────┬─────────┘    │
│                             │          │              │
│                    ┌────────▼──┐  ┌────▼───────────┐  │
│                    │ 获取表结构  │  │ Python编码节点  │  │
│                    │(list_tables│  │(run_python_    │  │
│                    │  _tool)    │  │ script_tool)   │  │
│                    └─────┬──────┘  └───────┬────────┘  │
│                          │                 │           │
│                    ┌─────▼─────────────────▼────────┐  │
│                    │     data_analysis_agent        │  │
│                    │  (create_react_agent 子图)      │  │
│                    │  ┌─ db_sql_tool (SQL查询)       │  │
│                    │  ├─ run_python_script_tool      │  │
│                    │  ├─ translate_to_python_plot    │  │
│                    │  ├─ analysis_product_reviews    │  │
│                    │  ├─ reviews_stars_correlation   │  │
│                    │  └─ sales_predict_tool          │  │
│                    └────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ ywfl_mcp   │ │search_db   │ │python_chart│ │machine_    │
│ :9005      │ │_mcp :9004  │ │_mcp :9002  │ │learning_   │
│            │ │            │ │            │ │mcp :9003   │
│ 业务分流    │ │ Text2SQL   │ │ 代码执行   │ │ 机器学习    │
│            │ │ 数据库查询  │ │ 绘图生成   │ │ 分析建模    │
└────────────┘ └────────────┘ └────────────┘ └────────────┘
```

### 核心设计思想

- **MCP 服务解耦**：4个独立 MCP 服务各自负责单一职责，通过标准协议与主图通信，可独立开发、部署和扩缩容
- **LangGraph 编排**：使用 StateGraph 定义业务流程，通过条件边实现智能路由
- **Text2SQL**：LLM 自动将自然语言转换为 SQL 查询，支持多表联合、聚合分析
- **ReAct Agent 子图**：数据分析阶段使用 `create_react_agent` 实现多轮工具调用推理

## 目录结构

```
langGraph_agent/
├── .env                                           # 环境变量配置
├── requirements.txt                                # Python 依赖
├── build_graph.png                                 # LangGraph 流程图
│
├── project_data/                                   # 业务数据
│   ├── 商品表/
│   │   ├── food_price_list.csv                     # 食品价格表
│   │   ├── handicraft_price_list.csv               # 手工艺品价格表
│   │   ├── home_goods_price_list.csv               # 家居用品价格表
│   │   └── sports_good_price_list.csv              # 运动用品价格表
│   ├── 用户表/
│   │   ├── user information_table.csv              # 用户信息表
│   │   └── user_online_record.csv                  # 用户在线记录表
│   ├── 用户活跃表/
│   │   └── user_activity_stats.csv                 # 用户活跃统计表
│   └── 销量表/
│       ├── product_monthly_sales_statistic.csv     # 商品月销量统计表
│       ├── product_reviews.csv                     # 商品评论表
│       └── sales_detail_ records_table.csv         # 销售明细表
│
├── smart_data_analysis_assistant/                  # 核心应用
│   ├── mcp_server/                                 # MCP 服务端
│   │   ├── ywfl_mcp.py                             # 业务分流服务 (9005)
│   │   ├── statistic_db_mcp_tools.py               # 数据库查询服务 (9004)
│   │   ├── python_chart_mcp.py                     # 代码执行+绘图服务 (9002)
│   │   ├── machine_learning_mcp.py                 # 机器学习分析服务 (9003)
│   │   └── public_function.py                      # LLM 调用公共函数
│   │
│   └── chatbi_graph/                               # LangGraph 图构建
│       ├── build_graph.py                          # ★ 核心：图定义与编译
│       ├── execute_graph.py                        # 命令行交互入口
│       ├── chat_api.py                             # FastAPI 服务部署入口
│       ├── my_llm.py                               # LLM 初始化配置
│       ├── my_state.py                             # 图状态定义
│       ├── tools_node.py                           # 工具节点与系统提示词
│       └── post_test.py                            # API 接口测试脚本
│
└── langGraph_basic_learning/                       # LangGraph 学习笔记
    ├── langGraph_code.py                           # 框架核心概念学习
    ├── langGraph_practise.py                       # LangSmith 追踪练习
    └── mcp_example_server.py                       # MCP Server 示例
```

## 数据库设计 (PostgreSQL - sales_chat)

| 表名 | 说明 | 关键字段 |
|------|------|----------|
| `food_price_list` | 食品价格表 | product_no, product_name, product_price |
| `handicraft_price_list` | 手工艺品价格表 | product_no, product_name, product_price |
| `home_goods_price_list` | 家居用品价格表 | product_no, product_name, product_price |
| `sports_good_price_list` | 运动用品价格表 | product_no, product_name, product_price |
| `product_monthly_sales_statistic` | 商品月销量表 | product_name, 1月销量~12月销量 |
| `product_reviews` | 商品评论表 | date, reviews, stars, product_name |
| `user_information_table` | 用户信息表 | userid, username, register_date, self_description, occupation |
| `user_online_record` | 用户在线记录表 | userid, load_time, online_time, username |
| `sales_detail_records_table` | 销售明细表 | purchase_date, userid, product_name, username |
| `user_activity_stats` | 用户活跃统计表 | 用户id, 活跃时间, 活跃时长/分钟 |

## 环境准备

### 依赖要求

- Python >= 3.10
- PostgreSQL >= 14
- 阿里云百炼平台 API Key (DASHSCOPE_API_KEY)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

复制并编辑 `.env` 文件：

```bash
# PostgreSQL 数据库连接
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=sales_chat
DB_USER=postgres
DB_PASSWORD=your_password

# LLM 配置（也可通过系统环境变量 DASHSCOPE_API_KEY 配置）
QWEN_API_KEY=
QWEN_MODEL=qwen-plus
PREFER_TONGYI=1

# 图片输出目录
IMAGE_DIR=/path/to/output_plots
```

### 导入数据

1. 在 PostgreSQL 中创建数据库 `sales_chat`
2. 将 `project_data/` 下的 CSV 文件导入对应数据表（推荐使用 DBeaver 或 pgAdmin）

## 启动服务

项目需要启动 **4个 MCP 服务 + 1个主图**，共 5 个终端：

### 1. 启动业务分流 MCP (端口 9005)

```bash
cd smart_data_analysis_assistant/mcp_server
python ywfl_mcp.py
```

### 2. 启动数据库查询 MCP (端口 9004)

```bash
cd smart_data_analysis_assistant/mcp_server
python statistic_db_mcp_tools.py
```

### 3. 启动代码执行+绘图 MCP (端口 9002)

```bash
cd smart_data_analysis_assistant/mcp_server
python python_chart_mcp.py
```

### 4. 启动机器学习 MCP (端口 9003)

```bash
cd smart_data_analysis_assistant/mcp_server
python machine_learning_mcp.py
```

### 5. 启动主图 (命令行交互模式)

```bash
python smart_data_analysis_assistant/chatbi_graph/execute_graph.py
```

启动成功后会出现 `用户：` 提示符，即可开始对话。

### 可选：API 服务部署

```bash
python smart_data_analysis_assistant/chatbi_graph/chat_api.py
# 服务监听 0.0.0.0:9008
```

## 功能演示

### 数据查询

```
用户：查询健身手套价格是多少
→ 自动生成 SQL 查询 sports_good_price_list 表并返回价格
```

### 聚合分析

```
用户：运动类商品有多少
→ SELECT COUNT(*) FROM sports_good_price_list
```

### 数据可视化

```
用户：查询商品洗碗布的月销量数据，绘制一张以月为维度的销量柱状图
→ 1. 查询数据库获取月销量数据
→ 2. 生成 Python matplotlib 绘图代码
→ 3. 执行代码并返回图表路径
```

### 用户画像分析

```
用户：分析一下王一珂的用户画像
→ 查询用户信息、购买记录、在线时长，综合生成画像分析
```

### 销量预测

```
用户：查询保鲜袋历史12个月的销量，基于此预测未来的销量
→ 1. 查询月销量数据
→ 2. 使用指数移动平均(EMA)模型预测
```

### 评论情感分析

```
用户：分析银耳的用户评论和星级数据的相关性
→ 1. 提取评论 → LLM 情感分类(满意/中性/不满意)
→ 2. 计算 Pearson 相关系数
→ 3. 判断星级与满意度的相关性
```

### Python 编码

```
用户：写一段冒泡排序的Python代码
→ LLM 生成代码 → 自动执行 → 返回结果
```

### 多轮对话

```
用户：查询抽纸单价是多少？
助手：抽纸单价为 5.5 元。

用户：查询一下它过去12个月的销量
助手：（自动理解"它"指代"抽纸"，查询月销量数据）

用户：根据其12个月的销量绘制一张销量折线图
助手：（自动获取之前查询的数据，生成折线图）
```

## MCP 服务详情

### 1. 业务分流服务 (ywfl_mcp - 9005)

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `ywfl_tool` | 意图识别与任务分类 | user_input: 用户输入文本 |

分类结果：`业务数据查询分析` / `纯python编码`

### 2. 数据库查询服务 (search_db_mcp - 9004)

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `list_tables_tool` | 获取所有表结构（含注释） | 无 |
| `db_sql_tool` | 执行 SQL 查询 | query: SQL 语句 |

传输协议：SSE

### 3. 代码执行+绘图服务 (python_chart_mcp - 9002)

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `run_python_script_tool` | 执行 Python 代码（计算/绘图） | script_content: 代码字符串 |
| `translate_to_python_plot_script` | 生成 matplotlib 绘图代码 | graph_demand: 绘图需求, data_desc: 数据描述 |

### 4. 机器学习服务 (machine_learning_mcp - 9003)

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `analysis_product_reviews_tool` | 评论情感分析（满意/中性/不满意） | reviews_list: 评论列表 |
| `reviews_stars_correlation_test_tool` | 星级与满意度相关性分析 (Pearson) | ItemName, reviews, stars |
| `sales_predict_tool` | 销量预测 (指数移动平均) | ItemName, sales_data_list |

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM | 通义千问 (qwen-plus) / DeepSeek |
| LLM 框架 | LangChain, LangGraph |
| 工具协议 | MCP (Model Context Protocol) |
| 数据库 | PostgreSQL |
| Web 框架 | FastAPI + Uvicorn |
| 数据分析 | Pandas, NumPy, SciPy |
| 可视化 | Matplotlib |
| 运行环境 | Python 3.11+, Windows/Linux |

## LangGraph 图节点说明

| 节点 | 类型 | 功能 |
|------|------|------|
| `call_identify_intention` | 执行节点 | LLM + ywfl_tool 实现业务分流 |
| `identify_intention_tool_node` | 工具节点 | 执行业务分流 MCP 调用 |
| `call_list_tables` | 执行节点 | 构造获取表结构的工具调用 |
| `list_tables_tool` | 工具节点 | 获取数据库表结构信息 |
| `data_analysis_agent` | ReAct子图 | Text2SQL + 数据分析 + 可视化 + ML |
| `call_python_coder` | 执行节点 | LLM 生成 Python 代码 |
| `python_run_tool_node` | 工具节点 | 执行 Python 代码 |

## 常见问题

**Q: 启动 MCP 服务报端口占用**
```bash
# 查找占用进程
netstat -ano | findstr ":9004"
# 终止进程
taskkill /PID <PID> /F
```

**Q: 主图连接 MCP 失败**
确认 4 个 MCP 服务均已启动并显示 `Uvicorn running` 后，再启动主图。

**Q: Windows 下 MCP 服务报 [Errno 22]**
Windows 控制台 GBK 编码与 UTF-8 不兼容导致，避免在 MCP 服务端 print 中文数据。

**Q: 如何切换 LLM 模型**
修改 `.env` 中 `QWEN_MODEL` 即可，如 `qwen-turbo`（更快）或 `qwen-max`（更强）。

## License

MIT
