# MCP-UFGraphFR 后续分支需求文档

本文档用于交接 MCP-UFGraphFR 后续开发需求。下一个对话应优先阅读本文档，再决定继续实现哪个分支。

## 0. 最高优先级约束

当前项目采用多分支递进开发方式。每个分支只证明一件事。

在用户明确允许完整集成测试之前：

```text
只测试当前分支自身功能。
不要跑完整训练。
不要把 num_round=1 完整训练作为单个分支的主要通过标准。
不要把多个创新点混在一个分支里一起验证。
```

完整训练和最终 `num_round=1` 集成 smoke 只能在所有功能分支完成后再执行。

同时，不要随意删除、移动、覆盖或清理：

```text
D:\EEE\vscode\UFGraphFR
```

尤其不要移动或删除 MCP 原始数据集文件。若需要复制数据，必须先询问用户，并使用“复制”，不要使用“移动”。

## 1. 当前研究目标

当前目标不是简单地给 UFGraphFR 加 item 文本特征，而是在原 UFGraphFR 框架上形成一个面向 MCP 场景的新模型。

建议最终研究定位为：

```text
面向 MCP 的图引导双分支方向校准联邦推荐框架
```

可选英文名称：

```text
Graph-Guided Dual-Branch Directional Calibration for MCP Federated Recommendation
```

核心组合关系：

```text
UFGraphFR:
  提供用户关系图和图引导 item 聚合骨架。

MCP item 双分支:
  提供 item ID 协同分支 + item 文本属性分支。

FeDecider 思想:
  提供方向归一化 / 方向成分共享，用于缓解联邦异质客户端的参数尺度噪声。
```

最终目标不是完整照搬 FeDecider，而是吸收其方向归一化思想，将其嵌入 UFGraphFR 的图聚合流程中。

## 2. 原 UFGraphFR 框架

原 UFGraphFR 的核心不是简单使用文本 embedding，而是：

```text
用户结构化属性
-> prompt
-> 冻结 PLM encoder
-> 静态用户语义向量 v_u
-> 客户端本地训练 joint embedding layer
-> 得到本地交互适配后的 W_u
-> 上传 W_u
-> Server 用 W_u 构建 user-user relationship graph
-> Server 用用户图聚合 item embedding
-> 下发 global item embedding
```

原 item 侧比较简单：

```text
iid
-> item ID embedding
-> e_item
```

原 Transformer 是：

```text
Temporal Transformer
```

它建模有时间戳数据中的用户交互序列依赖。

MCP 数据缺乏可靠 timestamp，因此后续不能声称 MCP Transformer 建模 temporal dependency。MCP 中应改写为：

```text
Interaction-Context Transformer
```

## 3. 已完成的分支

### 3.1 user 侧 MCP + HF/LLM 分支

分支：

```text
codex/mcp-user-llm-data-integration
```

已提交：

```text
4e61ef2 feat: integrate mcp user llm embeddings
```

目标：

```text
filtered_clients.json
+ filtered_client_id_mapping.csv
-> user prompt text
-> HF/LLM encoder
-> v_u
-> joint embedding layer
-> W_u
```

论文逻辑：

```text
v_u = frozen PLM/HF/LLM(prompt_u)
e_u = v_u W_u + b
```

预测分支：

```text
v_u
-> Linear(v_u) = e_u
-> UserMLP
-> prediction
```

上传 / 建图分支：

```text
训练后的 W_u
-> 上传到 Server
-> 构建 user-user graph
```

注意：按原文，server 主要使用 joint embedding weight matrix `W_u` 建图，不是直接使用静态 `v_u`。

### 3.2 C 分支：mcp-item-text

分支：

```text
codex/mcp-item-text
```

已提交：

```text
2cc3abe feat: generate mcp item text features
```

目标：

```text
只证明 item 文本特征能生成并对齐 iid。
```

流程：

```text
filtered_servers.json
+ filtered_servers_id_mapping.csv
-> item_attribute_text_A.csv
-> item_attribute_text_AB.csv
-> all-MiniLM-L6-v2 encoder
-> item_text_features_A.npy
-> item_text_features_AB.npy
```

当前产物：

```text
data/mcp_prepare/item_side/features/item_text_features_A.npy
data/mcp_prepare/item_side/features/item_text_features_AB.npy
```

功能测试结果：

```text
mapping rows = 5837
iid range = 0..5836
iid unique = 5837
full coverage = True

A:
  npy_shape = (5837, 384)
  iid_order = 0..5836
  zero_rows = 0

AB:
  npy_shape = (5837, 384)
  iid_order = 0..5836
  zero_rows = 0
```

这里的 `item_text_features` 是静态句向量矩阵：

```text
[num_items, text_dim] = [5837, 384]
```

运行时按 iid 查询：

```text
item_text_features[iid] -> [batch_size, 384]
```

它不是：

```text
[batch_size, seq_len, hidden_dim]
```

### 3.3 D0 分支：mcp-item-branch

分支：

```text
codex/mcp-item-branch
```

已提交：

```text
58fd377 feat: add mcp item attribute branch
```

当前 D0 目标：

```text
只证明 item_text_features 能接进模型，并形成双分支 item 表示。
```

当前逻辑：

```text
iid
-> item ID embedding
-> e_id
```

```text
iid
-> item_text_features[iid]
-> item_text_proj
-> e_attr
```

```text
e_id + alpha * e_attr
-> e_item
```

已完成的功能测试：

```text
A:
  text_features = (5837, 384)
  e_id = (3, 32)
  e_attr = (3, 32)
  e_item = (3, 32)
  fusion_ok = True

AB:
  text_features = (5837, 384)
  e_id = (3, 32)
  e_attr = (3, 32)
  e_item = (3, 32)
  fusion_ok = True
```

额外回归检查：

```text
use_item_attribute = False
-> e_item 等于原始 item ID embedding
```

这只是证明新增分支关闭时可以回退到原 UFGraphFR item 表示，不是 D0 的主实验配置。

## 4. 需要调整的关键点

当前 D0 的问题是：

```text
item_text_proj 是可训练层。
```

如果每个客户端都本地训练自己的 `item_text_proj`，但该层不上传、不聚合、不下发，则会产生投影空间漂移：

```text
同一个 item_text_features[iid]
-> client A 的 item_text_proj_A
-> e_attr_A

同一个 item_text_features[iid]
-> client B 的 item_text_proj_B
-> e_attr_B
```

结果：

```text
不同客户端的 e_attr 不在同一 32 维语义坐标系
-> e_item = e_id + alpha * e_attr 出现跨客户端错位
-> e_id 被迫适配本地漂移后的 e_attr
-> 上传到 Server 的 item embedding 携带本地投影偏差
-> UFGraphFR 的图聚合前提被破坏
```

所以 D0 只能作为：

```text
local item text branch baseline
```

不能作为最终方法。

最终方法中：

```text
item_text_features:
  静态，不训练，不上传。

item_text_proj:
  可训练，需要纳入联邦共享、聚合和下发。
```

## 5. 后续分支设计

### D1. mcp-item-proj-communication

建议新分支：

```text
codex/mcp-item-proj-communication
```

从：

```text
codex/mcp-item-branch
```

派生。

目标：

```text
只证明 item_text_proj 能进入联邦通信闭环。
```

需要实现：

```text
客户端本地训练后保存 item_text_proj.weight
客户端本地训练后保存 item_text_proj.bias
round_participant_params 中包含 item_text_proj 参数
Server 能接收这些参数
Server 能产生 global item_text_proj
客户端下一轮能加载 global item_text_proj
```

这一阶段先不要做方向归一化。

这一阶段也不要改 Transformer。

本分支功能测试：

```text
构造 mock client 参数
检查 item_text_proj.weight shape = [32, 384]
检查 item_text_proj.bias shape = [32]
检查 Server 聚合后 shape 不变
检查模型能加载 global item_text_proj
检查 get_item_embedding 前向 shape 正确
```

不要跑完整训练。

### D2. mcp-graph-guided-dual-aggregation

建议新分支：

```text
codex/mcp-graph-guided-dual-aggregation
```

从 D1 派生。

目标：

```text
只证明 Server 使用同一张 user-user graph 同时聚合两条 item 侧轨道。
```

两条轨道：

```text
轨道 A:
  item ID embedding

轨道 B:
  item_text_proj
```

逻辑：

```text
W_u
-> user-user graph
-> graph-guided aggregation(item_embedding)
-> graph-guided aggregation(item_text_proj)
```

本分支功能测试：

```text
构造 mock user graph
构造 mock item_embedding 参数
构造 mock item_text_proj 参数
检查两个轨道聚合输出 shape 正确
检查用户 id 到矩阵行再回映射不乱
检查 item_text_proj.weight / bias 能正确聚合
```

不要跑完整训练。

### D3. mcp-directional-proj-calibration

建议新分支：

```text
codex/mcp-directional-proj-calibration
```

从 D2 派生。

目标：

```text
只证明 item_text_proj 聚合前的方向归一化 / 方向校准有效。
```

借鉴 FeDecider 的思想，但不要完整照搬 FeDecider。

核心思想：

```text
不直接聚合原始 item_text_proj 参数；
先削弱不同客户端参数尺度差异；
保留主要方向信息；
再用用户关系图聚合。
```

本分支功能测试：

```text
构造两个同方向但不同范数的 projection 参数
方向归一化后它们方向一致，尺度影响被削弱

构造两个不同方向的 projection 参数
方向归一化后仍保留方向差异

检查 graph-guided aggregation 输出 shape 正确
检查无 NaN / inf
```

不要跑完整训练。

### E. mcp-interaction-context-transformer

建议等 D1/D2/D3 完成后再做。

目标：

```text
将原 Temporal Transformer 改为 MCP 场景下的 Interaction-Context Transformer。
```

不要声称 temporal dependency。

目标设计：

```text
Q = candidate item
K = user train-history context items
V = user train-history context items
```

具体流程：

```text
candidate iid
-> enhanced item embedding e_item(candidate)
-> Q
```

```text
user train positives
-> history iids
-> enhanced item embeddings
-> K/V
```

严格要求：

```text
user_context_items 只能来自 train positives
不能泄露 validation/test item
支持 max_context_len
支持 padding
支持 mask
```

本分支功能测试：

```text
检查某个 uid 的 context 只来自 train.csv
检查 context 不包含 val/test positives
检查 Q shape 正确
检查 K/V shape 正确
检查 mask shape 正确
检查 attention forward 正确
```

不要跑完整训练。

## 6. 最终完整模型流程

完整方法最终应为：

```text
MCP user side:
filtered_clients.json
+ filtered_client_id_mapping.csv
-> user prompt
-> HF/LLM encoder
-> v_u
-> joint embedding layer
-> e_u = v_u W_u + b
-> 上传 W_u
```

```text
MCP item side:
filtered_servers.json
+ filtered_servers_id_mapping.csv
-> item attribute text
-> frozen MiniLM encoder
-> item_text_features
```

```text
本地 item 双分支:
iid
-> item ID embedding
-> e_id

iid
-> item_text_features[iid]
-> item_text_proj
-> e_attr

e_id + alpha * e_attr
-> e_item
```

```text
Server:
W_u
-> user-user graph
-> aggregate item ID embedding
-> directional-calibrated aggregate item_text_proj
-> 下发 global item embedding + global item_text_proj
```

```text
Interaction-Context Transformer:
candidate e_item
-> Q

train-history e_item sequence/set
-> K/V

candidate-to-context attention
-> prediction
```

## 7. 实验与消融建议

后续完整训练阶段可以设计以下模型对比，但不要在单分支阶段跑完整训练：

```text
1. UFGraphFR original-style baseline on MCP
2. UFGraphFR + item text branch, local item_text_proj only
3. UFGraphFR + item text branch, FedAvg item_text_proj
4. UFGraphFR + item text branch, graph-guided item_text_proj aggregation
5. UFGraphFR + item text branch, graph-guided directional item_text_proj aggregation
6. Full model + Interaction-Context Transformer
```

关键消融：

```text
w/o item text branch
w/o item_text_proj federation
w/o graph-guided projection aggregation
w/o directional calibration
static v_u graph instead of trained W_u graph
FedAvg aggregation instead of graph aggregation
A vs AB item attribute text
different alpha
MiniLM vs other text encoders
```

稳定性分析：

```text
item_text_proj 参数范数变化
方向归一化前后的聚合稳定性
不同客户端 e_attr 分布是否漂移
训练收敛曲线
稀疏 client / 长尾 item 表现
```

## 8. 下一步建议

当前所在分支：

```text
codex/mcp-item-branch
```

当前分支已经完成 D0。

下一步建议新建：

```text
codex/mcp-item-proj-communication
```

只做 D1：

```text
item_text_proj 上传 / 聚合 / 下发闭环
```

测试只做 D1 分支功能测试，不跑完整训练。

