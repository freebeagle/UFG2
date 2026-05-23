# MCP-UFGraphFR Transformer 分支工作总结

## 当前分支

当前工作分支：

```text
codex/mcp-interaction-context-transformer
```

该分支基于 `mcp-data-protocol`，保留前序 MCP 数据协议修改。

## 已完成任务

### 1. 创建 Transformer 分支

已从 `mcp-data-protocol` 创建并切换到：

```text
codex/mcp-interaction-context-transformer
```

### 2. 整理 MCP 预处理目录

已创建：

```text
data/mcp_prepare/
  item_side/
    raw/
    mapping/
    features/
  user_side/
    raw/
    mapping/
  interactions/
```

已复制并检查：

```text
item_side/raw/filtered_servers.json
item_side/mapping/filtered_servers_id_mapping.csv

user_side/raw/filtered_clients.json
user_side/mapping/filtered_client_id_mapping.csv

interactions/train.csv
interactions/vali.csv
interactions/test.csv
interactions/filtered_client_id_mapping.csv
```

检查结果：

```text
item_mapping: 5837 items, iid 0-5836
user_mapping: 186 users, uid 0-185
train.csv: 27027 rows, uid 0-185, iid 0-5836
vali.csv: 167 rows, uid 1-185, iid 22-5812
test.csv: 175 rows, uid 1-185, iid 22-5809
```

### 3. 已生成 item attribute text

已生成：

```text
data/mcp_prepare/item_side/features/item_attribute_text_A.csv
data/mcp_prepare/item_side/features/item_attribute_text_AB.csv
data/mcp_prepare/item_side/features/item_feature_meta.json
```

说明：

```text
filtered_servers.json: 5838 条
filtered_servers_id_mapping.csv: 5837 条
重复原始 server id: 7520
处理方式: 跳过重复 id，保证最终 5837 行与 iid 对齐
```

### 4. 已创建脚本

已创建：

```text
prepare_mcp_item_features.py
```

作用：

```text
filtered_servers.json
+ filtered_servers_id_mapping.csv
-> item_attribute_text_A.csv
-> item_attribute_text_AB.csv
-> item_text_features_A.npy
-> item_text_features_AB.npy
```

目前 `.csv` 已生成，`.npy` 尚未生成。

## 已讨论并确定的核心思路

### 1. Item 侧设计

MCP server 被看作推荐系统中的 item。

item 表示由双分支组成：

```text
item ID embedding branch
+ item attribute embedding branch
```

保留原 UFGraphFR 的 item ID embedding：

```text
e_id = embedding_item(iid)
```

新增 item attribute branch：

```text
server attributes
-> attribute text
-> PLM encoder
-> item_text_features[iid]
-> item_text_proj
-> e_attr
```

第一版融合方式：

```text
e_item = e_id + alpha * e_attr
```

暂不做 gate、复杂 GNN 或复杂 fusion。

### 2. Item 属性版本

只保留两个版本：

```text
A:
name
title
category
description
tags
author_name
url

A+B:
A + GitHub attributes
```

GitHub attributes 包括：

```text
github.full_name
github.language
github.languages
github.license
github.stargazers_count
github.forks_count
github.contributors_count
github.open_issues_count
github.archived
github.has_docker
github.has_readme
github.has_requirements
github.last_commit
```

明确舍弃：

```text
tools
server_command
server_config
sse_url
```

原因：

```text
填充率低
噪声大
存在配置/API key/token/secret 泄露风险
```

### 3. item_text_features.npy 的 PLM

第一版推荐使用：

```text
sentence-transformers/all-MiniLM-L6-v2
```

Hugging Face 地址：

```text
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

建议本地下载到：

```text
D:\models\all-MiniLM-L6-v2
```

生成命令：

```powershell
$env:PYTHONPATH = ".tmp_pydeps"
python prepare_mcp_item_features.py `
  --servers_json data\mcp_prepare\item_side\raw\filtered_servers.json `
  --id_mapping data\mcp_prepare\item_side\mapping\filtered_servers_id_mapping.csv `
  --output_dir data\mcp_prepare\item_side\features `
  --attribute_sets A AB `
  --model_name D:\models\all-MiniLM-L6-v2
```

当前本地已有模型：

```text
D:\models\bge-small-zh-v1.5
D:\models\e5-small-v2
```

但第一版仍建议使用 `all-MiniLM-L6-v2`，因为 MCP server 描述主要是英文，且与 UFGraphFR 的 PLM/MiniLM 叙事更一致。

### 4. Transformer 设计

原 UFGraphFR 是：

```text
Temporal Transformer
```

因为原论文数据有 timestamp。

MCP 没有可靠 timestamp，因此不能声称建模 temporal dependency。

MCP 中改为：

```text
Interaction-Context Transformer
```

最终确定 Q/K/V 设计：

```text
Q = candidate item
K = user interaction context
V = user interaction context
```

也就是：

```text
q = e_item(candidate_iid)
K,V = [e_item(history_iid_1), ..., e_item(history_iid_L)]
```

含义：

```text
候选 MCP server 查询用户历史交互过的 MCP servers，
学习候选 server 与用户历史 context 的功能关联、共现关系和语义关联。
```

这不是 batch 内 attention，也不是 temporal transformer。

### 5. User 侧 LLM 分支

已有 `exp-llm-dataset` 分支做过：

```text
user prompt text
-> PLM/USE embedding
改成
-> MiniLM-L6 / HF_LOCAL / LLM embedding
```

该分支可复用 encoder 部分，但不能整体合并，因为它没有 MCP 数据协议、真实 uid 对齐和 split 逻辑。

最终 user 侧路线：

```text
user prompt
-> LLM/HF embedding v_u
-> joint embedding layer
-> local training adapts W_u
-> server uses trained W_u to construct user-user graph
```

保持 UFGraphFR 核心：

```text
不直接使用静态 v_u 建图
仍使用训练后的 W_u 建图
```

## 待完成任务

### Step 2 剩余部分

生成：

```text
item_text_features_A.npy
item_text_features_AB.npy
```

依赖：

```text
D:\models\all-MiniLM-L6-v2
```

### Step 3

接入 item attribute embedding branch：

```text
mymodel.py
  加载 item_text_features.npy
  注册 frozen item feature buffer
  增加 item_text_proj
  实现 e_item = e_id + alpha * e_attr
```

同时在 `train.py` 增加参数：

```text
--use_item_attribute
--item_attribute_set A/AB
--mcp_item_feature_path
--item_attribute_alpha
```

### Step 4

融合 user LLM encoder 分支：

```text
从 exp-llm-dataset 吸收：
--pre_model HF_LOCAL
--hf_embedding_model
embed_dim auto infer
```

但保留当前分支的 MCP 数据协议和 uid 对齐逻辑。

### Step 5

构造 user interaction context：

```text
user_context_items[uid] = 用户训练集中交互过的 iid list
```

要求：

```text
只使用 train positives
不能泄露 validation/test positives
支持 max_context_len
支持 padding 和 mask
无 timestamp 时先使用 iid ascending 固定顺序
```

### Step 6

改造 Transformer：

```text
当前:
batch item embeddings self-attention

目标:
candidate-to-context attention
Q = candidate item
K/V = user history context
```

### Step 7

实验消融：

```text
UFGraphFR-MCP baseline
+ item attribute A
+ item attribute A+B
+ user LLM encoder
+ item attribute + user LLM
+ Interaction-Context Transformer
without Transformer
without graph aggregation
static v_u graph vs trained W_u graph
```

## 接下来的执行顺序

```text
1. 下载 all-MiniLM-L6-v2 到 D:\models\all-MiniLM-L6-v2
2. 生成 item_text_features_A.npy / item_text_features_AB.npy
3. 接 item attribute embedding branch
4. 合并 user LLM encoder 能力
5. 构造 user_context_items
6. 改 Transformer 为 Q=candidate, K/V=context
7. 做编译检查和最小训练验证
```
