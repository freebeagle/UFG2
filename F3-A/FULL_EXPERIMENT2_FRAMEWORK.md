# full-experiment2 分支框架说明

本文档记录 `full-experiment2` 分支当前采用的 MCP + UFGraphFR 完整框架。该分支的核心目标是：在原 UFGraphFR 的 user-user graph 聚合框架上，引入 LLM user 表示与 item text branch，并使用 per-item 融合权重区分 ID 协同信号和文本属性信号。

## 总体结论

`full-experiment2` 当前代码与本分支设计一致：

- `embedding_item.weight` 是 per-item 参数，走 UFGraphFR user-user graph 聚合。
- `item_alpha` 是 `[num_items, 1]` 的 per-item 参数，初始化为 `-2.0`，与 `embedding_item.weight` 使用同一张 user-user graph 聚合。
- `item_text_proj.weight` 是 `[latent_dim, text_dim]` 的 per-model 全局语义投影参数，按行 L2 归一化后 FedAvg。
- `item_text_proj.bias` 是 `[latent_dim]` 的 per-model 参数，FedAvg。
- `embedding_user.weight` 是 user LLM embedding 到低维 user embedding 的线性映射权重，用于服务器端构建 user-user graph。

## Client 侧

### User 表示

MCP user 原始属性来自：

```text
data/mcp_prepare/user_side/raw/filtered_clients.json
```

该文件包含富属性 user/client 信息。`train.py` 的 `load_mcp_split()` 会将预划分交互数据与 user mapping 合并，再交给 `EmbeddingUtils`。当 `use_jointembedding=true` 时，每个客户端本地通过冻结文本编码器得到用户语义向量：

```text
filtered_clients.json -> frozen LLM / sentence embedder -> v_u [embed_dim]
```

随后模型使用：

```text
embedding_user = Linear(embed_dim, latent_dim)
```

将 `v_u` 投影为：

```text
e_u = v_u @ W_u + b_u
```

当前 MCP + MiniLM/HF_LOCAL 设置下，`embed_dim=384`，`latent_dim=32`。

相关代码：

- `train.py`: `load_mcp_split()`
- `embedding.py`: `EmbeddingUtils._build_mcp_prompt()` 与 `EmbeddingUtils.embedding_users()`
- `mymodel.py`: `self.embedding_user = torch.nn.Linear(...)`

### Item 表示

Item 有两条分支。

ID 分支：

```text
iid -> embedding_item -> e_id [latent_dim]
```

文本属性分支：

```text
iid -> item_text_features[iid] [text_dim]
    -> item_text_proj
    -> e_attr [latent_dim]
```

当前 MCP item text feature 文件为：

```text
data/mcp_prepare/item_side/features/item_text_features_A.npy
data/mcp_prepare/item_side/features/item_text_features_AB.npy
```

两者 shape 均为：

```text
[5837, 384]
```

相关代码：

- `mymodel.py`: `self.item_text_features`
- `mymodel.py`: `self.item_text_proj = torch.nn.Linear(...)`
- `mymodel.py`: `self.embedding_item = torch.nn.Embedding(...)`

### Item 双分支融合

`full-experiment2` 使用 per-item 可学习融合权重：

```text
item_alpha [num_items, 1]
```

初始化为：

```text
-2.0
```

因此：

```text
sigmoid(-2.0) ~= 0.12
```

初始阶段 item 表示更偏向 text branch，适合冷 item；随着本地训练进行，热 item 的梯度可以把对应 item 的 `item_alpha[iid]` 拉向 ID branch。

融合公式为：

```text
alpha = sigmoid(item_alpha[iid])
e_item = alpha * e_id + (1 - alpha) * e_attr
```

相关代码：

- `mymodel.py`: `self.item_alpha = torch.nn.Parameter(torch.full((self.num_items, 1), -2.0))`
- `mymodel.py`: `get_item_embedding()`

## Client -> Server 上传

每轮参与客户端上传以下参数：

| 参数 | Shape | 粒度 | 用途 |
| --- | --- | --- | --- |
| `embedding_item.weight` | `[num_items, latent_dim]` | per-item | item ID 协同信号 |
| `item_alpha` | `[num_items, 1]` | per-item | ID/text 融合权重 |
| `item_text_proj.weight` | `[latent_dim, text_dim]` | per-model | 全局文本语义投影 |
| `item_text_proj.bias` | `[latent_dim]` | per-model | 全局文本语义投影 bias |
| `embedding_user.weight` | `[latent_dim, embed_dim]` | per-client | 构建 user-user graph |

在当前 MCP 设置下：

```text
num_items = 5837
latent_dim = 32
text_dim = 384
```

因此关键 shape 为：

```text
embedding_item.weight: [5837, 32]
item_alpha:            [5837, 1]
item_text_proj.weight: [32, 384]
item_text_proj.bias:   [32]
embedding_user.weight: [32, 384]
```

相关代码：

- `engine.py`: `round_participant_params[user]['embedding_item.weight']`
- `engine.py`: `round_participant_params[user]['item_text_proj.weight']`
- `engine.py`: `round_participant_params[user]['item_text_proj.bias']`
- `engine.py`: `round_participant_params[user]['item_alpha']`
- `engine.py`: `round_participant_params[user]['embedding_user.weight']`

## Server 聚合

### User-user graph 构建

当 `use_jointembedding=true` 且 alias 为 `UFGraphFR` 时，服务器端使用各客户端上传的 `embedding_user.weight` 构建 user-user graph：

```text
W_u[1], W_u[2], ..., W_u[N] -> user-user graph
```

相关代码：

- `engine.py`: `aggregate_clients_params_user()`
- `utils.py`: `construct_user_relation_graph_via_user()`
- `utils.py`: `select_topk_neighboehood()`

### 同一张图聚合 per-item 参数

同一张 user-user graph 用于聚合：

```text
embedding_item.weight [num_items, latent_dim]
item_alpha            [num_items, 1]
```

`embedding_item.weight` 通过原 UFGraphFR 的 `MP_on_graph()` 聚合。

`item_alpha` 通过通用参数图聚合函数 `MP_on_graph_param()` 聚合，shape 固定为：

```text
[num_items, 1]
```

相关代码：

- `engine.py`: `updated_item_embedding = MP_on_graph(...)`
- `engine.py`: `updated_alpha = MP_on_graph_param(..., 'item_alpha', (num_items, 1), ...)`
- `utils.py`: `MP_on_graph()`
- `utils.py`: `MP_on_graph_param()`

### item_text_proj 聚合

`item_text_proj` 是 per-model 全局投影，不作为 per-user/per-item 个性化图参数下发。

聚合策略：

```text
item_text_proj.weight: per-row L2 normalize -> FedAvg
item_text_proj.bias:   FedAvg
```

这样保留 FeDecider 启发的 direction/scale 解耦思想：投影矩阵每一行的方向作为共享语义信息，行范数中的尺度差异被削弱；但最终仍维持一个全局一致的语义投影空间。

相关代码：

- `engine.py`: `_aggregate_item_text_params()`

## Server -> Client 下发

下发逻辑按参数粒度区分：

| 参数 | 下发方式 |
| --- | --- |
| `embedding_item.weight` | 按 user 下发图聚合后的 per-user item embedding |
| `item_alpha` | 按 user 下发图聚合后的 per-user item alpha |
| `item_text_proj.weight` | 下发全局 FedAvg 后的 tensor |
| `item_text_proj.bias` | 下发全局 FedAvg 后的 tensor |

相关代码：

- `engine.py`: `user_param_dict['embedding_item.weight'] = self.server_model_param['embedding_item.weight'][user]`
- `engine.py`: `user_param_dict['item_alpha'] = self.server_model_param['item_alpha'][user]`
- `engine.py`: `user_param_dict['item_text_proj.weight'] = self.server_model_param['item_text_proj.weight']`
- `engine.py`: `user_param_dict['item_text_proj.bias'] = self.server_model_param['item_text_proj.bias']`

## 设计理由

| 参数 | 聚合 | 原因 |
| --- | --- | --- |
| `embedding_item.weight` | 图聚合 | per-item 协同信号，依赖用户交互分布，是 UFGraphFR 原有验证对象 |
| `item_alpha` | 图聚合 | per-item 融合权重；相似用户对同一 item 的 ID/text 依赖程度可以共享 |
| `item_text_proj.weight` | per-row L2 + FedAvg | per-model 全局语义投影应保持一致；L2 保留方向、削弱尺度噪声 |
| `item_text_proj.bias` | FedAvg | per-model bias，不需要按 item 或按 user 图聚合 |
| `embedding_user.weight` | 建图用 | UFGraphFR 的 user-user graph 构建依据 |

## 本分支创新点

1. 双分支 item 表示：ID embedding + LLM/text attribute embedding。
2. per-item 可学习融合权重：冷 item 初始偏 text，热 item 可被梯度拉向 ID。
3. FeDecider 启发的方向归一化：对 `item_text_proj.weight` 做 per-row L2，再 FedAvg。
4. 统一使用 `W_u` 构建的 user-user graph 聚合 per-item 参数：`embedding_item.weight` 与 `item_alpha`。

## 当前实现注意点

当前代码沿用了 UFGraphFR 原有 graph 工具：当 `similarity_metric='cosine'` 时，`sklearn.metrics.pairwise_distances(..., metric='cosine')` 返回的是 cosine distance，而不是 cosine similarity。

这意味着：

```text
值越小 -> 越相似
值越大 -> 越不相似
```

但当前 `select_topk_neighboehood()` 使用 `argsort()[-k:]` 选择较大的值作为邻居。也就是说，如果 metric 是 cosine，当前实现可能会选到距离更大的用户。这个问题属于 graph 构建方向的实现细节，不影响上面参数通路是否已按本分支框架接好；但如果之后修正为选择距离更小的邻居，user-user graph 会变化，进而影响 `embedding_item.weight` 和 `item_alpha` 的聚合结果，所以实验指标可能发生明显变化。

因此建议：

```text
先保留当前 full-experiment2 结果作为当前实现版本的实验记录。
如果修 graph neighbor 选择逻辑，应单独开一次实验，并在结果中标注为 graph similarity fix 后的版本。
```

## 局部验证建议

按当前多分支工作约束，本分支验证应优先使用局部功能检查，而不是直接跑完整训练作为唯一证明。建议检查：

1. `item_alpha` shape 是否为 `[5837, 1]`。
2. `get_item_embedding()` 输出是否为 `[batch_size, 32]`。
3. `_aggregate_item_text_params()` 后：
   - `item_text_proj.weight` shape 为 `[32, 384]`
   - `item_text_proj.bias` shape 为 `[32]`
   - `item_alpha[user]` shape 为 `[5837, 1]`
4. Server 下发时：
   - `item_text_proj.*` 使用全局 tensor
   - `item_alpha` 使用 per-user graph 聚合结果

