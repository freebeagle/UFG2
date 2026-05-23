# MCP 数据集接入 UFGraphFR 的创新思路总结

## 目标

目标是在 UFGraphFR 论文框架基础上接入 MCP 推荐数据集，同时保留 UFGraphFR 的核心优势，并引入适合 MCP 场景的语义增强创新。

后续开发建议以 `D:\EEE\vscode\UFGraphFR\UFGraphFR` 作为对照组和基础工程，不建议继续沿用之前 PyCharm 侧探索版本，因为该版本中混入了较多临时修改，容易导致训练和评估错位。

## 对 UFGraphFR 核心思想的理解

UFGraphFR 主要有两条技术主线：

1. 基于用户文本特征的图联邦推荐。
2. 基于 Transformer 的本地推荐建模增强。

论文真正的核心创新不只是“使用文本 embedding”，而是：

- 将用户结构化属性通过 prompt 转换为自然语言描述。
- 使用冻结的 PLM 将用户文本编码为静态语义向量 `v_u`。
- 在客户端使用本地交互数据训练 joint embedding 线性层。
- 服务器不直接使用原始文本和原始交互，而是使用训练后的用户侧 joint embedding 权重 `W_u` 构建 user-user 关系图。
- 服务器基于该用户关系图聚合各客户端上传的 item embedding，并将更新后的全局 item embedding 下发给客户端。

因此，即使将 PLM 替换为 LLM，用户图构建也最好仍然基于训练后的 `W_u`，而不是直接使用静态 LLM 向量 `v_u`。论文消融实验中也说明，直接使用原始文本向量 `v_u` 建图通常不如使用经过本地交互适配后的 `W_u`。

## 总体模型方向

建议将 MCP 版本的模型定位为：

```text
面向 MCP 的双侧语义增强图联邦推荐
```

也可以表述为：

```text
Bidirectional Semantic Graph Federated Recommendation for MCP
```

核心扩展是：从原 UFGraphFR 的“仅用户侧语义增强”，扩展为“用户侧 + 物品侧双侧语义增强”。

UFGraphFR 论文最后明确提到，当前模型只关注用户侧文本特征，未来可以引入物品侧文本特征。因此，MCP 数据集中 user 侧和 item 侧都有丰富文本属性这一点，正好可以作为对 UFGraphFR 的自然扩展，而不是偏离论文框架。

## 创新点一：用户侧 LLM 编码器

将原论文中的 PLM 用户文本编码器替换或扩展为 LLM 用户语义编码器。

建议表述为：

- 原 UFGraphFR 使用 USE、MiniLM、T5、TinyBERT、LaBSE 等 PLM。
- MCP 版本引入 LLM 编码器，以获得更强的用户语义表示。
- LLM 输出仍然可以记作静态用户语义向量 `v_u`。
- 图构建仍然使用本地训练后的 joint embedding 权重 `W_u`。

需要区分：

- `v_u`：静态语义信息。
- `W_u`：经过本地交互数据适配后的语义偏好信息。

论文强调 `W_u` 同时包含静态语义和动态偏好，因此更适合用于用户关系图构建。

## 创新点二：物品侧文本属性矩阵

MCP item 侧也有丰富描述和标签，因此可以引入物品侧文本属性矩阵。

第一版建议设计为：

```text
item 文本属性
  -> item 文本编码器
  -> item_text_feature_matrix: num_items x text_dim
  -> 可训练投影层: text_dim -> latent_dim
  -> item 语义 embedding
```

然后与原有 item ID embedding 融合：

```text
item_final = item_id_embedding + alpha * item_text_projection
```

也可以做其他融合方式：

```text
item_final = MLP([item_id_embedding || item_text_projection])
```

```text
item_final = gate * item_id_embedding + (1 - gate) * item_text_projection
```

推荐实现顺序：

1. 先做加法融合。
2. 再尝试拼接后过 MLP。
3. 最后再考虑门控融合。

第一阶段不建议直接用 item 文本去构建用户图。更稳妥的路线是先将 item 语义注入 item 表示，同时保留 UFGraphFR 原本的用户图聚合逻辑。

## 创新点三：保留 Transformer，但调整语义解释

Transformer 是 UFGraphFR 论文的特色之一，但论文中 Transformer 的作用是建模用户交互序列中的时间依赖关系。

MCP 数据集没有可靠时间戳。因此，不能直接声称 Transformer 在 MCP 中建模 temporal dependencies。

更合适的处理方式是：

```text
原 UFGraphFR：Temporal Transformer
MCP-UFGraphFR：Interaction-Context Transformer
```

在 MCP 场景中，Transformer 可以保留，但它建模的是：

- 用户已交互 item 之间的上下文关系；
- item 共现关系；
- item 语义关联；
- 用户交互集合中的高阶依赖。

而不是严格意义上的时间依赖。

可选输入顺序：

- 原始文件顺序：仅当该顺序有业务含义时使用。
- item ID 顺序：可复现，但语义较弱。
- rating 或交互强度排序。
- 基于 item 文本相似度的排序。
- 使用 Set Transformer 或 attention pooling，弱化顺序依赖。

推荐实验表述：

如果 Transformer 在 MCP 上提升效果，可以说明它在无时间戳场景下仍然能够捕获交互上下文中的 item 关联和语义共现关系。不要说它捕获时间依赖，除非 MCP 数据中确实存在可靠的时间或顺序信息。

## 创新点四：无时间戳 MCP 数据的划分协议

UFGraphFR 论文采用严格的 within-user leave-one-out 时间序列划分：

- 最新交互作为 test item。
- 次新交互作为 validation item。
- 更早交互作为 train data。

MCP 没有可靠时间戳，因此不建议伪造 timestamp 来硬套原协议。

推荐将 MCP 视为静态隐式反馈推荐数据集：

- 按用户内部划分，而不是全局划分。
- 保留“一个用户对应一个客户端”的联邦推荐假设。
- 只评估拥有有效验证或测试正样本的用户。

建议划分规则：

```text
如果某用户交互数 >= 3：
    划分为 train / validation / test
如果某用户交互数 == 2：
    一条用于 train，一条用于 validation 或 test
如果某用户交互数 == 1：
    只放入 train，不参与 ranking evaluation
```

你原本设想的 `80% train, 20% holdout` 可以使用，但要在每个用户内部做，而不是对全体交互做全局顺序划分。

holdout 可以继续拆成：

```text
20% holdout = 70% validation + 30% test
```

或者简化为：

```text
80% train / 10% validation / 10% test
```

## 评估协议

为了尽量贴近 UFGraphFR 论文，建议继续使用：

- HR@K
- NDCG@K

候选集构造也可以沿用论文方式：

```text
1 个正样本 item + 99 个负样本 item
```

但评估时必须使用真实 user ID，而不能将测试集中的行号当成 user ID。

正确逻辑应是：

```text
real_user = test_users[idx]
test_item = test_items[idx]
```

再用 `real_user` 去取对应客户端模型、用户 embedding、用户文本向量等。

对于 MCP，也可以额外报告：

- Recall@10
- 稀疏用户上的表现
- 冷启动或弱交互用户表现

## 推荐实验路线

由于这里同时引入了 MCP 数据、用户 LLM、item 文本矩阵、Transformer 语义调整等多个变化，实验必须分阶段做。

推荐路线：

1. MCP 基础适配。
   - 在 VSCode 对照组 UFGraphFR 项目中添加 MCP 数据加载。
   - 使用按用户内部划分的静态 split。
   - 尽量保持原 UFGraphFR 模型不变。
   - 修正训练和评估中的 user ID 对齐问题。

2. 用户侧 LLM 编码器。
   - 比较 PLM user encoder 和 LLM user encoder。
   - 这一阶段先不要加入 item 文本。

3. 物品侧文本属性矩阵。
   - 加入 item text feature matrix。
   - 比较不使用 item 文本和使用 item 文本的效果。

4. item 融合方式消融。
   - 加法融合。
   - 拼接后 MLP 融合。
   - 门控融合。

5. Transformer 消融。
   - 去掉 Transformer。
   - 使用固定顺序的 Interaction-Context Transformer。
   - 可选：Set Transformer 或 attention pooling。

6. 完整模型。
   - 用户侧 LLM。
   - 物品侧文本属性矩阵。
   - 图引导 item embedding 聚合。
   - Interaction-Context Transformer。

## 建议消融实验表

推荐设置如下模型变体：

```text
GPFedRec baseline
UFGraphFR original-style baseline on MCP
UFGraphFR + User LLM
UFGraphFR + Item Text
UFGraphFR + User LLM + Item Text
UFGraphFR + User LLM + Item Text, without Transformer
UFGraphFR + User LLM + Item Text, without graph aggregation
UFGraphFR + User LLM + Item Text, using static v_u graph instead of W_u graph
```

最后一个变体很重要，因为论文强调 `W_u` 比静态 `v_u` 更适合建图。这个实验可以证明你的 MCP 版本是否仍然符合 UFGraphFR 的核心假设。

## 实现注意事项

最重要的问题是 user ID 对齐。

原框架默认一个用户对应一个客户端。MCP 数据经过按用户划分、过滤稀疏用户、排除部分评估用户后，不能再默认 list 下标等于 user ID。

推荐数据结构：

```text
train_data_by_user[user_id] = 当前用户的训练数据
val_data 中保存真实 user_id 和 item_id
test_data 中保存真实 user_id 和 item_id
```

评估时：

```text
real_user = test_users[idx]
test_item = test_items[idx]
```

不要使用 `idx` 作为 user ID。

图聚合时，参与用户需要先映射到连续矩阵行：

```text
users = list(round_user_params.keys())
for row, user in enumerate(users):
    matrix[row] = round_user_params[user][...]
```

聚合完成后，再将连续行号映射回原始 user ID。

## 最终研究叙事

可以将研究故事表述为：

UFGraphFR 通过用户文本描述构建隐私保护的用户关系图，从而缓解联邦推荐中用户孤立带来的协同信号缺失问题。但在 MCP 推荐场景中，用户侧和物品侧都包含丰富自然语言属性，同时数据缺少可靠时间戳。基于这一特点，我们将 UFGraphFR 扩展为面向 MCP 的双侧语义增强图联邦推荐框架：使用 LLM 增强用户语义表示，引入 item 文本属性矩阵增强物品表示，并将原 Temporal Transformer 调整为 Interaction-Context Transformer，以建模无时间戳场景下的交互上下文关系。

整体创新可以概括为：

```text
从用户侧语义图联邦推荐
扩展到
面向 MCP 的用户-物品双侧语义图联邦推荐
```

最稳妥的开发原则是：

```text
先修正 MCP 数据协议和 user ID 对齐，再逐步加入 LLM、item 语义和 Transformer 变体。
```

