# UFG 当前状态与论文实验总参考

本文档是后续工作的主参考文档。当前目标不是继续堆新模块，而是把 `full-experiment2` 分支稳定成一个可写论文、可做消融、可解释机制的完整框架。

本文档依据：

1. `D:\files\论文\UFG论文\修改后的完整框架.md`
2. 当前仓库 `D:\EEE\vscode\UFGraphFR\UFGraphFR`
3. 当前分支 `full3`
4. 参考思想：UFGraphFR 的 user-user graph 聚合机制，以及 FeDecider 的 direction/scale 解耦思想

---

## 1. 目前理想的完整框架

### 1.1 Client 侧

#### User 分支

MCP user 原始信息来自：

```text
data/mcp_prepare/user_side/raw/filtered_clients.json
```

理想流程：

```text
filtered_clients.json (98列富属性)
  -> frozen LLM / sentence encoder
  -> v_u [1, 384]
  -> Linear(384, 32)
  -> e_u = v_u @ W_u + b_u
```

含义：

- `v_u` 是冻结 LLM / sentence encoder 得到的静态 user 语义向量。
- `W_u` 是客户端本地可训练的 user projection。
- Server 端不拿原始 user 文本或交互明细，而是使用上传的 `embedding_user.weight` 构建 user-user graph。
- 这继承 UFGraphFR 的核心：用 user 表示构建用户关系图，再在图上聚合参数。

当前 MCP 论文主设定应固定为：

```text
pre_model = MiniLM-L6 或 HF_LOCAL
embed_dim = 384
latent_dim = 32
use_jointembedding = true
```

#### Item 分支

ID 分支：

```text
iid -> embedding_item -> e_id
embedding_item.weight: [5837, 32]
```

文本属性分支：

```text
iid -> item_text_features[iid]
item_text_features: [5837, 384]
item_text_proj: Linear(384, 32)
e_attr: [5837, 32]
```

当前可用 item text feature：

```text
data/mcp_prepare/item_side/features/item_text_features_A.npy
data/mcp_prepare/item_side/features/item_text_features_AB.npy
```

#### Fusion

理想融合：

```text
alpha = sigmoid(item_alpha[iid])
e_item = alpha * e_id + (1 - alpha) * e_attr
```

其中：

```text
item_alpha: [5837, 1]
init = -2.0
sigmoid(-2.0) ~= 0.12
```

设计含义：

- 冷 item 初始偏向 text branch。
- 热 item 可以通过本地梯度逐渐拉向 ID branch。
- `item_alpha` 是 per-item 参数，不是全局标量。

---

### 1.2 Client -> Server 上传

每轮参与客户端上传：

| 参数 | Shape | 粒度 | 作用 |
| --- | --- | --- | --- |
| `embedding_item.weight` | `[5837, 32]` | per-item | item ID 协同信号 |
| `item_alpha` | `[5837, 1]` | per-item | ID/text 融合权重 |
| `item_text_proj.weight` | `[32, 384]` | per-model | 文本语义投影矩阵 |
| `item_text_proj.bias` | `[32]` | per-model | 文本语义投影偏置 |
| `embedding_user.weight` | `[32, 384]` | per-client | Server 建 user-user graph |

`embedding_user.bias [32]` 当前存在于模型中，但理想框架的 graph 构建核心使用 `embedding_user.weight`。论文中建议只把 `W_u` 作为建图参数叙述，bias 可作为本地 user projection 的普通模型参数处理，不作为创新点。

---

### 1.3 Server 聚合

#### Step 1: 用 W_u 构建 user-user graph

```text
embedding_user.weight from clients
  -> user-user graph
```

这是 UFGraphFR 的核心继承点：Server 通过上传模型参数近似用户关系，而不是收集原始交互或原始文本。

#### Step 2: 同一张图聚合 per-item 参数

同一张 user-user graph 聚合：

```text
embedding_item.weight [5837, 32] -> graph aggregation
item_alpha            [5837, 1]  -> graph aggregation
```

设计含义：

- `embedding_item` 是 per-item 协同参数，天然适合 user-user graph 聚合。
- `item_alpha` 也是 per-item 参数，表达“某个 item 对 ID/text 的依赖程度”，相似用户可以共享这种融合偏好。
- 这比全局 scalar alpha 更细，也比纯 FedAvg alpha 更贴近 UFGraphFR 的图聚合逻辑。

#### Step 3: item_text_proj 做方向校准后 FedAvg

```text
item_text_proj.weight [32, 384]
  -> per-row L2 normalization
  -> FedAvg

item_text_proj.bias [32]
  -> FedAvg
```

设计含义：

- `item_text_proj` 是 per-model 全局语义投影，应保持全局一致。
- 不让每个客户端拥有完全漂移的 text projection space。
- per-row L2 来自 FeDecider 的 direction/scale 解耦启发：保留方向信息，削弱不同客户端数据规模和异质性带来的尺度噪声。
- 本工作不是照搬 FeDecider 的跨域推荐框架，而是把方向校准嵌入 UFGraphFR 的 MCP 联邦推荐场景。

---

### 1.4 Server -> Client 下发

理想下发：

| 参数 | 下发方式 |
| --- | --- |
| `embedding_item.weight` | 下发该 user 对应的图聚合结果 |
| `item_alpha` | 下发该 user 对应的图聚合结果 |
| `item_text_proj.weight` | 下发全局 L2+FedAvg 结果 |
| `item_text_proj.bias` | 下发全局 FedAvg 结果 |
| 其他本地 MLP / user_mlp 参数 | 保持本地个性化训练逻辑 |

论文描述中应强调：

```text
per-item 参数走 graph aggregation
per-model 语义投影走 L2 + FedAvg
W_u 只用于建图
```

---

## 2. 当前 full3 代码检查结果与问题

### 2.1 当前代码已经符合的部分

当前代码核心通路已经实现理想框架的主要结构。

已确认 shape：

```text
embedding_item.weight (5837, 32)
embedding_user.weight (32, 384)
embedding_user.bias   (32,)
item_text_features    (5837, 384)
item_text_proj.weight (32, 384)
item_text_proj.bias   (32,)
item_alpha            (5837, 1)
get_item_embedding output for 3 items: (3, 32)
```

数据侧已确认：

```text
filtered_clients.json: 187 rows, 98 columns
mapped users: 186
train.csv: 27027 interactions
vali.csv: 167 interactions
test.csv: 175 interactions
uid range: 0..185
iid range: 0..5836
```

核心代码对应：

- `mymodel.py`
  - `embedding_item = Embedding(num_items, latent_dim)`
  - `item_text_features` 从 `.npy` 加载
  - `item_text_proj = Linear(384, 32)`
  - `item_alpha = Parameter([num_items, 1], init=-2.0)`
  - `get_item_embedding()` 使用 per-item alpha 融合
- `engine.py`
  - 上传 `embedding_item.weight`
  - 上传 `item_text_proj.weight`
  - 上传 `item_text_proj.bias`
  - 上传 `item_alpha`
  - 上传 `embedding_user.weight`
  - `embedding_item.weight` 走 `MP_on_graph`
  - `item_alpha` 走 `MP_on_graph_param(..., (num_items, 1))`
  - `item_text_proj.weight` 逐行 L2 后 FedAvg
  - `item_text_proj.bias` FedAvg

结论：

```text
full-experiment2 的核心训练参数通路基本符合当前理想框架。
```

---

### 2.2 full3 已补齐的工程问题

#### 已补 1: cosine graph 语义已修正

当前 `utils.py` 已将 cosine distance 转为 cosine similarity：

```text
similarity = 1 - pairwise_distances(..., metric='cosine')
```

同时 `select_topk_neighboehood()` 在 `neighborhood_size > 0` 时会排除 self-loop，再选择 similarity 最大的邻居。因此正式论文中可以按“相似用户之间进行图聚合”来叙述。

#### 已补 2: evaluation 已支持 server-synced 参数

当前 `engine.py` 已新增：

```text
fed_evaluate(..., server_synced=True)
_sync_server_aggregated_params_for_eval(...)
```

评估时会在 client local cache 基础上覆盖 Server 聚合后的：

```text
embedding_item.weight[user]
item_alpha[user]
item_text_proj.weight
item_text_proj.bias
```

因此正式结果代表 Server 聚合并下发后的模型表现。

#### 已补 3: MCP user merge 前已去重

当前 `train.py::load_mcp_split()` 已在 merge 前对 `clients` 按 `id` 去重：

```text
clients = clients.drop_duplicates(subset=['id'], keep='first')
```

这解决了 `filtered_clients.json` 187 行但 mapping 只有 186 个 user 时的重复 merge 风险。

#### 已补 4: full2/full3 专用功能测试已新增

当前新增：

```text
test_full2_framework.py
```

覆盖：

```text
MCP user 去重
model shape
fusion formula
item_alpha graph aggregation
item_text_proj L2 + FedAvg
server-synced eval overlay
cosine graph similar-neighbor selection
```

#### 已补 5: 固定实验入口已新增

当前新增：

```text
scripts/run_full2_mcp.ps1
configs/full2_mcp.json
```

用于固定 MCP full2/full3 主实验参数，避免误用 `train.py` 原始默认值。

---

### 2.3 当前仍需注意的问题

#### 问题 1: 旧文档仍然保留旧框架

以下文档不再代表当前 full2 最终思路：

```text
FINAL_APPROACH.md
MCP_UFGraphFR_NEXT_REQUIREMENTS.md
UFGraphFR_MCP_LLM_HANDOFF.md
```

典型冲突：

- 旧文档中 `item_alpha` 仍被描述为全局 scalar 或 FedAvg。
- 旧文档中 `item_text_proj` 被描述为 graph aggregation 或 graph+directional aggregation。
- 当前新框架已经改为：

```text
item_alpha: per-item [5837,1] + graph aggregation
item_text_proj.weight: per-row L2 + FedAvg
item_text_proj.bias: FedAvg
```

处理建议：

```text
以后以本 UFG_CURRENT_STATUS.md 和 FULL_EXPERIMENT2_FRAMEWORK.md 为准。
旧文档只作为历史分支记录，不再作为实现标准。
```

#### 问题 2: 旧 D1/D2/D3 functional tests 已经过期

当前测试情况：

```text
test_d1_functional.py: 失败
原因: 仍断言 item_alpha 是 scalar，但当前正确 shape 是 [5837,1]

test_d2_functional.py: 通过，但测试内容过期
原因: 仍测试 item_alpha stays FedAvg，与当前 graph aggregation 不一致

test_d3_functional.py: 通过，但测试内容只覆盖旧的 graph-calibrated projection 思路
原因: 当前 full2 是 item_text_proj.weight L2 + FedAvg，不是 graph aggregation
```

已处理：

```text
新增 test_full2_framework.py 作为当前 full3 的通过标准。
```

该测试已验证：

1. `item_alpha` shape 是 `[5837,1]`
2. `get_item_embedding()` fusion 公式正确
3. 上传参数包含五类：
   - `embedding_item.weight`
   - `item_alpha`
   - `item_text_proj.weight`
   - `item_text_proj.bias`
   - `embedding_user.weight`
4. Server 聚合后：
   - `embedding_item.weight[user]`: `[5837,32]`
   - `item_alpha[user]`: `[5837,1]`
   - `item_text_proj.weight`: `[32,384]`
   - `item_text_proj.bias`: `[32]`
5. 下发逻辑：
   - `embedding_item.weight` 按 user 下发
   - `item_alpha` 按 user 下发
   - `item_text_proj.*` 全局 tensor 下发

#### 问题 3: train.py 默认参数仍不等于论文主设置

`train.py` 当前默认值仍是原项目默认：

```text
dataset = 100k
pre_model = USE
embed_dim = 100
use_item_attribute = false
use_transfermer = true
dp = 0.1
neighborhood_size = 0
lr_eta = 80
```

这些默认值会导致不小心运行时偏离当前论文框架。

论文主实验建议显式使用：

```text
--dataset mcp
--use_jointembedding true
--use_item_attribute true
--pre_model HF_LOCAL 或 MiniLM-L6
--hf_embedding_model D:\models\all-MiniLM-L6-v2
--embed_dim 384
--use_transformer false
--dp 0.0
--lr 0.01
--lr_eta 1
--neighborhood_size 5
--mp_layers 1
--reg 1.0 或待调参
```

已处理：

```text
新增 scripts/run_full2_mcp.ps1
新增 configs/full2_mcp.json
```

#### 问题 4: MCP user JSON 仍是 187 行，但训练 merge 输出应为 186 个 user

```text
filtered_clients.json: 187 rows
filtered_client_id_mapping.csv: 186 rows
```

当前处理：

```text
load_mcp_split() merge 前已 drop_duplicates('id')
test_full2_framework.py 会检查 merge 后 user_infos 为 186 行且 uid 唯一。
```

#### 问题 5: `embedding_user.bias` 的角色需要在论文中明确

当前模型存在：

```text
embedding_user.weight [32,384]
embedding_user.bias   [32]
```

当前上传建图使用的是 `embedding_user.weight`，未使用 bias。

这可以接受，但论文中要写清楚：

```text
W_u 用于 graph construction
b_u 是本地 projection bias，不作为 graph construction signal
```

---

## 3. 接下来要做的完整论文实验

实验目标不是只证明“加了 LLM/text 有提升”，而是证明：

```text
引入双分支 item 表示后，会出现 projection/fusion 的联邦一致性问题；
full2 用 per-item graph alpha + L2 FedAvg projection 解决该问题；
UFGraphFR 的 W_u graph 不只可聚合 item embedding，也可聚合 per-item fusion behavior。
```

---

### 3.1 代码修正与验证实验

正式跑论文实验前，以下工程项已在 full3 补齐：

#### Fix A: 更新测试

已新增：

```text
test_full2_framework.py
```

覆盖：

```text
shape check
fusion formula check
upload keys check
aggregation output check
download state_dict check
server-synced evaluation check
```

#### Fix B: 修正 cosine graph 语义

已修正为 similarity 语义：

```text
if metric == cosine:
  similarity = 1 - pairwise_distances(...)
  top-k 选最大 similarity
```

并已在 `test_full2_framework.py` 中补测试：

```text
两个方向相近的 W_u 应该互为 top neighbor
两个方向相反/较远的 W_u 不应被选为 top neighbor
```

#### Fix C: 评估时同步 server 聚合参数

已新增：

```text
fed_evaluate(server_synced=True)
```

确保正式结果代表 Server 聚合后下发的模型表现。

#### Fix D: 固定实验配置入口

已新增：

```text
scripts/run_full2_mcp.ps1
```

或：

```text
configs/full2_mcp.json
```

固定主实验参数，避免默认参数误导。

---

### 3.2 主效果实验

主指标：

```text
HR@10
NDCG@10
```

主数据集：

```text
MCP
186 users
5837 items
27027 train interactions
167 validation interactions
175 test interactions
```

主表建议：

| 编号 | 模型 | 目的 |
| --- | --- | --- |
| M0 | Original UFGraphFR-style on MCP, ID only | 原始 UFGraphFR 迁移到 MCP 的 baseline |
| M1 | User LLM graph + ID item only | 验证 rich user LLM / W_u graph 的价值 |
| M2 | M1 + item text branch, local proj only | 验证 item text branch 基础价值 |
| M3 | M2 + item_text_proj FedAvg | 验证 projection 进入联邦通信的价值 |
| M4 | M3 + item_text_proj per-row L2 + FedAvg | 验证 FeDecider 方向/尺度解耦思想 |
| M5 | M4 + scalar alpha | 对比全局融合权重 |
| M6 | M4 + per-item alpha FedAvg | 对比 per-item 但不用 graph |
| M7 | Full2: M4 + per-item alpha graph | 最终模型 |

预期叙事：

```text
M2 > M1: item text 有用
M3 > M2: projection federation 有用
M4 > M3: L2 direction calibration 有用
M7 > M6/M5: per-item alpha graph 有用
```

---

### 3.3 消融实验

#### Ablation A: item text branch

```text
ID only
ID + text
```

目的：

证明双分支 item representation 带来增益。

#### Ablation B: item_text_proj 聚合

```text
local only
FedAvg
L2 + FedAvg
graph aggregation
L2 + graph aggregation
```

注意：

当前 full2 主张是：

```text
item_text_proj = L2 + FedAvg
```

但为了论文完整，可以把 graph aggregation 作为对照，说明为什么最终选择 per-model global projector，而不是 per-user graph projector。

#### Ablation C: item_alpha 形式

```text
fixed alpha = 0
fixed alpha = 0.5
fixed alpha = 1
learnable scalar alpha
learnable per-item alpha + FedAvg
learnable per-item alpha + graph aggregation
```

目的：

证明 per-item alpha 和 graph aggregation 都有必要。

#### Ablation D: user graph source

```text
random graph
FedAvg no graph
item embedding graph
static v_u graph
trained W_u graph
```

目的：

对齐 UFGraphFR：证明 trained `W_u` graph 是核心。

#### Ablation E: text attribute set

```text
A
AB
```

目的：

证明 item text feature 构造方式对结果的影响。

#### Ablation F: text encoder

```text
USE
MiniLM-L6
HF_LOCAL e5-small 或其他本地 encoder
```

目的：

证明框架不依赖单一 encoder，并选择主实验 encoder。

---

### 3.4 冷启动与长尾实验

这是 full2 论文最重要的补充实验之一。

按 item 训练交互次数分组：

```text
cold item: 低交互
warm item: 中交互
hot item: 高交互
```

每组报告：

```text
HR@10
NDCG@10
```

核心预期：

```text
cold item: text branch 更重要，alpha 更小
hot item: ID branch 更重要，alpha 更大
```

同时分析：

```text
mean sigmoid(item_alpha) by popularity bucket
```

论文图建议：

```text
x轴: item popularity bucket
y轴: mean sigmoid(item_alpha)
```

如果趋势成立，这是 per-item alpha 的强解释性证据。

---

### 3.5 FeDecider 方向校准分析

为证明不是简单 FedAvg，应分析：

```text
item_text_proj.weight row norm before L2
item_text_proj.weight row norm after L2
client-wise projection cosine similarity
projection drift across rounds
training curve stability
```

建议图表：

1. L2 前后 row norm 分布图
2. 不同客户端 projection direction 相似度热力图
3. FedAvg vs L2+FedAvg 的 HR/NDCG 收敛曲线
4. 不同 round 的 projection drift 曲线

---

### 3.6 参数敏感性实验

#### Graph 参数

```text
neighborhood_size: 1, 3, 5, 10, 20
mp_layers: 1, 2, 3
similarity_metric: cosine, euclidean
```

注意：

cosine graph 修正前后的结果不能混在同一张主表里。

#### 联邦采样参数

```text
clients_sample_ratio: 0.05, 0.1, 0.2, 0.5, 1.0
local_epoch: 1, 2, 3
```

#### 优化参数

```text
lr: 0.001, 0.005, 0.01, 0.05
lr_eta: 1, 5, 10
reg: 0, 0.1, 1.0
batch_size: 64, 128, 256
```

---

### 3.7 通信与复杂度分析

每个参与客户端新增上传量：

```text
embedding_item.weight: 5837 * 32
item_alpha: 5837 * 1
item_text_proj.weight: 32 * 384
item_text_proj.bias: 32
embedding_user.weight: 32 * 384
```

论文中建议报告：

```text
新增参数量
相对 ID-only UFGraphFR 的通信开销比例
每轮训练时间
server aggregation 时间
```

尤其要解释：

```text
item_alpha 虽然是 per-item，但只有 [5837,1]，相对 embedding_item [5837,32] 开销较小。
item_text_proj 是 per-model 小矩阵 [32,384]，通信可控。
```

---

### 3.8 最终论文表格建议

#### Table 1: Main Performance

```text
HR@10 / NDCG@10
M0-M7 主模型对比
```

#### Table 2: Projection Aggregation Ablation

```text
local
FedAvg
L2+FedAvg
graph
L2+graph
```

#### Table 3: Alpha Ablation

```text
fixed alpha
scalar learnable
per-item FedAvg
per-item graph
```

#### Table 4: Cold/Warm/Hot Item Performance

```text
cold item HR/NDCG
warm item HR/NDCG
hot item HR/NDCG
```

#### Figure 1: Framework Overview

展示：

```text
User LLM -> W_u -> graph
Item ID/text dual branch
item_alpha graph aggregation
item_text_proj L2+FedAvg
```

#### Figure 2: Alpha vs Item Popularity

展示 per-item alpha 的解释性。

#### Figure 3: Projection Norm / Direction Calibration

展示 FeDecider 启发的 L2 校准有效性。

#### Figure 4: Training Curves

展示主要模型收敛速度和稳定性。

---

## 4. 推荐的下一步执行顺序

### Step 1: 先确认 full3 局部验证通过

1. 运行 `test_full2_framework.py`。
2. 运行 `test_server_synced_eval_functional.py`。
3. 可选运行旧 `test_d3_functional.py`，但旧 D1/D2 不再作为 full3 通过标准。

### Step 2: 小规模 sanity

使用：

```text
scripts/run_full2_mcp.ps1 -NumRound 3
```

确认训练、验证、测试全链路能跑通。

### Step 3: 正式主实验

使用：

```text
scripts/run_full2_mcp.ps1 -NumRound 50
```

### Step 4: 跑消融和解释性实验

优先级：

1. alpha 消融
2. projection 聚合消融
3. cold/warm/hot item
4. graph source 消融
5. 参数敏感性

---

## 5. 当前结论

当前 `full-experiment2` 已经完成核心框架：

```text
ID + text 双分支 item 表示
per-item alpha
item_alpha graph aggregation
item_text_proj L2 + FedAvg
W_u user-user graph
```

工程侧 full3 已补齐：

```text
测试更新
cosine graph 语义修正
server-synced evaluation
MCP user 去重
固定实验入口
```

论文实验侧还必须补齐：

```text
完整主实验
alpha / projection / graph / cold-start 消融
FeDecider 风格 direction calibration 分析
通信复杂度分析
```

从论文叙事上，最终主线建议固定为：

```text
Projection-consistent dual-branch federated recommendation
via W_u-guided per-item fusion aggregation
and direction-calibrated global text projection.
```
