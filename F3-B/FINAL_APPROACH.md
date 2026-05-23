# UFGraphFR-MCP 思路终极版

## 一句话

> 你不是「用了 LLM + graph + transformer 的推荐模型」，
> 你是「解决联邦双分支 item representation 中 projection inconsistency 的框架」。

论文应该让人记住的不是你用了什么模块，而是你发现了一个什么问题、怎么解决的。

---

## 问题定义：Projection Inconsistency in Federated Dual-Branch Item Representation

### 问题从哪来

原 UFGraphFR 的 item 侧只有一个 ID embedding `e_item = Embedding(iid)`。所以不存在"不同信号源需要对齐"的问题。

当我们引入 item text branch 后，item 表示变成双分支：

```
e_item = σ(α)·e_id + (1-σ(α))·e_attr
```

`e_id` 来自 `embedding_item`（联邦共享的协同信号），`e_attr` 来自 `item_text_proj(item_text_features[iid])`（文本语义信号）。

**问题**：如果 `item_text_proj` 只在本地训练不上传，每个客户端学到不同的投影空间。同一个 item 的 `e_attr` 在客户端 A 和 B 的语义坐标系不同。这导致：

1. `e_item = e_id + e_attr` 的加法融合在不同客户端表示不同的东西——`e_id` 被迫适配本地漂移后的 `e_attr`
2. 上传到 Server 的 `e_id` 携带本地投影偏差
3. Weaken-based 的图聚合前提被削弱

### 为什么这是新问题

- 原 UFGraphFR 没有 text branch，不存在这个问题
- GPFedRec 只有 ID embedding，不存在这个问题
- FeDecider/IFedRec 有 item attribute network，但它的 attribute network 在 Server 上做 meta-learning，不在客户端训练
- **我们的设置**：item text features 在客户端本地通过 `item_text_proj` 投影后与 `e_id` 融合，`item_text_proj` 在本地训练 → 这是联邦推荐中一个没有被明确讨论过的问题

---

## 解法：Graph-Guided Projection Calibration

### 借鉴来源

**FeDecider**（基于 LLM 的联合跨域推荐框架）的核心思想：

> 联邦学习中不同客户端因数据异质性，训练出的参数存在 direction（语义方向，应共享）和 scale（尺度，噪声）的混合。FeDecider 在聚合前将两者解耦——保留 direction 的共享价值，削弱 scale 的干扰。

我们将这个思想应用到 UFGraphFR 的图聚合管道中——不是照搬 FeDecider 框架，而是将其方向归一化嵌入 UFGraphFR 的 user-user graph 聚合之前的 preprocessing 阶段。

**与 FeDecider 的关键差异**：
- FeDecider 做的是跨域推荐，方向解耦用于 LLM prompt embedding
- 我们做的是单域联邦推荐，方向归一化用于 `item_text_proj` matrix，然后用 UFGraphFR 的 user-user graph（而非 FedAvg）聚合

### 整体思路

```
不是继续堆模块，而是用一个统一的机制解决一个明确的问题：

  projection inconsistency → graph-guided calibration → consistent dual-branch fusion
```

### 分三步

**Step 1（D1）- FedAvg baseline**：把 `item_text_proj` 纳入联邦通信，用 FedAvg 聚合。证明"只要联邦了"就比纯本地好。

**Step 2（D2）- 图聚合**：把 `item_text_proj` 的聚合从 FedAvg 升级为图聚合（与 `item_embedding` 共用同一张 user-user graph）。证明图聚合比 FedAvg 更好——因为图上相近的用户本来就应该有相似的投影偏好。

**Step 3（D3）- 方向校准**：在 D2 图聚合之前，对每个客户端上传的 `item_text_proj.weight` 按行做 L2 归一化。消除不同客户端因数据量/交互密度不同导致的 scale noise，让图聚合在方向空间中操作。证明校准后的图聚合比直接图聚合更好。

---

## 整体架构

```
                    Server
  ┌──────────────────────────────────────────────┐
  │                                              │
  │   W_u[1]  W_u[2]  ...  W_u[N]               │
  │     ↓       ↓            ↓                   │
  │   ┌──────────────────────────────┐           │
  │   │  user-user graph (基于W_u)   │           │
  │   └──────────────────────────────┘           │
  │     ↓          ↓              ↓              │
  │   ┌──────────────────────────────────────┐   │
  │   │ 同一张图聚合三条轨道:                │   │
  │   │  A: item_embedding           [图]   │   │
  │   │  B: item_text_proj           [图+校准] │   │
  │   │  C: item_alpha               [FedAvg] │   │
  │   └──────────────────────────────────────┘   │
  │                                              │
  └──────────────────────────────────────────────┘
                    ↕
  ┌──────────────────────────────────────────────┐
  │              Client i                         │
  │                                              │
  │  user: LLM → v_u → joint_embedding(W_u) → e_u│
  │        W_u → 上传建图                        │
  │                                              │
  │  item: iid → embedding_item → e_id           │
  │        iid → text_features → proj → e_attr   │
  │        e_item = σ(α)·e_id + (1-σ(α))·e_attr  │
  └──────────────────────────────────────────────┘
```

---

## 关键设计决策

### item_alpha: FedAvg

`item_alpha` 是全局标量，表达 ID vs text 的整体偏好平衡。图聚合的语义是「相似用户对同一 item 有相似的看法」，与全局标量不匹配。alpha 始终用 FedAvg。

### item_text_proj: FedAvg → 图聚合 → +方向校准

- D1: FedAvg（打管道，证明联邦有用）
- D2: 图聚合（共享 user-user graph，证明图比平均好）
- D3: 方向校准（图聚合前按行 L2 归一化，证明需要消 scale noise）

### item_embedding: 图聚合

UFGraphFR 原有逻辑，不变。

---

## 分支 — 到底做什么

| 分支 | 代码改动 | 聚合方式 | 证明什么 |
|------|---------|---------|---------|
| D0 | 双分支 + local proj + 固定 alpha | 不上传 | 双分支先跑通 |
| D1 | alpha→可学习 + upload→FedAvg→download | FedAvg | 联邦比本地好 |
| D2 | item_text_proj 从 FedAvg 升级到 MP_on_graph | 图 | 图聚合比 FedAvg 好 |
| D3 | 图聚合前 per-row L2 normalize | 图+校准 | 方向校准比直接图聚合好 |

### 每个分支的功能测试（不跑完整训练）

**D2 功能测试**：
- mock 2 个 client + user graph
- 检查 item_text_proj 经过 `MP_on_graph` 后 shape 不变 [32, 384]
- 检查 item_embedding 和 item_text_proj 走同一张图但各自聚合
- 检查 user id 到矩阵行再回映射不乱

**D3 功能测试**：
- mock 同方向不同范数的 projection → 归一化后方向一致
- mock 不同方向的 projection → 归一化后差异保留
- 无 NaN/inf

---

## 不做的事

```
✂️ Transformer (E): 砍掉。不放主论文。
   - 当前 novelty 是 projection consistency，不是 sequence modeling
   - 加 Transformer 审稿人分不清提升来自哪里

✂️ 不再加任何模块（cross-attention / contrastive loss / MoE / adapter）
```

---

## 论文叙事

### 题目方向

```
Projection-Consistent Federated Recommendation via Graph-Guided Dual-Branch Calibration
```

### 摘要逻辑链

```
1. 联邦推荐引入双侧语义（user text + item text）→ 自然趋势
2. 引入 item text branch 后 → 出现 projection-space inconsistency（新问题）
3. local proj 导致 e_attr 坐标系漂移 → 污染 e_id → 破坏图聚合
4. 提出 graph-guided projection calibration:
   a. item_text_proj 进联邦 → baseline (FedAvg)
   b. 用 user-user graph 聚合 projection → 利用相似用户偏好
   c. 聚合前 direction normalization → 消除 scale noise
5. 消融: 本地 < FedAvg < 图 < 图+校准
6. 不增加额外数据依赖，不增加客户端计算负担
```

### 贡献列表

```
1. 指出联邦双分支 item 表示中存在 projection inconsistency 问题
2. 提出 graph-guided projection calibration 框架
3. 证明 user-user graph 可同时用于聚合 embedding 和 projection
4. 在 MCP 数据集上消融验证每一步的有效性
```

---

## 消融实验（主实验）

```
01. UFGraphFR baseline on MCP（无 item text branch）
02. + item text branch, local proj only (D0)
03. + item text branch, FedAvg proj (D1)
04. + item text branch, graph-guided proj (D2)
05. + item text branch, graph-guided + directional calibration (D3)
06. Full model = 02(LLM user encoder) + 05

消融对比:
  D0 vs D1: 联邦 vs 本地 → 证明联邦有用 (= 证明 drift 存在)
  D1 vs D2: 图 vs FedAvg → 证明图聚合的合理性
  D2 vs D3: 校准 vs 无校准 → 证明 scale noise 需要处理

附加消融:
  w/o item text branch（ID only）
  FedAvg vs graph aggregation for item_text_proj
  static v_u graph vs trained W_u graph
  A vs AB item attribute text
  alpha: fixed 0/0.5/1 vs learnable
```

---

## 开发路线

```
✅ D0 (codex/mcp-item-branch)
✅ D1 (codex/mcp-item-proj-communication)  ← 当前分支
⬜ D2 (codex/mcp-graph-guided-dual-aggregation)
⬜ D3 (codex/mcp-directional-proj-calibration)
⬜ 主实验
✂️ E: 砍掉
```

---

## 与 UFGraphFR 的差异

| | UFGraphFR | 本工作 |
|------|------|------|
| user 侧 | PLM | LLM（更强语义） |
| item 侧 | ID embedding only | ID + text 双分支 |
| 存在问题 | 无（单分支不存在 inconsistency） | **projection inconsistency** |
| 解法 | graph-guided item embedding agg | graph-guided projection calibration |
| Transformer | Temporal Transformer | 实验阶段先不加 |

---

## 参考

- UFGraphFR (Wang et al., The Journal of Supercomputing, 2026)
- GPFedRec (Zhang et al., KDD 2024)
- FeDecider (基于 LLM 的联合跨域推荐框架): LLM-based federated cross-domain recommendation; we borrow its direction decoupling idea — separate direction (semantic, shared) from scale (noise, local) before federated aggregation
- IFedRec (Zhang et al., WWW 2024): item-aligned federated aggregation for cold-start recommendation
