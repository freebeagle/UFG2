# UFG2 实验目录说明

这个目录用于整理 **MCP 数据集接入 UFG / UFGraphFR 框架** 后的消融实验代码。

核心目标：

```text
先把 MCP 数据集跑进原 UFG，
再逐步验证 user 侧 LLM、item 文本分支、projection 聚合、full3 alpha 聚合到底哪一步有效或有害。
```

## 1. 实验路线

### 主线：B0 → B1 → D1 → D2 → D3

验证 FINAL_APPROACH 的 projection calibration 是否有效。每一步只改一个变量。

### 旁线：full3 → F3-A → F3-B / F3-C

full3 是一个混合候选方法（item_text_proj L2+FedAvg + item_alpha graph）。
F3 系列不回答"哪个方法更好"，只回答：**"full3 如果差，是不是 item_alpha 的通信方式造成的？"**

- F3-A = full3 的干净复制（旁线基线）
- F3-B = 怀疑 alpha graph 拖后腿 → 改成 FedAvg 验证
- F3-C = 怀疑 alpha 根本不该通信 → 改成 local only 验证

旁线结果不用于证明主线。旁线只服务于 full3 这个具体候选的诊断。

## 2. 每个目录代表什么

| 目录 | 含义 |
|---|---|
| `B0-mcp-baseline` | 原 UFG + MCP 数据接口 |
| `B1-user-hflocal` | B0 + 用户侧 HF_LOCAL / LLM encoder |
| `D1` | B1 + item 文本分支，`item_text_proj` 和 `item_alpha` 都 FedAvg |
| `D2` | D1 + `item_text_proj` 改成 graph aggregation，`item_alpha` 仍 FedAvg |
| `D3` | D2 + `item_text_proj.weight` 图聚合前 row-L2 |
| `full3` | 原 full3 参考版本（完整候选方法） |
| `F3-A` | full3 的干净复制（旁线基线）：`item_alpha` graph aggregation |
| `F3-B` | F3-A 改成 `item_alpha` FedAvg |
| `F3-C` | F3-A 改成 `item_alpha` local only |

## 3. 分支检查重点

### B0 / B1

不应该有 item 文本分支：

```text
无 item_text_proj
无 item_alpha
```

### D1

应该是最简单通信：

```text
item_text_proj.weight: FedAvg
item_text_proj.bias: FedAvg
item_alpha: FedAvg
```

### D2

只比 D1 多一步：

```text
item_text_proj.weight/bias: graph aggregation
item_alpha: FedAvg
```

不能改 alpha、optimizer、user 侧。

### D3

只比 D2 多一步：

```text
item_text_proj.weight 先 row-L2
再 graph aggregation
item_alpha 仍 FedAvg
```

### F3-A / F3-B / F3-C

这三个只改 alpha 通信方式，其他跟 full3 完全一致：

```text
F3-A: item_alpha graph aggregation
F3-B: item_alpha FedAvg
F3-C: item_alpha local only
```

## 4. 共同检查项

所有目录都应该满足：

```text
cosine distance 要转 similarity: 1.0 - adj
top-k 选邻居时要排除自己
能通过 py_compile
```

检查命令：

```powershell
$dirs = 'B0-mcp-baseline','B1-user-hflocal','D1','D2','D3','F3-A','F3-B','F3-C','full3'
foreach ($d in $dirs) {
  cd "D:\EEE\vscode\UFG2\$d"
  D:\EEE\vscode\UFGraphFR\.venv\Scripts\python.exe -m py_compile train.py engine.py utils.py mymodel.py
}
```

## 5. 

控制变量 必须保证这些参数一致：

```text
num_round
clients_sample_ratio
lr
lr_eta
batch_size
neighborhood_size
reg
pre_model
hf_embedding_model
```

D1/D2/D3 同一套参数，F3-A/F3-B/F3-C 同一套参数。两条线之间不跨比。
