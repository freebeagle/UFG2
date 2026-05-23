# UFGraphFR MCP / LLM / Item 文本 / Transformer 交接文档

本文档用于把本轮对话中的工作、当前进度、后续方向和注意事项完整交接给后续对话使用。

## 最高优先级规则

不要随意删除、移动、覆盖或清理下面路径中的任何内容：

```text
D:\EEE\vscode\UFGraphFR
```

除非用户明确同意某一个具体操作。

尤其不要移动或删除 MCP 数据集文件。若需要把数据复制到其他目录，必须先询问用户，并且使用“复制”，不要使用“移动”。

原始 MCP 数据保留在：

```text
D:\EEE\vscode\IFedRec\data\mcp
```

## 仓库与分支现状

主仓库路径：

```text
D:\EEE\vscode\UFGraphFR\UFGraphFR
```

重要分支：

```text
main
exp-llm-dataset
mcp-data-protocol
codex/mcp-interaction-context-transformer
```

各分支含义：

```text
main:
  原始 UFGraphFR 基线。

exp-llm-dataset:
  user 侧 PLM/LLM/HF embedding 替换分支。
  目标是把原论文的 user 侧 PLM 编码器替换或扩展为本地/HF sentence embedding 模型。

mcp-data-protocol:
  注意：这个分支名字像 MCP 数据接口分支，但当前实际只包含 MCP_UFGraphFR_idea.md。
  它目前没有真正实现 MCP 数据接口代码。

codex/mcp-interaction-context-transformer:
  当前真正包含 MCP 数据接口实现、MCP item 侧准备工作，以及 Transformer 改造计划的分支。
```

## 关键概念澄清

原始 UFGraphFR 论文中的 PLM 用在 user 侧。

原论文/原代码中提到的：

```text
USE
MiniLM
T5
TinyBERT
LaBSE
```

这些模型是用于编码用户文本描述，不是用于编码 item 文本。

原始 UFGraphFR 的 user 侧流程是：

```text
用户结构化属性
-> prompt / 用户文本描述
-> 冻结 PLM 编码器
-> 静态用户语义向量 v_u
-> 可训练 joint embedding 层
-> 本地交互适配后的用户侧权重 W_u
-> 服务端使用 W_u 构建 user-user graph
```

原始 UFGraphFR 的 item 侧主要是普通可训练 item ID embedding：

```text
embedding_item = Embedding(num_items, latent_dim)
```

所以需要明确：

```text
原论文中的 MiniLM = user 侧文本编码器。
MCP item/server 文本使用 MiniLM = 我们在 MCP 场景下新增的 item 侧扩展。
```

## 已完成工作

### 1. exp-llm-dataset 分支已修复并通过 smoke test

分支：

```text
exp-llm-dataset
```

用于 smoke test 的临时 worktree：

```text
C:\Users\Administrator\Documents\New project\ufg_exp_llm_smoke
```

已提交：

```text
6d5e454 fix: make llm embedding smoke test runnable
```

提交修改文件：

```text
embedding.py
engine.py
train.py
```

具体修复：

```text
1. embedding.py
   删除顶层 mediapipe 和 sentence_transformers 强制导入。
   mediapipe 只在 pre_model == "USE" 时按需导入。
   sentence_transformers 只在 MiniLM-L6 / HF_LOCAL 路径中按需导入。

2. engine.py
   tensorboardX SummaryWriter 改为可选导入。
   避免 tensorboardX / protobuf 缺失时阻塞训练。

3. train.py
   写入 sh_result/ 和 res/ 前自动创建目录。
   避免训练完成后因为目录不存在而失败。
```

意义：

```text
HF_LOCAL / LLM smoke test 不再依赖 mediapipe。
user 侧 LLM/HF embedding 路径可以独立于 USE 跑通。
```

### 2. LLM/HF 在论文自带 100k 数据集上 smoke test 通过

成功命令：

```powershell
cd "C:\Users\Administrator\Documents\New project\ufg_exp_llm_smoke"
$env:PYTHONPATH="C:\Users\Administrator\Documents\New project\ufg_exp_llm_smoke\.pydeps_llm"

python train.py `
  --dataset 100k `
  --num_round 1 `
  --clients_sample_ratio 0.005 `
  --pre_model HF_LOCAL `
  --hf_embedding_model "D:\models\e5-small-v2" `
  --use_jointembedding true `
  --use_transformer false `
  --reg 0.0 `
  --neighborhood_size 1 `
  --batch_size 128
```

关键输出：

```text
[embed_dim auto-override] 100 -> 384 for pre_model=HF_LOCAL

Testing Epoch 0:
HR = 0.0986
NDCG = 0.0438

Validation Epoch 0:
HR = 0.1135
NDCG = 0.0493
```

结论：

```text
HF_LOCAL 模型可以加载。
user embedding 可以生成。
embed_dim 可以自动识别。
UFGraphFR 可以使用 HF_LOCAL user embedding 训练一轮。
```

### 3. smoke test 的本地环境处理

系统 Python 的全局包存在版本不兼容：

```text
numpy 1.21.5
pandas 2.3.3
sentence-transformers 5.1.2
transformers 4.57.6
```

问题：

```text
pandas / transformers / sentence-transformers 会因为 numpy 太旧而失败。
```

临时解决方案是在 smoke worktree 中使用本地依赖目录：

```text
C:\Users\Administrator\Documents\New project\ufg_exp_llm_smoke\.pydeps_llm
```

其中安装了：

```text
numpy==1.26.4
tensorboardX==2.6.2.2 --no-deps
```

运行 smoke test 前需要：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Documents\New project\ufg_exp_llm_smoke\.pydeps_llm"
```

不要把 `.pydeps_llm` 当成源码提交。

## PLM vs LLM 对比实验命令

建议在 smoke worktree 中运行：

```powershell
cd "C:\Users\Administrator\Documents\New project\ufg_exp_llm_smoke"
$env:PYTHONPATH="C:\Users\Administrator\Documents\New project\ufg_exp_llm_smoke\.pydeps_llm"
```

### PLM 基线：MiniLM-L6

```powershell
python train.py `
  --dataset 100k `
  --num_round 1 `
  --clients_sample_ratio 0.005 `
  --pre_model MiniLM-L6 `
  --use_jointembedding true `
  --use_transformer false `
  --reg 0.0 `
  --neighborhood_size 1 `
  --batch_size 128
```

### LLM/HF 本地模型：e5-small-v2

```powershell
python train.py `
  --dataset 100k `
  --num_round 1 `
  --clients_sample_ratio 0.005 `
  --pre_model HF_LOCAL `
  --hf_embedding_model "D:\models\e5-small-v2" `
  --use_jointembedding true `
  --use_transformer false `
  --reg 0.0 `
  --neighborhood_size 1 `
  --batch_size 128
```

### LLM/HF 本地模型：all-MiniLM-L6-v2

用户计划下载：

```text
D:\models\all-MiniLM-L6-v2
```

下载后可运行：

```powershell
python train.py `
  --dataset 100k `
  --num_round 1 `
  --clients_sample_ratio 0.005 `
  --pre_model HF_LOCAL `
  --hf_embedding_model "D:\models\all-MiniLM-L6-v2" `
  --use_jointembedding true `
  --use_transformer false `
  --reg 0.0 `
  --neighborhood_size 1 `
  --batch_size 128
```

## MCP 数据接口现状

目标 MCP 数据接口范围：

```text
读取 MCP train/vali/test.csv。
使用 uid/iid 作为内部 userId/itemId。
不依赖 timestamp。
修复真实 user id 对齐。
修复评估时 idx != userId 的问题。
为后续 item_features / 原始 json 留接口位置。
```

重要纠正：

```text
mcp-data-protocol 分支目前没有实现这些代码。
它目前只包含 MCP_UFGraphFR_idea.md。
```

实际实现位于：

```text
codex/mcp-interaction-context-transformer
```

该分支中的 MCP 数据接口包括：

```text
train.py:
  --mcp_data_dir
  load_mcp_split()
  dataset == "mcp" 加载路径

data.py:
  SampleGenerator 支持预切分 train/val/test
  MCP 不再需要 timestamp

engine.py:
  评估使用真实 user id：
  user = int(test_users[eval_idx].item())

utils.py:
  图构建时将真实 user id 映射到临时矩阵行
  聚合后再映射回真实 user id

prepare_mcp_user_split.py:
  构建 MCP 用户内部静态 split
```

## MCP 数据划分逻辑

因为 MCP 没有可靠 timestamp，不要伪造时间戳。

推荐划分方式：

```text
先合并原始/暂存 train + vali + test 为全量交互。
按 uid 分组。
使用固定随机种子。

如果用户交互数 >= 3：
    1 个 item -> test
    1 个 item -> validation
    剩余 item -> train

如果用户交互数 == 2：
    1 个 item -> test
    1 个 item -> train

如果用户交互数 == 1：
    1 个 item -> train
    该用户不参与 ranking evaluation
```

Transformer 分支中生成的 split 统计：

```text
train: 27027 rows, 186 users
vali:  167 rows, 167 users
test:  175 rows, 175 users
```

原始暂存 split 看起来更像按 item 范围切分：

```text
train item range: 0 - 4668
vali item range: 4669 - 5018
test item range: 5019 - 5836
```

所以它不适合作为最终推荐评估的用户内部 split。

## Transformer 分支现状

分支：

```text
codex/mcp-interaction-context-transformer
```

重要文档：

```text
MCP_UFGraphFR_idea.md
MCP_DATA_PROTOCOL_SUMMARY.md
MCP_TRANSFORMER_BRANCH_PLAN.md
```

该分支已经完成：

```text
1. MCP 数据接口实现。
2. MCP 用户内部静态 split。
3. MCP raw/item 预处理目录。
4. item attribute text CSV 生成。
5. prepare_mcp_item_features.py。
```

item 侧已准备文件：

```text
data/mcp_prepare/item_side/features/item_attribute_text_A.csv
data/mcp_prepare/item_side/features/item_attribute_text_AB.csv
data/mcp_prepare/item_side/features/item_feature_meta.json
```

尚未完成：

```text
1. item_text_features_A.npy / item_text_features_AB.npy 尚未生成。
2. item attribute branch 尚未接入 mymodel.py。
3. Interaction-Context Transformer 尚未真正实现。
4. user_context_items 尚未接入模型 forward/evaluation。
5. exp-llm-dataset 的 user LLM/HF encoder 尚未和 MCP 数据接口干净合并。
```

当前 Transformer 设计解释：

```text
原始 UFGraphFR:
  Temporal Transformer

MCP:
  Interaction-Context Transformer
```

目标 attention 设计：

```text
Q = candidate item
K = user train-history context items
V = user train-history context items
```

含义：

```text
候选 MCP server 查询用户历史交互过的 MCP servers，
学习功能关联、共现关系和语义上下文。
```

不要把 MCP 场景下的 Transformer 描述为时间依赖建模，除非数据中确实有可靠 timestamp。

## item 侧下一步

用户计划下载：

```text
D:\models\all-MiniLM-L6-v2
```

下载后生成：

```text
item_text_features_A.npy
item_text_features_AB.npy
```

建议命令：

```powershell
python prepare_mcp_item_features.py `
  --servers_json data\mcp_prepare\item_side\raw\filtered_servers.json `
  --id_mapping data\mcp_prepare\item_side\mapping\filtered_servers_id_mapping.csv `
  --output_dir data\mcp_prepare\item_side\features `
  --attribute_sets A AB `
  --model_name "D:\models\all-MiniLM-L6-v2"
```

预期检查：

```text
shape[0] == 5837
iid range == 0..5836
feature row 与 filtered_servers_id_mapping.csv 对齐
```

## 后续推荐开发顺序

不要一次性拼完整模型。

### Step A：稳定 user 侧 LLM 分支

分支：

```text
exp-llm-dataset
```

目标：

```text
PLM -> HF_LOCAL/LLM 替换能在原论文数据集上跑通。
```

当前状态：

```text
已使用 D:\models\e5-small-v2 在 100k 上通过 smoke test。
```

下一步：

```text
如果需要，增加轮数做 PLM vs LLM 对比。
```

### Step B：整理干净 MCP 数据接口分支

因为 `mcp-data-protocol` 当前只有文档，建议创建或修复一个干净分支，只包含：

```text
MCP train/vali/test 读取
uid/iid 协议
无 timestamp 依赖
真实 user id 评估修复
prepare_mcp_user_split.py
MCP_DATA_PROTOCOL_SUMMARY.md
```

不要包含：

```text
item text features
raw json parsing
Transformer 改动
实验 log
res/sh_result 输出
```

目标 smoke test：

```powershell
python train.py `
  --dataset mcp `
  --num_round 1 `
  --use_jointembedding false `
  --use_transformer false
```

### Step C：合并 MCP 数据接口 + user LLM

建议新分支：

```text
codex/mcp-user-llm-data
```

目标：

```text
MCP clients 作为 users
MCP uid/iid 数据协议
HF_LOCAL user encoder
```

测试：

```powershell
python train.py `
  --dataset mcp `
  --num_round 1 `
  --clients_sample_ratio 0.05 `
  --pre_model HF_LOCAL `
  --hf_embedding_model "D:\models\e5-small-v2" `
  --use_jointembedding true `
  --use_transformer false
```

### Step D：item 文本特征生成

建议分支：

```text
codex/mcp-item-text-features
```

目标：

```text
生成 item_text_features_A.npy / AB.npy。
此阶段先不接模型。
```

### Step E：item attribute branch

建议分支：

```text
codex/mcp-item-text-branch
```

目标：

```text
加载 item_text_features.npy。
增加 item_text_proj。
第一版简单融合：
e_item = e_id + alpha * e_attr
```

测试时：

```text
关闭 user LLM
关闭 Transformer
开启 item attribute
```

### Step F：Interaction-Context Transformer

分支：

```text
codex/mcp-interaction-context-transformer
```

目标：

```text
Q = candidate item
K/V = user train-history context
```

严格要求：

```text
user context 只能使用 train positives。
不能泄露 validation/test item。
```

## 测试原则

每个分支只证明一件事。

最低要求：

```text
每完成一个分支改动，就跑 num_round=1 smoke test。
不要在各部分独立跑通前合并 LLM + item text + Transformer。
```

推荐测试顺序：

```text
1. 数据接口：
   dataset=mcp, jointembedding=false, transformer=false

2. User LLM：
   先 dataset=100k，再 dataset=mcp

3. Item text feature 生成：
   检查 .npy shape 和 iid 对齐

4. Item branch：
   dataset=mcp, item_attribute=true, transformer=false

5. Transformer：
   dataset=mcp, context 只来自 train positives
```

## 当前注意事项

1. `mcp-data-protocol` 分支名字有误导性。
   它目前只有文档，没有实际 MCP 数据接口代码。

2. 实际 MCP 数据接口代码在 `codex/mcp-interaction-context-transformer`。

3. `exp-llm-dataset` 已经在临时 worktree 中完成并通过 user 侧 LLM/HF smoke test。

4. 全局 Python 包版本不一致。
   推荐使用本地 PYTHONPATH 临时方案，或之后新建干净 venv/conda 环境。

5. 不要提交依赖目录：

```text
.pydeps_llm/
.tmp_pydeps/
```

6. 不要提交运行日志/结果，除非用户明确要求：

```text
log/
res/
sh_result/
```

