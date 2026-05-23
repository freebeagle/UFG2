# UFGraphFR B0/B1/D0 对话交接记录

日期：2026-05-21  
项目路径：`D:\EEE\vscode\UFGraphFR`

## 核心实验链

本轮讨论基于消融文档中的归因链：

```text
B0: UFGraphFR + MCP data interface
B1: B0 + user-side LLM/HF_LOCAL encoder
D0: B1 + item text branch local projection
D1: D0 + item_text_proj FedAvg
D2: D1 + item_text_proj graph aggregation
D3: D2 + row-L2 directional calibration
```

重要约束：

- 单个分支只验证当前分支功能，不做全代码/全训练集成式证明。
- B0/B1/D0 尽量控制变量，20 轮实验参数保持一致，只改必要变量。
- 输出逻辑沿用 full3/main 风格：完整过程 log + `sh_result` 追加摘要 + `res` 追加 jsonl 完整曲线。

## B0 baseline

B0 被确认的基线 commit：

```text
3d825c3 feat: mcp-baseline — MCP data interface on top of original UFGraphFR
```

B0 worktree / 分支：

```text
D:\EEE\vscode\UFGraphFR\mcp-baseline
branch: codex/b0-20r-mcp-baseline
base/head: 3d825c3
```

B0 含义：

```text
UFGraphFR 原模型
+ MCP data interface
+ user side USE embedding
不加 item text branch
不加 D0-D3
不加 projection calibration
```

B0 运行时关键参数：

```text
--dataset mcp
--pre_model USE
--use_jointembedding True
--use_transfermer True
--num_round 20
```

## B1 baseline

B1 不直接使用旧的 `codex/mcp-user-llm-data-integration` 分支，因为它不是从 `3d825c3` 干净继承出来，而且夹带 item-side 数据/实验产物。

B1 worktree / 分支：

```text
D:\EEE\vscode\UFGraphFR\b1-user-hflocal
branch: codex/b1-user-hflocal-20r
base/head: 3d825c3
```

B1 含义：

```text
B0
+ user-side HF_LOCAL encoder
```

B1 相对 B0 只改：

```text
--pre_model HF_LOCAL
--hf_embedding_model D:\models\e5-small-v2
--embed_dim 384
--ps B1_3d825c3_HF_LOCAL_e5-small-v2_20r
```

用户实验结果：

```text
B1 比 B0 高约 2.4 个点
```

解释：

- user-side backbone upgrade 在 MCP baseline 上是有帮助的。
- B1 作为后续 D0-D3 的前置基线是合理的。

## D0 版本

一开始发现旧提交：

```text
58fd377 feat: add mcp item attribute branch
```

很像 D0，但不适合直接跑，因为它会倒退/丢失 B1 所需的 HF_LOCAL、rich MCP user prompt 和输出逻辑。最终决定：

```text
不直接用 58fd377 原样跑。
在 3d825c3 基线上新建理想 D0 commit。
保留 B0/B1 的 HF_LOCAL/user 侧逻辑和输出逻辑。
只加入 D0 必需的 item text local projection。
```

D0 worktree / 分支 / commit：

```text
D:\EEE\vscode\UFGraphFR\d0-item-local-proj
branch: codex/d0-item-local-proj-20r
commit: efe45c7 feat: D0 item text local projection baseline
base: 3d825c3
```

D0 代码改动：

- 加入 item text feature 文件：

```text
data/mcp_prepare/item_side/features/item_attribute_text_A.csv
data/mcp_prepare/item_side/features/item_attribute_text_AB.csv
data/mcp_prepare/item_side/features/item_feature_meta.json
data/mcp_prepare/item_side/features/item_text_features_A.npy
data/mcp_prepare/item_side/features/item_text_features_AB.npy
```

- `train.py` 增加 D0 参数：

```text
--use_item_attribute
--item_attribute_set
--mcp_item_feature_path
--item_attribute_alpha
```

- `mymodel.py` 增加本地 item text projection：

```text
item_text_features -> item_text_proj -> item_attribute_embedding
e_item = e_id + item_attribute_alpha * e_attr
```

- `engine.py` 让本地 item optimizer 同时更新：

```text
embedding_item
item_text_proj
```

D0 明确没有加入：

```text
item_text_proj FedAvg
server-side item_text_proj aggregation
graph-guided item_text aggregation
row-L2 directional calibration
```

验证过：

```text
item_text_features_A.npy shape = (5837, 384)
item_text_features_AB.npy shape = (5837, 384)
python -m py_compile train.py mymodel.py engine.py embedding.py 通过
```

## D0 实验结果与解释

用户实验结果：

```text
D0 比 B1 掉约 13.8 Val HR
```

最初解释：

- D0 增加了 item text branch，但没有校准和通信机制。
- `e_item = e_id + e_attr` 是粗暴相加，ID 空间和文本投影空间未对齐。

后续讨论澄清出更关键的问题：

```text
D0 forward/predict 侧:
e_item = e_id + e_attr

D0 user 侧:
embedding_user.weight 上传 + 构图，和原 UFG 一样

D0 server 聚合侧:
仍用 embedding_user.weight 构图，并聚合 item ID embedding

D0 item 上传侧:
只上传 e_id，没有上传 item_text_proj，也没有上传融合后的 e_item
```

因此 D0 的真正问题是：

```text
本地训练时 e_id 和 e_attr 绑在一起训练。
上传/聚合时 server 只看到 e_id。
聚合回来的 e_id 再和本地 item_text_proj 搭配。
两者空间容易对不上。
```

重要结论：

```text
D0 可以被理解为一个“故意不完整/naive”的消融。
它不是正确最终方法。
它测试的是：双分支 forward 但仍沿用单分支 item communication 会不会坏。
结果证明会坏。
```

论文表述建议：

```text
D0: naive local item-text branch
或
D0: uncommunicated item-text branch
```

不要把 D0 表述成完整合理方法，而是表述成诊断性消融：

```text
D0 错得有诊断价值。
它证明只改 forward、不改 item-side communication 会显著伤害。
```

## 后续 D1/D2/D3 思路

关键方向：

双分支后 item 侧上传对象不能再只是 `e_id`。需要把 item-side state 扩展为：

```text
方案 A:
上传 fused item embedding = e_id + e_attr
server 聚合 fused
client forward 时直接使用 fused，不再重复加 e_attr

方案 B:
分别上传/同步 e_id 与 item_text_proj / alpha
client forward 仍按 e_item = e_id + e_attr
```

当前计划中的 D1-D3 更接近方案 B：

```text
D1: item_text_proj / item_alpha 进入 federated communication，先 FedAvg
D2: item_text_proj 从 FedAvg 升级为 graph-guided aggregation
D3: row-L2 directional calibration，解决方向和尺度问题
```

D0 掉分为 D1-D3 提供动机：

```text
B1 -> D0:
证明裸加 item text branch 且不通信会伤害

D0 -> D1:
验证同步 text projection 是否缓解

D1 -> D2:
验证 graph aggregation 是否更适合 text projection

D2 -> D3:
验证 row-L2 calibration 是否修正方向/尺度
```

## 输出逻辑结论

B0/B1/D0 输出逻辑已经够用，和 full3 的核心记录方式一致：

```text
log/*.log:
完整 stdout/stderr 过程，由 Tee-Object 保存。

sh_result/item-mcp.txt:
每次实验结束追加一行 result_str，一眼看最终 HR/NDCG/best_round/参数。

res/*.jsonl:
追加完整结构化结果，包括 hit_list、ndcg_list、val_hr_list、val_ndcg_list、
train_loss_list、test_loss_list、val_loss_list、best_val_hr、final_test_round、result_str。
```

注意：

`res/*.jsonl` 文件名不一定包含 `pre_model`，因此命令里的 `--ps` 必须写清实验身份，例如：

```text
B1_3d825c3_HF_LOCAL_e5-small-v2_20r
D0_efe45c7_HF_LOCAL_e5-small-v2_item-text-local-proj_20r
```

## D0 运行命令

最后给出的 D0 命令：

```powershell
Set-Location 'D:\EEE\vscode\UFGraphFR\d0-item-local-proj'

New-Item -ItemType Directory -Force -Path .\log | Out-Null
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'

& 'D:\EEE\vscode\UFGraphFR\.venv\Scripts\python.exe' .\train.py `
  --alias UFGraphFR `
  --dataset mcp `
  --num_round 20 `
  --clients_sample_ratio 1.0 `
  --clients_sample_num 0 `
  --local_epoch 1 `
  --construct_graph_source item `
  --neighborhood_size 0 `
  --neighborhood_threshold 1.0 `
  --mp_layers 1 `
  --similarity_metric cosine `
  --reg 1.0 `
  --lr_eta 80 `
  --batch_size 256 `
  --optimizer sgd `
  --lr 0.1 `
  --latent_dim 32 `
  --num_negative 4 `
  --layers '64, 32, 16, 8' `
  --l2_regularization 0.0 `
  --dp 0.1 `
  --use_cuda False `
  --use_mps False `
  --use_transfermer True `
  --use_jointembedding True `
  --device_id 0 `
  --ind 0 `
  --update_round 1 `
  --embed_dim 384 `
  --pre_model HF_LOCAL `
  --hf_embedding_model 'D:\models\e5-small-v2' `
  --mcp_data_dir 'data/mcp_user_split' `
  --mcp_user_json 'data/mcp_prepare/user_side/raw/filtered_clients.json' `
  --use_item_attribute True `
  --item_attribute_set A `
  --mcp_item_feature_path 'data/mcp_prepare/item_side/features/item_text_features_A.npy' `
  --item_attribute_alpha 1.0 `
  --ps 'D0_efe45c7_HF_LOCAL_e5-small-v2_item-text-local-proj_20r' 2>&1 | Tee-Object -FilePath ".\log\d0_item_local_proj_e5_20r_$ts.log"
```

## 下个对话建议入口

如果继续推进，应从 D1 开始，先讨论/确认 item-side communication 设计：

```text
问题 1:
D1 是同步 item_text_proj / item_alpha，还是直接上传 fused e_item？

问题 2:
如果上传 fused e_item，forward 侧如何避免 e_attr 被重复加两次？

问题 3:
如果沿用当前 D1-D3 方案，D1 应该如何最小实现：
只 FedAvg item_text_proj + item_alpha，不引入 graph aggregation，不引入 calibration。
```

推荐下个任务：

```text
基于 D0 commit efe45c7 新建 D1 worktree/branch，
只加入 item_text_proj / item_alpha 的 FedAvg 通信，
保持 B1/D0 其他参数一致。
```
