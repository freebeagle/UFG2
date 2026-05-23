# MCP Data Protocol Integration Summary

This document summarizes the MCP dataset interface work on the `mcp-data-protocol` branch.

## Goal

The goal of this branch is to adapt UFGraphFR to the MCP dataset format while keeping the model logic unchanged as much as possible.

This branch focuses only on:

- MCP data loading.
- Static within-user split for a dataset without timestamps.
- Real `uid` alignment during training and evaluation.
- Compatibility with UFGraphFR's existing `1 positive + 99 negative` ranking evaluation protocol.

This branch does not implement:

- Item text embedding.
- Text-similarity-based item ordering.
- Interaction-Context Transformer changes.
- `item_features.npy` fusion into item representations.
- Raw MCP JSON parsing.

Those should be handled in a later model/semantic branch.

## Data Directories

The original IFedRec MCP data is preserved at:

```text
D:\EEE\vscode\IFedRec\data\mcp
```

The copied raw/pre-split MCP files inside UFGraphFR are staged at:

```text
D:\EEE\vscode\UFGraphFR\UFGraphFR\data\mcp_staging
```

Files in `data/mcp_staging`:

```text
train.csv
vali.csv
test.csv
filtered_client_id_mapping.csv
item_features.npy
```

The processed within-user static split used by default is:

```text
D:\EEE\vscode\UFGraphFR\UFGraphFR\data\mcp_user_split
```

Files in `data/mcp_user_split`:

```text
train.csv
vali.csv
test.csv
filtered_client_id_mapping.csv
item_features.npy
```

## Why The Original Split Was Not Used Directly

The staged MCP files had the following item ranges:

```text
train item range: 0 - 4668
vali item range: 4669 - 5018
test item range: 5019 - 5836
```

There was no item overlap across `train`, `vali`, and `test`. This indicates that the original files are closer to an item-range split, not a within-user recommendation split.

For UFGraphFR-style ranking evaluation, the preferred protocol is a per-user static split:

- Each user keeps local train interactions.
- Validation/test positives are held out per user.
- The model is evaluated with `1 positive + 99 negative` candidates.
- Metrics remain HR@10 and NDCG@10.

## Static Within-User Split Protocol

Because MCP has no reliable timestamp, no temporal leave-one-out split is used.

Instead, the split script merges the staged `train.csv`, `vali.csv`, and `test.csv` into full interaction data, then splits interactions within each user with a fixed random seed.

Current rule:

```text
if user interaction count >= 3:
    1 item -> test
    1 item -> validation
    remaining items -> train

elif user interaction count == 2:
    1 item -> test
    1 item -> train

elif user interaction count == 1:
    1 item -> train
    user does not participate in ranking evaluation
```

The split seed is:

```text
2026
```

The generated split statistics are:

```text
train: 27027 rows, 186 users
vali:  167 rows, 167 users
test:  175 rows, 175 users
```

## Split Script

The split script is:

```text
prepare_mcp_user_split.py
```

Default input:

```text
data/mcp_staging
```

Default output:

```text
data/mcp_user_split
```

Example:

```powershell
python prepare_mcp_user_split.py `
  --input_dir data/mcp_staging `
  --output_dir data/mcp_user_split `
  --seed 2026
```

The script also copies the following auxiliary files into the processed split directory when present:

```text
filtered_client_id_mapping.csv
item_features.npy
```

## Training Entry

The MCP dataset can now be selected with:

```powershell
python train.py --dataset mcp
```

The default MCP data directory is:

```text
data/mcp_user_split
```

It can be overridden with:

```powershell
python train.py --dataset mcp --mcp_data_dir path\to\mcp_split
```

## MCP File Format

The MCP interaction files use:

```text
uid,iid
```

During loading, they are converted to UFGraphFR's internal format:

```text
uid -> userId
iid -> itemId
rating = 1.0
```

No `timestamp` column is required for MCP.

## Code Changes

### `train.py`

Added:

- `--mcp_data_dir`, defaulting to `data/mcp_user_split`.
- `load_mcp_split(data_dir)`.
- MCP-specific loading path for `--dataset mcp`.

For MCP, the code does not read:

```text
data/mcp/ratings.dat
```

Instead, it reads:

```text
{mcp_data_dir}/train.csv
{mcp_data_dir}/vali.csv
{mcp_data_dir}/test.csv
{mcp_data_dir}/filtered_client_id_mapping.csv
```

For MCP, `num_users` and `num_items` are inferred as:

```python
num_users = max(userId) + 1
num_items = max(itemId) + 1
```

### `data.py`

`SampleGenerator` now supports pre-split data:

```python
SampleGenerator(
    ratings=rating,
    train_ratings=train_ratings,
    val_ratings=val_ratings,
    test_ratings=test_ratings,
    num_users=config["num_users"],
    num_items=config["num_items"],
)
```

For MCP, it does not call timestamp-based leave-one-out splitting.

Training data is stored by real `userId` position:

```text
all_train_data[0][userId]
all_train_data[1][userId]
all_train_data[2][userId]
```

This preserves UFGraphFR's one-user-one-client assumption while avoiding user/list-index mismatch.

### `engine.py`

Evaluation now uses the real user id from evaluation data:

```python
user = int(test_users[eval_idx].item())
```

It no longer assumes:

```python
user == eval_idx
```

Negative samples are also sliced by evaluation instance:

```python
negative_item = negative_items[
    eval_idx * eval_num_negatives : (eval_idx + 1) * eval_num_negatives
]
```

This fixes the MCP user alignment problem.

When the same user has multiple evaluation positives, metrics use evaluation-instance ids internally to avoid mixing positives during ranking, while model lookup still uses the real `uid`.

### `utils.py`

Graph construction and message passing no longer assume participating user ids are `0..n-1`.

They now map real user ids to temporary matrix rows:

```python
users = sorted(round_user_params.keys())
for row, user in enumerate(users):
    matrix[row] = ...
```

After aggregation, rows are mapped back to the original user ids.

### `embedding.py`

Added a minimal MCP user prompt:

```text
The MCP user has internal id {uid} and original client id {original_uid}.
```

Heavy embedding dependencies are imported lazily so data-interface tests do not fail when joint embedding is disabled.

## Current Validation

Completed checks:

```powershell
python -m py_compile data.py train.py engine.py utils.py embedding.py prepare_mcp_user_split.py
git diff --check
```

Both checks passed.

Full training was not executed in this environment because:

- The system Python has incompatible `pandas` and `numpy` versions.
- The bundled Codex Python has `pandas`, but does not have `torch`.

The MCP split script and CSV statistics were validated successfully.

## Next Recommended Branch

The next branch should focus on MCP semantic modeling, not data protocol.

Suggested branch purpose:

```text
MCP Interaction-Context Transformer
```

Recommended scope:

- Parse raw MCP JSON files.
- Extract item text fields.
- Build item text embeddings.
- Construct text-similarity-based item ordering.
- Reinterpret Transformer as an Interaction-Context Transformer rather than a temporal sequence model.
- Optionally integrate `item_features.npy` or newly generated text embeddings into item representation.

This should be separate from `mcp-data-protocol` to keep data-interface changes and model-innovation changes cleanly separated.
