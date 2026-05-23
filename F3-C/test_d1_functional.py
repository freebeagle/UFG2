"""D1 functional test: item_text_proj + item_alpha federated communication pipeline.

Tests per MCP_UFGraphFR_NEXT_REQUIREMENTS.md D1 section:
  1. item_text_proj.weight shape = [32, 384]
  2. item_text_proj.bias shape = [32]
  3. Server FedAvg aggregation preserves shapes
  4. Model can load global item_text_proj
  5. get_item_embedding forward shape is correct
"""
import copy
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine
from mymodel import UFGraphFR

LATENT_DIM = 32
TEXT_DIM = 384
NUM_ITEMS = 5837
NUM_CLIENTS = 2

# ---- Test 1 & 2: Upload param shapes ----
print("=" * 60)
print("Test 1 & 2: item_text_proj and item_alpha shapes at upload")
print("=" * 60)

config = {
    'num_users': 2, 'num_items': NUM_ITEMS, 'latent_dim': LATENT_DIM,
    'layers': [64, 32, 16, 8], 'use_jointembedding': False,
    'use_item_attribute': True, 'item_attribute_set': 'A',
    'mcp_item_feature_path': '',  # use default path construction
    'item_attribute_alpha': 1.0, 'embed_dim': 100, 'use_transfermer': False,
    'use_cuda': False, 'use_mps': False,
}

model = UFGraphFR(config)
assert model.item_text_proj is not None, "item_text_proj not created"
assert model.item_alpha is not None, "item_alpha not created"

proj_w = model.item_text_proj.weight.data
proj_b = model.item_text_proj.bias.data
alpha = model.item_alpha.data

assert tuple(proj_w.shape) == (LATENT_DIM, TEXT_DIM), \
    f"weight shape {tuple(proj_w.shape)} != ({LATENT_DIM}, {TEXT_DIM})"
assert tuple(proj_b.shape) == (LATENT_DIM,), \
    f"bias shape {tuple(proj_b.shape)} != ({LATENT_DIM},)"
assert alpha.numel() == 1, f"alpha shape {tuple(alpha.shape)} != scalar"

print(f"  item_text_proj.weight: {tuple(proj_w.shape)}  OK")
print(f"  item_text_proj.bias:   {tuple(proj_b.shape)}  OK")
print(f"  item_alpha:            {tuple(alpha.shape)} value={alpha.item():.4f}  OK")

# ---- Test 3: FedAvg aggregation preserves shapes ----
print("\n" + "=" * 60)
print("Test 3: Server FedAvg aggregation preserves shapes")
print("=" * 60)

engine = Engine(config)
round_params = {}
for c in range(NUM_CLIENTS):
    round_params[c] = {
        'embedding_item.weight': torch.randn(NUM_ITEMS, LATENT_DIM),
        'embedding_user.weight': torch.randn(LATENT_DIM * 100),
        'item_text_proj.weight': torch.randn(LATENT_DIM, TEXT_DIM) + c * 0.1,
        'item_text_proj.bias': torch.randn(LATENT_DIM) + c * 0.1,
        'item_alpha': torch.tensor([0.3 + c * 0.2]),
    }

engine._fedavg_item_text_params(round_params)

agg_w = engine.server_model_param['item_text_proj.weight']
agg_b = engine.server_model_param['item_text_proj.bias']
agg_alpha = engine.server_model_param['item_alpha']

assert tuple(agg_w.shape) == (LATENT_DIM, TEXT_DIM), \
    f"aggregated weight shape {tuple(agg_w.shape)} != ({LATENT_DIM}, {TEXT_DIM})"
assert tuple(agg_b.shape) == (LATENT_DIM,), \
    f"aggregated bias shape {tuple(agg_b.shape)} != ({LATENT_DIM},)"
assert agg_alpha.numel() == 1, f"aggregated alpha shape != scalar"

expected_alpha = (0.3 + 0.5) / 2
assert abs(agg_alpha.item() - expected_alpha) < 1e-4, \
    f"FedAvg alpha {agg_alpha.item():.4f} != expected {expected_alpha:.4f}"

print(f"  Aggregated weight: {tuple(agg_w.shape)}  OK")
print(f"  Aggregated bias:   {tuple(agg_b.shape)}  OK")
print(f"  Aggregated alpha:  {agg_alpha.item():.4f} (expected {expected_alpha:.4f})  OK")

# ---- Test 4: Model loads global params ----
print("\n" + "=" * 60)
print("Test 4: Client model loads global item_text_proj + item_alpha")
print("=" * 60)

model2 = UFGraphFR(config)
state_dict = model2.state_dict()
state_dict['item_text_proj.weight'] = copy.deepcopy(agg_w)
state_dict['item_text_proj.bias'] = copy.deepcopy(agg_b)
state_dict['item_alpha'] = copy.deepcopy(agg_alpha)
model2.load_state_dict(state_dict)

assert torch.allclose(model2.item_text_proj.weight.data, agg_w), "weight not loaded"
assert torch.allclose(model2.item_text_proj.bias.data, agg_b), "bias not loaded"
assert torch.allclose(model2.item_alpha.data, agg_alpha), "alpha not loaded"
print("  Global params loaded into client model  OK")

# ---- Test 5: get_item_embedding forward shape ----
print("\n" + "=" * 60)
print("Test 5: get_item_embedding forward shape")
print("=" * 60)

batch_size = 4
item_ids = torch.randint(0, NUM_ITEMS, (batch_size,))
item_emb = model2.get_item_embedding(item_ids)

assert tuple(item_emb.shape) == (batch_size, LATENT_DIM), \
    f"output shape {tuple(item_emb.shape)} != ({batch_size}, {LATENT_DIM})"
print(f"  Input:  {tuple(item_ids.shape)}")
print(f"  Output: {tuple(item_emb.shape)}  OK")

# Alpha close to 0.5 initially (sigmoid(0) ≈ 0.5)
print(f"  Learnable alpha (raw): {model2.item_alpha.item():.4f}")
print(f"  Effective alpha (sigmoid): {torch.sigmoid(model2.item_alpha).item():.4f}")

# Verify fusion: e_item = sigmoid(alpha)*e_id + (1-sigmoid(alpha))*e_attr
model3 = UFGraphFR(config)
e_id = model3.embedding_item(item_ids).float()
item_text_feats = model3.item_text_features[item_ids].float()
e_attr = model3.item_text_proj(item_text_feats)
alpha_val = torch.sigmoid(model3.item_alpha)
expected_fusion = alpha_val * e_id + (1 - alpha_val) * e_attr
actual_fusion = model3.get_item_embedding(item_ids)
assert torch.allclose(actual_fusion, expected_fusion, atol=1e-5), \
    f"Fusion formula mismatch"
print(f"  Fusion formula verified: sigmoid(alpha)*e_id + (1-sigmoid(alpha))*e_attr  OK")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED")
print("=" * 60)
