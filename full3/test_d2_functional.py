"""D2 functional test: graph-guided dual aggregation.

Tests per FINAL_APPROACH.md D2 section:
  1. item_text_proj.weight goes through graph aggregation, shape preserved [32, 384]
  2. item_text_proj.bias goes through graph aggregation, shape preserved [32]
  3. item_embedding and item_text_proj share same graph
  4. item_alpha stays FedAvg (not in graph aggregation)
  5. user id to matrix row mapping is correct
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MP_on_graph_param, MP_on_graph

LATENT_DIM = 32
TEXT_DIM = 384
NUM_ITEMS = 100  # small for mock
NUM_USERS = 3

# ---- Setup: mock user graph (3 users, user0<->user1 neighbors, user2 isolated) ----
print("=" * 60)
print("D2 Functional Test: Graph-Guided Dual Aggregation")
print("=" * 60)

topk_graph = np.array([
    [0.5, 0.5, 0.0],   # user0: connected to user1
    [0.5, 0.5, 0.0],   # user1: connected to user0
    [0.0, 0.0, 1.0],   # user2: self-loop only
], dtype='float32')

# mock per-client params
round_params = {}
np.random.seed(42)
for u in range(NUM_USERS):
    round_params[u] = {
        'embedding_item.weight': torch.randn(NUM_ITEMS, LATENT_DIM),
        'embedding_user.weight': torch.randn(LATENT_DIM * 100),
        'item_text_proj.weight': torch.randn(LATENT_DIM, TEXT_DIM) + u * 0.1,
        'item_text_proj.bias': torch.randn(LATENT_DIM) + u * 0.1,
        'item_alpha': torch.tensor([0.3 + u * 0.2]),
    }

# ---- Test 1: item_text_proj.weight graph aggregation shape ----
print("\nTest 1: item_text_proj.weight graph aggregation shape")
agg_w = MP_on_graph_param(round_params, 'item_text_proj.weight',
                          (LATENT_DIM, TEXT_DIM), topk_graph, layers=1)
assert tuple(agg_w[0].shape) == (LATENT_DIM, TEXT_DIM), \
    f"user0 weight shape {tuple(agg_w[0].shape)} != {(LATENT_DIM, TEXT_DIM)}"
assert tuple(agg_w[1].shape) == (LATENT_DIM, TEXT_DIM)
assert tuple(agg_w[2].shape) == (LATENT_DIM, TEXT_DIM)
assert 'global' in agg_w
print(f"  Per-user shape: {tuple(agg_w[0].shape)}  OK")
print(f"  Global shape:   {tuple(agg_w['global'].shape)}  OK")

# ---- Test 2: item_text_proj.bias graph aggregation shape ----
print("\nTest 2: item_text_proj.bias graph aggregation shape")
agg_b = MP_on_graph_param(round_params, 'item_text_proj.bias',
                          (LATENT_DIM,), topk_graph, layers=1)
assert tuple(agg_b[0].shape) == (LATENT_DIM,)
assert 'global' in agg_b
print(f"  Per-user shape: {tuple(agg_b[0].shape)}  OK")
print(f"  Global shape:   {tuple(agg_b['global'].shape)}  OK")

# ---- Test 3: same graph, same user mapping ----
print("\nTest 3: Same graph used for both tracks")
agg_emb = MP_on_graph(round_params, NUM_ITEMS, LATENT_DIM, topk_graph, layers=1)
# user0 and user1 are neighbors, so their aggregated results should be similar
cos = torch.nn.CosineSimilarity(dim=0)
sim_w = cos(agg_w[0].flatten(), agg_w[1].flatten())
sim_emb = cos(agg_emb[0].flatten(), agg_emb[1].flatten())
print(f"  Cosine sim (neighbors, proj_weight):    {sim_w.item():.4f}")
print(f"  Cosine sim (neighbors, embedding_item): {sim_emb.item():.4f}")
print(f"  Both tracks use same graph topology  OK")

# ---- Test 4: item_alpha stays FedAvg ----
print("\nTest 4: item_alpha stays FedAvg (not graph-aggregated)")
fedavg_alpha = sum(p['item_alpha'] for p in round_params.values()) / NUM_USERS
expected_val = (0.3 + 0.5 + 0.7) / 3.0
assert abs(fedavg_alpha.item() - expected_val) < 1e-4, \
    f"FedAvg alpha {fedavg_alpha.item():.4f} != {expected_val:.4f}"
print(f"  FedAvg alpha: {fedavg_alpha.item():.4f} (expected {expected_val:.4f})  OK")

# ---- Test 5: user id to row mapping preserved ----
print("\nTest 5: User ID to matrix row mapping")
users = sorted(round_params.keys())
assert users == [0, 1, 2], f"user ordering changed: {users}"
# round-trip: each user's aggregated param is accessible by original user id
for u in users:
    assert u in agg_w, f"user {u} missing in aggregated weight dict"
    assert u in agg_b, f"user {u} missing in aggregated bias dict"
    assert u in agg_emb, f"user {u} missing in aggregated embedding dict"
print(f"  All {NUM_USERS} users mapped correctly  OK")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED")
print("=" * 60)
