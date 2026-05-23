"""D3 functional test: directional calibration (per-row L2 normalization).

Tests per FINAL_APPROACH.md D3 section:
  1. Same direction, different norms → normalized directions are close
  2. Different directions → normalized still differ (semantic diff preserved)
  3. Graph aggregation with calibrated params, shape preserved
  4. No NaN / inf after normalization
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MP_on_graph_param, MP_on_graph

LATENT_DIM = 32
TEXT_DIM = 384
NUM_ITEMS = 100
NUM_USERS = 3

print("=" * 60)
print("D3 Functional Test: Directional Calibration")
print("=" * 60)

# ---- Test 1: Same direction, different norms → normalized close ----
print("\nTest 1: Same direction, different norms → calibrated close")
base_direction = torch.randn(1, TEXT_DIM)
base_direction = base_direction / base_direction.norm(dim=1, keepdim=True)

w_a = base_direction.repeat(LATENT_DIM, 1)  # all rows same direction, norm≈1
w_b = base_direction.repeat(LATENT_DIM, 1) * 5.0  # same direction, norm≈5

# Apply same calibration as D3 engine code
def calibrate(w):
    row_norms = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return w / row_norms

cal_a = calibrate(w_a)
cal_b = calibrate(w_b)

diff = (cal_a - cal_b).abs().max().item()
assert diff < 1e-4, f"Calibrated matrices should be nearly identical, max diff={diff:.6f}"
cos = torch.nn.CosineSimilarity(dim=1)
sim = cos(cal_a, cal_b).mean().item()
assert sim > 0.9999, f"Calibrated cosine sim {sim:.6f} should be ~1.0"
print(f"  Before calibration: norm_a={w_a.norm(dim=1).mean():.2f}, norm_b={w_b.norm(dim=1).mean():.2f}")
print(f"  After calibration:  norm_a={cal_a.norm(dim=1).mean():.2f}, norm_b={cal_b.norm(dim=1).mean():.2f}")
print(f"  Cosine similarity after calibration: {sim:.6f}  OK")

# ---- Test 2: Different directions → calibrated still differ ----
print("\nTest 2: Different directions → calibrated still differ")
w_c = torch.randn(LATENT_DIM, TEXT_DIM) * 0.5
w_d = torch.randn(LATENT_DIM, TEXT_DIM) * 3.0

cal_c = calibrate(w_c)
cal_d = calibrate(w_d)

sim_cd = cos(cal_c, cal_d).mean().item()
assert sim_cd < 0.99, f"Different directions should not become identical after calibration, sim={sim_cd:.6f}"
print(f"  Cosine similarity after calibration: {sim_cd:.6f} (should be < 0.99)  OK")

# ---- Test 3: Graph aggregation with calibrated params, shape OK ----
print("\nTest 3: Graph aggregation with calibrated params")
topk_graph = np.array([
    [0.5, 0.5, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.0, 1.0],
], dtype='float32')

# Build params with DRAMATICALLY different norms per client
round_params = {}
for u in range(NUM_USERS):
    w = torch.randn(LATENT_DIM, TEXT_DIM) + u * 0.1
    # client 1 gets 10x norm
    if u == 1:
        w = w * 10.0
    round_params[u] = {
        'item_text_proj.weight': calibrate(w),  # pre-calibrated
        'item_text_proj.bias': torch.randn(LATENT_DIM) + u * 0.1,
        'embedding_item.weight': torch.randn(NUM_ITEMS, LATENT_DIM),
    }

agg_w = MP_on_graph_param(round_params, 'item_text_proj.weight',
                          (LATENT_DIM, TEXT_DIM), topk_graph, layers=1)
agg_b = MP_on_graph_param(round_params, 'item_text_proj.bias',
                          (LATENT_DIM,), topk_graph, layers=1)

assert tuple(agg_w[0].shape) == (LATENT_DIM, TEXT_DIM)
assert tuple(agg_b[0].shape) == (LATENT_DIM,)
assert 'global' in agg_w
print(f"  Weight shape: {tuple(agg_w[0].shape)}  OK")
print(f"  Bias shape:   {tuple(agg_b[0].shape)}  OK")

# ---- Test 4: No NaN / inf ----
print("\nTest 4: No NaN / inf after calibration")
has_nan = torch.isnan(agg_w['global']).any().item()
has_inf = torch.isinf(agg_w['global']).any().item()
assert not has_nan, "NaN detected in aggregated weight"
assert not has_inf, "Inf detected in aggregated weight"

# Edge case: zero row → should produce 0 after calibration, not NaN
w_zero = torch.zeros(LATENT_DIM, TEXT_DIM)
w_zero[0, 0] = 0.0  # explicitly zero row
cal_zero = calibrate(w_zero)
assert not torch.isnan(cal_zero).any(), "Zero row should not produce NaN (clamp min=1e-8)"
assert not torch.isinf(cal_zero).any(), "Zero row should not produce Inf"
print(f"  Zero-row calibration: no NaN/Inf  OK")

print("\n" + "=" * 60)
print("ALL 4 TESTS PASSED")
print("=" * 60)
