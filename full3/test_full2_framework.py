"""Full2 framework functional test.

This test validates the current full2/full3 framework contract:
  - MCP user metadata is deduplicated before merge.
  - item_alpha is per-item [num_items, 1], not scalar.
  - item_text_proj is a global model projector [latent_dim, text_dim].
  - item_alpha and embedding_item can use the same user-user graph.
  - item_text_proj.weight is row-L2 calibrated then FedAvg.
  - evaluation overlays server-aggregated full2 parameters.
  - cosine graph scores select similar users after the similarity fix.
"""
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Engine
from mymodel import UFGraphFR
from train import load_mcp_split
from utils import (
    MP_on_graph,
    construct_user_relation_graph_via_user,
    select_topk_neighboehood,
)


LATENT_DIM = 32
TEXT_DIM = 384
NUM_ITEMS = 5837


def assert_shape(tensor, shape, name):
    actual = tuple(tensor.shape)
    assert actual == tuple(shape), f"{name} shape {actual} != {tuple(shape)}"


def test_mcp_user_dedup():
    _, _, _, _, user_infos = load_mcp_split(
        "data/mcp_user_split",
        "data/mcp_prepare/user_side/raw/filtered_clients.json",
    )
    assert len(user_infos) == 186, f"user_infos rows {len(user_infos)} != 186"
    assert user_infos["uid"].is_unique, "uid should be unique after MCP merge"
    assert user_infos["original_uid"].is_unique, "original_uid should be unique after MCP merge"


def test_model_shapes_and_fusion():
    config = {
        "alias": "UFGraphFR",
        "num_users": 186,
        "num_items": NUM_ITEMS,
        "latent_dim": LATENT_DIM,
        "layers": [64, 32, 16, 8],
        "use_jointembedding": True,
        "use_item_attribute": True,
        "item_attribute_set": "A",
        "mcp_item_feature_path": "data/mcp_prepare/item_side/features/item_text_features_A.npy",
        "embed_dim": TEXT_DIM,
        "use_transfermer": False,
        "use_cuda": False,
        "use_mps": False,
    }
    model = UFGraphFR(config)

    assert_shape(model.embedding_item.weight.data, (NUM_ITEMS, LATENT_DIM), "embedding_item.weight")
    assert_shape(model.embedding_user.weight.data, (LATENT_DIM, TEXT_DIM), "embedding_user.weight")
    assert_shape(model.item_text_features, (NUM_ITEMS, TEXT_DIM), "item_text_features")
    assert_shape(model.item_text_proj.weight.data, (LATENT_DIM, TEXT_DIM), "item_text_proj.weight")
    assert_shape(model.item_text_proj.bias.data, (LATENT_DIM,), "item_text_proj.bias")
    assert_shape(model.item_alpha.data, (NUM_ITEMS, 1), "item_alpha")

    item_ids = torch.tensor([0, 1, NUM_ITEMS - 1], dtype=torch.long)
    e_id = model.embedding_item(item_ids).float()
    e_attr = model.item_text_proj(model.item_text_features[item_ids].float())
    alpha = torch.sigmoid(model.item_alpha[item_ids])
    expected = alpha * e_id + (1 - alpha) * e_attr
    actual = model.get_item_embedding(item_ids)
    assert_shape(actual, (3, LATENT_DIM), "fused item embedding")
    assert torch.allclose(actual, expected, atol=1e-6), "fusion formula mismatch"
    assert torch.allclose(torch.sigmoid(model.item_alpha[0]), torch.tensor([0.11920292]), atol=1e-6)


def test_full2_server_aggregation_contract():
    config = {
        "alias": "UFGraphFR",
        "num_users": 3,
        "num_items": NUM_ITEMS,
        "latent_dim": LATENT_DIM,
        "use_cuda": False,
        "use_mps": False,
        "use_item_attribute": True,
        "mp_layers": 1,
    }
    engine = Engine(config)
    topk_graph = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )

    round_params = {}
    for user in range(3):
        round_params[user] = {
            "embedding_item.weight": torch.randn(NUM_ITEMS, LATENT_DIM),
            "embedding_user.weight": torch.randn(LATENT_DIM * TEXT_DIM),
            "item_text_proj.weight": torch.randn(LATENT_DIM, TEXT_DIM) * (user + 1),
            "item_text_proj.bias": torch.randn(LATENT_DIM) + user,
            "item_alpha": torch.randn(NUM_ITEMS, 1) + user,
        }

    updated_item_embedding = MP_on_graph(round_params, NUM_ITEMS, LATENT_DIM, topk_graph, layers=1)
    engine._aggregate_item_text_params(round_params, topk_graph, mp_layers=1)

    assert_shape(updated_item_embedding[0], (NUM_ITEMS, LATENT_DIM), "aggregated embedding_item[0]")
    assert_shape(engine.server_model_param["item_alpha"][0], (NUM_ITEMS, 1), "aggregated item_alpha[0]")
    assert_shape(engine.server_model_param["item_text_proj.weight"], (LATENT_DIM, TEXT_DIM), "aggregated proj weight")
    assert_shape(engine.server_model_param["item_text_proj.bias"], (LATENT_DIM,), "aggregated proj bias")

    manual_w = sum(
        p["item_text_proj.weight"] / p["item_text_proj.weight"].norm(dim=1, keepdim=True).clamp(min=1e-8)
        for p in round_params.values()
    ) / len(round_params)
    manual_b = sum(p["item_text_proj.bias"] for p in round_params.values()) / len(round_params)
    assert torch.allclose(engine.server_model_param["item_text_proj.weight"], manual_w, atol=1e-6)
    assert torch.allclose(engine.server_model_param["item_text_proj.bias"], manual_b, atol=1e-6)


def test_server_synced_eval_overlay():
    config = {
        "alias": "UFGraphFR",
        "use_cuda": False,
        "use_mps": False,
        "use_item_attribute": True,
    }
    engine = Engine(config)
    user = 0
    user_param_dict = {
        "embedding_item.weight": torch.zeros(2, 2),
        "item_alpha": torch.zeros(2, 1),
        "item_text_proj.weight": torch.zeros(2, 3),
        "item_text_proj.bias": torch.zeros(2),
        "local_only.weight": torch.full((1,), 3.0),
    }
    original = copy.deepcopy(user_param_dict)
    engine.server_model_param = {
        "embedding_item.weight": {user: torch.full((2, 2), 2.0)},
        "item_alpha": {user: torch.full((2, 1), 4.0)},
        "item_text_proj.weight": torch.full((2, 3), 5.0),
        "item_text_proj.bias": torch.full((2,), 6.0),
    }

    engine._sync_server_aggregated_params_for_eval(user_param_dict, user)

    assert torch.equal(user_param_dict["embedding_item.weight"], torch.full((2, 2), 2.0))
    assert torch.equal(user_param_dict["item_alpha"], torch.full((2, 1), 4.0))
    assert torch.equal(user_param_dict["item_text_proj.weight"], torch.full((2, 3), 5.0))
    assert torch.equal(user_param_dict["item_text_proj.bias"], torch.full((2,), 6.0))
    assert torch.equal(user_param_dict["local_only.weight"], original["local_only.weight"])


def test_cosine_graph_selects_similar_users():
    round_params = {
        0: {"embedding_user.weight": torch.tensor([1.0, 0.0])},
        1: {"embedding_user.weight": torch.tensor([0.9, 0.1])},
        2: {"embedding_user.weight": torch.tensor([-1.0, 0.0])},
    }
    graph = construct_user_relation_graph_via_user(round_params, latent_dim=2, similarity_metric="cosine")
    topk = select_topk_neighboehood(graph, neighborhood_size=1, neighborhood_threshold=1.0)
    assert topk[0, 1] == 1.0, "user0 should select similar user1"
    assert topk[0, 2] == 0.0, "user0 should not select opposite user2"


def main():
    test_mcp_user_dedup()
    test_model_shapes_and_fusion()
    test_full2_server_aggregation_contract()
    test_server_synced_eval_overlay()
    test_cosine_graph_selects_similar_users()
    print("Full2 framework functional test passed")


if __name__ == "__main__":
    main()
