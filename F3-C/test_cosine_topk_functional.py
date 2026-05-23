import numpy as np
import torch

from utils import construct_user_relation_graph_via_user, select_topk_neighboehood


def main():
    round_user_params = {
        0: {'embedding_user.weight': torch.tensor([1.0, 0.0])},
        1: {'embedding_user.weight': torch.tensor([0.98, 0.02])},
        2: {'embedding_user.weight': torch.tensor([-1.0, 0.0])},
    }

    graph = construct_user_relation_graph_via_user(
        round_user_params, latent_dim=2, similarity_metric='cosine')
    topk = select_topk_neighboehood(
        graph, neighborhood_size=1, neighborhood_threshold=1.0)

    assert topk[0, 0] == 0.0, 'self should be excluded before top-k'
    assert topk[0, 1] == 1.0, 'most similar user should be selected'
    assert topk[0, 2] == 0.0, 'least similar user should not be selected'

    topk_large = select_topk_neighboehood(
        graph, neighborhood_size=10, neighborhood_threshold=1.0)
    assert topk_large[0, 0] == 0.0, 'self should stay excluded for large k'
    assert np.isclose(topk_large[0].sum(), 1.0), 'weights should sum to one'
    assert np.isclose(topk_large[0, 1], 0.5), 'weight should use actual_k'
    assert np.isclose(topk_large[0, 2], 0.5), 'weight should use actual_k'

    print('Cosine top-k functional test passed')


if __name__ == '__main__':
    main()
