import torch

from engine import Engine


def main():
    config = {
        'alias': 'UFGraphFR',
        'use_cuda': False,
        'use_mps': False,
        'use_item_attribute': True,
    }
    engine = Engine(config)
    user = 0
    user_param_dict = {
        'embedding_item.weight': torch.zeros(2, 2),
        'item_alpha': torch.zeros(2, 1),
        'item_text_proj.weight': torch.zeros(2, 3),
        'item_text_proj.bias': torch.zeros(2),
        'local_only.weight': torch.full((1,), 7.0),
    }
    engine.client_model_params[user] = {
        'embedding_item.weight': torch.full((2, 2), 1.0),
        'item_alpha': torch.full((2, 1), 1.0),
        'item_text_proj.weight': torch.full((2, 3), 1.0),
        'item_text_proj.bias': torch.full((2,), 1.0),
        'local_only.weight': torch.full((1,), 3.0),
    }

    for key, value in engine.client_model_params[user].items():
        user_param_dict[key] = value.clone()

    engine.server_model_param = {
        'embedding_item.weight': {
            user: torch.full((2, 2), 2.0),
            1: torch.full((2, 2), 9.0),
        },
        'item_alpha': {
            user: torch.full((2, 1), 4.0),
            1: torch.full((2, 1), 9.0),
        },
        'item_text_proj.weight': torch.full((2, 3), 5.0),
        'item_text_proj.bias': torch.full((2,), 6.0),
    }

    engine._sync_server_aggregated_params_for_eval(user_param_dict, user)

    assert torch.equal(user_param_dict['embedding_item.weight'], torch.full((2, 2), 2.0))
    assert torch.equal(user_param_dict['item_alpha'], torch.full((2, 1), 4.0))
    assert torch.equal(user_param_dict['item_text_proj.weight'], torch.full((2, 3), 5.0))
    assert torch.equal(user_param_dict['item_text_proj.bias'], torch.full((2,), 6.0))
    assert torch.equal(user_param_dict['local_only.weight'], torch.full((1,), 3.0))

    print('Server-synced eval functional test passed')


if __name__ == '__main__':
    main()
