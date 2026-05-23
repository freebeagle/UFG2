import argparse
import os
import sys
import types


class _FakeLongTensor(list):
    def view(self, *args):
        return self

    def tolist(self):
        return list(self)

    @property
    def data(self):
        return self


class _FakeFloatTensor(_FakeLongTensor):
    pass


def _install_torch_stub_if_needed():
    try:
        import torch  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    torch_stub = types.ModuleType("torch")
    torch_stub.LongTensor = lambda values: _FakeLongTensor(values)
    torch_stub.FloatTensor = lambda values: _FakeFloatTensor(values)

    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = object
    data_mod.DataLoader = object
    utils_mod.data = data_mod
    torch_stub.utils = utils_mod

    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod


def _install_tqdm_stub_if_needed():
    try:
        import tqdm  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    tqdm_stub = types.ModuleType("tqdm")

    def passthrough(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    tqdm_stub.tqdm = passthrough
    sys.modules["tqdm"] = tqdm_stub


def load_mcp_split(data_dir):
    import pandas as pd

    def read_split(name):
        path = os.path.join(data_dir, name + ".csv")
        frame = pd.read_csv(path)
        frame = frame.rename(columns={"uid": "userId", "iid": "itemId"})
        frame["rating"] = 1.0
        return frame[["userId", "itemId", "rating"]]

    train_ratings = read_split("train")
    val_ratings = read_split("vali")
    test_ratings = read_split("test")
    ratings = pd.concat([train_ratings, val_ratings, test_ratings], ignore_index=True)
    return ratings, train_ratings, val_ratings, test_ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp_data_dir", default="data/mcp_user_split")
    parser.add_argument("--num_negatives", type=int, default=4)
    args = parser.parse_args()

    _install_torch_stub_if_needed()
    _install_tqdm_stub_if_needed()
    from data import SampleGenerator

    ratings, train_ratings, val_ratings, test_ratings = load_mcp_split(args.mcp_data_dir)
    num_users = int(ratings["userId"].max()) + 1
    num_items = int(ratings["itemId"].max()) + 1

    sample_generator = SampleGenerator(
        ratings=ratings,
        train_ratings=train_ratings,
        val_ratings=val_ratings,
        test_ratings=test_ratings,
        num_users=num_users,
        num_items=num_items,
    )

    all_train_data = sample_generator.store_all_train_data(args.num_negatives)
    validate_data = sample_generator.validate_data
    test_data = sample_generator.test_data

    assert len(all_train_data) == 3
    assert len(all_train_data[0]) == num_users
    assert len(all_train_data[1]) == num_users
    assert len(all_train_data[2]) == num_users
    assert len(validate_data[0]) == len(validate_data[1])
    assert len(test_data[0]) == len(test_data[1])
    assert len(validate_data[2]) == 99 * len(validate_data[0])
    assert len(test_data[2]) == 99 * len(test_data[0])

    first_test_user = int(test_data[0][0])
    assert first_test_user != 0 or len(test_data[0]) == 0
    assert len(all_train_data[0][first_test_user]) > 0

    print("MCP data protocol smoke test passed")
    print("data_dir:", args.mcp_data_dir)
    print("num_users:", num_users)
    print("num_items:", num_items)
    print("train_rows:", len(train_ratings))
    print("val_rows:", len(val_ratings))
    print("test_rows:", len(test_ratings))
    print("first_test_user:", first_test_user)
    print("first_test_user_train_examples:", len(all_train_data[0][first_test_user]))


if __name__ == "__main__":
    main()
