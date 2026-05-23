import argparse
import csv
import os
import random
import shutil
from collections import defaultdict


def read_pairs(path):
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((int(row["uid"]), int(row["iid"])))
    return pairs


def write_pairs(path, pairs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "iid"])
        writer.writerows(pairs)


def build_user_static_split(input_dir, output_dir, seed):
    all_pairs = []
    for split in ("train", "vali", "test"):
        all_pairs.extend(read_pairs(os.path.join(input_dir, split + ".csv")))

    user_items = defaultdict(list)
    for uid, iid in sorted(set(all_pairs)):
        user_items[uid].append(iid)

    rng = random.Random(seed)
    train_pairs = []
    val_pairs = []
    test_pairs = []

    for uid in sorted(user_items):
        items = user_items[uid][:]
        rng.shuffle(items)

        if len(items) >= 3:
            test_item = items[0]
            val_item = items[1]
            train_items = items[2:]
            val_pairs.append((uid, val_item))
            test_pairs.append((uid, test_item))
        elif len(items) == 2:
            test_item = items[0]
            train_items = items[1:]
            test_pairs.append((uid, test_item))
        else:
            train_items = items

        train_pairs.extend((uid, iid) for iid in train_items)

    os.makedirs(output_dir, exist_ok=True)
    write_pairs(os.path.join(output_dir, "train.csv"), train_pairs)
    write_pairs(os.path.join(output_dir, "vali.csv"), val_pairs)
    write_pairs(os.path.join(output_dir, "test.csv"), test_pairs)

    for filename in ("filtered_client_id_mapping.csv", "item_features.npy"):
        src = os.path.join(input_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, filename))

    return {
        "train_rows": len(train_pairs),
        "val_rows": len(val_pairs),
        "test_rows": len(test_pairs),
        "users": len(user_items),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/mcp_staging")
    parser.add_argument("--output_dir", default="data/mcp_user_split")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    stats = build_user_static_split(args.input_dir, args.output_dir, args.seed)
    print(stats)


if __name__ == "__main__":
    main()
