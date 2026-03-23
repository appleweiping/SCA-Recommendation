import pandas as pd
from pathlib import Path


def split_leave_one_out(input_path, output_dir):
    df = pd.read_csv(input_path)

    # 必须包含 user, item, timestamp
    assert {"user_id", "item_id", "timestamp"}.issubset(df.columns)

    df = df.sort_values(["user_id", "timestamp"])

    train_rows = []
    valid_rows = []
    test_rows = []

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")

        if len(group) < 3:
            # 太少的用户直接全放 train
            train_rows.append(group)
            continue

        train = group.iloc[:-2]
        valid = group.iloc[-2:-1]
        test = group.iloc[-1:]

        train_rows.append(train)
        valid_rows.append(valid)
        test_rows.append(test)

    train_df = pd.concat(train_rows)
    valid_df = pd.concat(valid_rows)
    test_df = pd.concat(test_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("✅ Split done")
    print(f"train: {len(train_df)}")
    print(f"valid: {len(valid_df)}")
    print(f"test: {len(test_df)}")


if __name__ == "__main__":
    split_leave_one_out(
        input_path="data/raw/ml-1m/interactions.csv",
        output_dir="data/processed/ml-1m"
    )