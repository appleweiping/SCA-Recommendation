import pandas as pd
from pathlib import Path


def convert_ml1m(input_dir, output_path):
    input_dir = Path(input_dir)

    ratings_file = input_dir / "ratings.dat"

    # MovieLens-1M 用 "::" 分隔
    df = pd.read_csv(
        ratings_file,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    # 推荐系统通常只用 implicit（有行为就是正样本）
    df = df[["user_id", "item_id", "timestamp"]]

    df.to_csv(output_path, index=False)

    print("✅ Converted to CSV:", output_path)
    print(df.head())


if __name__ == "__main__":
    convert_ml1m(
        input_dir="data/raw/ml-1m",
        output_path="data/raw/ml-1m/interactions.csv"
    )