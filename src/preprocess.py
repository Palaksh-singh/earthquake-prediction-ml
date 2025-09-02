
### ✅ `src/preprocess.py`
import argparse
import os
import pandas as pd

def to_unix_timestamp(date_series, time_series):
    dt = pd.to_datetime(
        date_series.astype(str) + " " + time_series.astype(str),
        errors="coerce",
        format="%m/%d/%Y %H:%M:%S"  # matches the example format
    )
    # convert to unix seconds
    return (dt.view("int64") // 10**9)

def main(args):
    assert os.path.exists(args.input), f"Input file not found: {args.input}"
    df = pd.read_csv(args.input)

    # Keep only required columns
    cols = ["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in dataset.")
    df = df[cols].copy()

    # Create Timestamp and drop invalid rows
    df["Timestamp"] = to_unix_timestamp(df["Date"], df["Time"])
    df = df.dropna(subset=["Timestamp", "Latitude", "Longitude", "Depth", "Magnitude"])

    # Reorder & cast
    final_df = df[["Timestamp", "Latitude", "Longitude", "Magnitude", "Depth"]].copy()
    final_df = final_df.astype({
        "Timestamp": "int64",
        "Latitude": "float64",
        "Longitude": "float64",
        "Magnitude": "float64",
        "Depth": "float64"
    })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(f"[OK] Processed data saved to {args.output}")
    print(final_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/database.csv")
    parser.add_argument("--output", default="data/processed.csv")
    main(parser.parse_args())
