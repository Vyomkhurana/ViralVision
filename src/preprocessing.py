import os
import pandas as pd
import numpy as np

# laod raw data #

RAW_DATA_DIR = "data/raw"

#get csv files from raw folder#

raw_files = sorted(
    [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
)

if not raw_files:
    raise FileNotFoundError("No raw CSV files found in data/raw")

latest_file = raw_files[-1]
raw_path = os.path.join(RAW_DATA_DIR, latest_file)


print(f"Loading data from: {raw_path}")

df = pd.read_csv(raw_path)

#  INITIAL INSPECTIOn #

print("\nShape of data (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# basic cleaning #
# convert numeric column from string to numeric (yt data)#

numeric_columns = ["view_count", "like_count", "comment_count"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Remove rows with missing essential values#

df.dropna(subset=["view_count"], inplace=True)

#feature engineering#

# Title length
df["title_length"] = df["title"].astype(str).apply(len)

# Description length
df["description_length"] = df["description"].astype(str).apply(len)

# Tag count
df["tag_count"] = df["tags"].apply(
    lambda x: len(str(x).split("|")) if pd.notna(x) else 0
)


# engagement matrics #

# Like ratio 
df["like_ratio"] = df["like_count"] / df["view_count"]

# Comment ratio
df["comment_ratio"] = df["comment_count"] / df["view_count"]

# save processed data #

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

output_path = os.path.join(PROCESSED_DIR, "processed_videos.csv")
df.to_csv(output_path, index=False)

print(f"\nProcessed data saved to: {output_path}")