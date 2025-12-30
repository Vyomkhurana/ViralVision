import os
from turtle import pd
import pandas as np
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
