import os
import pandas as pd
import numpy as np
from datetime import datetime

# load raw data #

RAW_DATA_DIR = "data/raw"

#get csv files from raw folder#

try:
    raw_files = sorted(
        [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    )
except FileNotFoundError:
    raise FileNotFoundError(f"Directory not found: {RAW_DATA_DIR}")

if not raw_files:
    raise FileNotFoundError("No raw CSV files found in data/raw")

latest_file = raw_files[-1]
raw_path = os.path.join(RAW_DATA_DIR, latest_file)


print(f"Loading data from: {raw_path}")

try:
    df = pd.read_csv(raw_path)
except Exception as e:
    raise Exception(f"Error reading CSV file: {e}")

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

# ==========================================
# NEW FEATURES FOR BETTER PREDICTIONS
# ==========================================

# 1. WORD COUNT (different from character length)
# Why: Titles with 8-12 words often perform better
df["title_word_count"] = df["title"].astype(str).apply(lambda x: len(x.split()))

# 2. UPPERCASE RATIO (how much of title is CAPS)
# Why: ALL CAPS or excessive caps can indicate clickbait
def calc_uppercase_ratio(text):
    text = str(text)
    if len(text) == 0:
        return 0
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0
    return sum(1 for c in letters if c.isupper()) / len(letters)

df["title_uppercase_ratio"] = df["title"].apply(calc_uppercase_ratio)

# 3. HAS QUESTION MARK (asking questions engages viewers)
# Why: Questions create curiosity
df["title_has_question"] = df["title"].astype(str).str.contains(r"\?", regex=True).astype(int)

# 4. HAS EXCLAMATION (excitement/urgency)
# Why: Creates emotional response
df["title_has_exclamation"] = df["title"].astype(str).str.contains("!", regex=False).astype(int)

# 5. TIME-BASED FEATURES (when was it published?)
# Convert published_at to datetime
df["published_datetime"] = pd.to_datetime(df["published_at"], errors="coerce")

# Day of week (0=Monday, 6=Sunday)
# Why: Weekends might have different viral patterns
df["day_of_week"] = df["published_datetime"].dt.dayofweek

# Hour of day (0-23)
# Why: Upload time affects initial views
df["hour_of_day"] = df["published_datetime"].dt.hour

# Is weekend? (Saturday or Sunday)
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

print(f"\n✨ Added 7 new features:")
print("   - title_word_count")
print("   - title_uppercase_ratio")
print("   - title_has_question")
print("   - title_has_exclamation")
print("   - day_of_week")
print("   - hour_of_day")
print("   - is_weekend")


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