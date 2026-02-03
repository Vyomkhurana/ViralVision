"""Data preprocessing module for ViralVision.

This module handles loading, cleaning, and feature engineering for video data.
"""

import os
import logging
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_VIDEOS_FILE,
    NUMERIC_COLUMNS, LOG_FORMAT, LOG_LEVEL
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

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

logger.info(f"Loading data from: {raw_path}")

try:
    df = pd.read_csv(raw_path)
except Exception as e:
    logger.error(f"Error reading CSV file: {e}")
    raise Exception(f"Error reading CSV file: {e}")

#  INITIAL INSPECTION #

logger.info(f"Shape of data (rows, columns): {df.shape}")
logger.info(f"Column names: {list(df.columns)}")
logger.debug(f"First 5 rows:\n{df.head()}")

# basic cleaning #
# convert numeric column from string to numeric (yt data)#

for col in NUMERIC_COLUMNS:
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

logger.info("✨ Added 7 new features:")
features = [
    "title_word_count", "title_uppercase_ratio", "title_has_question",
    "title_has_exclamation", "day_of_week", "hour_of_day", "is_weekend"
]
for feature in features:
    logger.info(f"   - {feature}")


# engagement matrics #

# Like ratio 
df["like_ratio"] = df["like_count"] / df["view_count"]

# Comment ratio
df["comment_ratio"] = df["comment_count"] / df["view_count"]

# save processed data #

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

output_path = os.path.join(PROCESSED_DATA_DIR, PROCESSED_VIDEOS_FILE)
try:
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to: {output_path}")
    logger.info(f"Total rows processed: {len(df)}")
except Exception as e:
    logger.error(f"Failed to save processed data: {e}")
    raise