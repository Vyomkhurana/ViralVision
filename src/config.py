"""Configuration management for ViralVision.

Centralized configuration for paths, model parameters, and feature definitions.
"""

import os
from typing import List

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data file names
PROCESSED_VIDEOS_FILE = "processed_videos.csv"
LABELED_VIDEOS_FILE = "labeled_videos.csv"

# Model file names
MODEL_FILE = "virality_model.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"

# Feature definitions
NUMERIC_COLUMNS = ["view_count", "like_count", "comment_count"]

FEATURE_COLUMNS: List[str] = [
    # Original features
    "title_length",
    "description_length",
    "tag_count",
    "like_ratio",
    "comment_ratio",
    # Advanced features
    "title_word_count",
    "title_uppercase_ratio",
    "title_has_question",
    "title_has_exclamation",
    "day_of_week",
    "hour_of_day",
    "is_weekend",
]

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "max_depth": None,
    "min_samples_split": 2,
}

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# YouTube API settings
DEFAULT_MAX_RESULTS = 50
DEFAULT_REGION_CODE = "US"

# Virality thresholds (views)
VIRAL_THRESHOLD = 1_000_000
MEDIUM_THRESHOLD = 100_000

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = "INFO"
