"""Model training module for ViralVision.

Trains machine learning models to predict video virality.
"""

import pandas as pd
import pickle
import os
import logging
from typing import Tuple, List
import numpy as np

# Import LabelEncoder to convert text labels into numbers
from sklearn.preprocessing import LabelEncoder

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split, cross_val_score

# Import RandomForestClassifier as ML model
from sklearn.ensemble import RandomForestClassifier

# Import accuracy_score to evaluate how good the model is
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (
    PROCESSED_DATA_DIR, LABELED_VIDEOS_FILE, MODEL_DIR,
    FEATURE_COLUMNS, MODEL_PARAMS, TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    MODEL_FILE, LABEL_ENCODER_FILE, FEATURE_NAMES_FILE,
    LOG_FORMAT, LOG_LEVEL
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


LABELED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, LABELED_VIDEOS_FILE)
logger.info(f"Loading labeled data from: {LABELED_DATA_PATH}")
try:
    df = pd.read_csv(LABELED_DATA_PATH)
    logger.info(f"Loaded {len(df)} labeled videos")
except FileNotFoundError:
    logger.error(f"File not found: {LABELED_DATA_PATH}")
    raise
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

# Select input features (X) - Now with 7 NEW features!
X = df[FEATURE_COLUMNS]

logger.info(f"📊 Training with {len(X.columns)} features:")
for i, feat in enumerate(X.columns, 1):
    logger.info(f"   {i}. {feat}")

# Select output label (y)
y = df["virality_label"]


# Create label encoder
label_encoder = LabelEncoder()

# Convert text labels into numbers
y_encoded = label_encoder.fit_transform(y)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# Create Random Forest model
model = RandomForestClassifier(**MODEL_PARAMS)

# ==========================================
# STEP 1: CROSS-VALIDATION (More Reliable Testing)
# ==========================================

# Cross-validation splits data into 5 parts (folds)
# Trains 5 times, each time using a different fold for testing
# This gives us a better estimate of how well the model works

logger.info("🔄 Running Cross-Validation (5-fold)...")
logger.info("This tests the model 5 times with different data splits")

cv_scores = cross_val_score(
    model,  # Our Random Forest model
    X,  # All input features
    y_encoded,  # All labels
    cv=CV_FOLDS,  # Split into 5 folds
    scoring='accuracy'  # Measure accuracy
)

logger.info("📊 Cross-Validation Scores (5 folds):")
for i, score in enumerate(cv_scores, 1):
    logger.info(f"   Fold {i}: {score * 100:.2f}%")

logger.info(f"📈 Average Accuracy: {cv_scores.mean() * 100:.2f}%")
logger.info(f"📉 Standard Deviation: {cv_scores.std() * 100:.2f}%")
logger.info("   (Lower std = more consistent model)")


# ==========================================
# STEP 2: TRAIN ON FULL TRAINING SET
# ==========================================

# Now train the model on training data for final evaluation
logger.info("Training model on full training set...")
model.fit(X_train, y_train)


# ==========================================
# STEP 3: DETAILED EVALUATION ON TEST SET
# ==========================================

# Make predictions on test data (data model hasn't seen before)
y_pred = model.predict(X_test)

# Calculate accuracy - what % of predictions are correct
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Show detailed metrics for each class (Low, Medium, Viral)
logger.info("📊 Classification Report:")
report = classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_
)
logger.info(f"\n{report}")

# Show confusion matrix - which classes are confused with each other
logger.info("🔀 Confusion Matrix:")
logger.info("Rows = Actual, Columns = Predicted")
logger.info(f"Classes: {label_encoder.classes_}")
cm = confusion_matrix(y_test, y_pred)
logger.info(f"\n{cm}")

# Show which features are most important for predictions
logger.info("⭐ Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
logger.info(f"\n{feature_importance}")

# ==========================================
# INTERPRETATION GUIDE
# ==========================================

logger.info("\n" + "="*50)
logger.info("📚 WHAT DO THESE METRICS MEAN?")
logger.info("="*50)

logger.info("\n1️⃣ CROSS-VALIDATION:")
logger.info("   - Tests model on different data splits")
logger.info("   - More reliable than single train/test split")
logger.info("   - Look for: high average, low std deviation")

logger.info("\n2️⃣ CLASSIFICATION REPORT:")
logger.info("   - Precision: When model says 'Viral', how often is it right?")
logger.info("   - Recall: Of all actual 'Viral' videos, how many did we catch?")
logger.info("   - F1-score: Balance of precision and recall")

logger.info("\n3️⃣ CONFUSION MATRIX:")
logger.info("   - Diagonal = correct predictions")
logger.info("   - Off-diagonal = mistakes")

logger.info("\n4️⃣ FEATURE IMPORTANCE:")
logger.info("   - Which features the model relies on most")
logger.info("   - Higher = more influential in predictions")

logger.info("\n✅ Model training and evaluation complete!")

# ==========================================
# STEP 4: SAVE MODEL AND ENCODER
# ==========================================

logger.info("💾 Saving model and label encoder...")

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Save the trained model
model_path = os.path.join(MODEL_DIR, MODEL_FILE)
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Save the label encoder to decode predictions later
encoder_path = os.path.join(MODEL_DIR, LABEL_ENCODER_FILE)
with open(encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)

# Save feature names for prediction script
features_path = os.path.join(MODEL_DIR, FEATURE_NAMES_FILE)
with open(features_path, "wb") as f:
    pickle.dump(FEATURE_COLUMNS, f)

logger.info(f"✅ Model saved to {model_path}")
logger.info(f"✅ Label encoder saved to {encoder_path}")
logger.info(f"✅ Feature names saved to {features_path}")
print("✅ Feature names saved to models/feature_names.pkl")
print("\n🎉 Ready for predictions!")
