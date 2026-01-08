"""
Advanced Model Training with Hyperparameter Tuning
Improves model performance through GridSearchCV and feature selection
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import time


print("="*70)
print("🚀 ADVANCED MODEL TRAINING - HYPERPARAMETER TUNING")
print("="*70)

# Load data
print("\n📊 Loading labeled data...")
df = pd.read_csv("data/processed/labeled_videos.csv")
print(f"✅ Loaded {len(df)} videos")

# Select features
feature_columns = [
    "title_length",
    "description_length",
    "tag_count",
    "like_ratio",
    "comment_ratio",
    "title_word_count",
    "title_uppercase_ratio",
    "title_has_question",
    "title_has_exclamation",
    "day_of_week",
    "hour_of_day",
    "is_weekend",
]

X = df[feature_columns]
y = df["virality_label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"📊 Features: {len(X.columns)}")
print(f"📊 Samples: {len(X)}")
print(f"📊 Classes: {list(label_encoder.classes_)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✂️  Train set: {len(X_train)} samples")
print(f"✂️  Test set: {len(X_test)} samples")


# ==========================================
# STEP 1: FEATURE SELECTION
# ==========================================

print("\n" + "="*70)
print("📌 STEP 1: FEATURE SELECTION")
print("="*70)

# Select top K features
selector = SelectKBest(f_classif, k='all')
selector.fit(X_train, y_train)

# Get feature scores
feature_scores = pd.DataFrame({
    'feature': feature_columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("\n📊 Feature Importance (F-statistic):")
print(feature_scores.to_string(index=False))

# Select top features (you can adjust this threshold)
top_features = feature_scores.head(10)['feature'].tolist()
print(f"\n✅ Selected top {len(top_features)} features")


# ==========================================
# STEP 2: BASELINE MODEL
# ==========================================

print("\n" + "="*70)
print("📌 STEP 2: BASELINE MODEL (Default Parameters)")
print("="*70)

baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred, average='weighted')

print(f"\n📊 Baseline Accuracy: {baseline_accuracy * 100:.2f}%")
print(f"📊 Baseline F1-Score: {baseline_f1:.4f}")


# ==========================================
# STEP 3: HYPERPARAMETER TUNING - RANDOM FOREST
# ==========================================

print("\n" + "="*70)
print("📌 STEP 3: HYPERPARAMETER TUNING - RANDOM FOREST")
print("="*70)

print("\n🔍 Searching for best hyperparameters...")
print("This may take a few minutes...\n")

# Define parameter grid
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Simplified grid for faster execution (comment out to use full grid)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

# Grid search with cross-validation
start_time = time.time()

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search_rf.fit(X_train, y_train)

elapsed_time = time.time() - start_time

print(f"\n⏱️  Training completed in {elapsed_time:.1f} seconds")
print(f"\n🎯 Best Parameters: {grid_search_rf.best_params_}")
print(f"📊 Best CV Score: {grid_search_rf.best_score_:.4f}")

# Test the best model
best_rf_model = grid_search_rf.best_estimator_
rf_pred = best_rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')

print(f"\n🎯 Test Accuracy: {rf_accuracy * 100:.2f}%")
print(f"📊 Test F1-Score: {rf_f1:.4f}")


# ==========================================
# STEP 4: TRY GRADIENT BOOSTING
# ==========================================

print("\n" + "="*70)
print("📌 STEP 4: GRADIENT BOOSTING CLASSIFIER")
print("="*70)

print("\n🔍 Training Gradient Boosting model...\n")

param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
}

start_time = time.time()

grid_search_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gb,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search_gb.fit(X_train, y_train)

elapsed_time = time.time() - start_time

print(f"\n⏱️  Training completed in {elapsed_time:.1f} seconds")
print(f"\n🎯 Best Parameters: {grid_search_gb.best_params_}")
print(f"📊 Best CV Score: {grid_search_gb.best_score_:.4f}")

# Test
best_gb_model = grid_search_gb.best_estimator_
gb_pred = best_gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred, average='weighted')

print(f"\n🎯 Test Accuracy: {gb_accuracy * 100:.2f}%")
print(f"📊 Test F1-Score: {gb_f1:.4f}")


# ==========================================
# STEP 5: MODEL COMPARISON & SELECTION
# ==========================================

print("\n" + "="*70)
print("📌 STEP 5: MODEL COMPARISON")
print("="*70)

comparison = pd.DataFrame({
    'Model': ['Baseline RF', 'Tuned RF', 'Gradient Boosting'],
    'Accuracy': [baseline_accuracy, rf_accuracy, gb_accuracy],
    'F1-Score': [baseline_f1, rf_f1, gb_f1]
})

print("\n📊 Model Performance Comparison:")
print(comparison.to_string(index=False))

# Select best model
best_model_idx = comparison['F1-Score'].idxmax()
best_model_name = comparison.loc[best_model_idx, 'Model']

if best_model_name == 'Tuned RF':
    final_model = best_rf_model
elif best_model_name == 'Gradient Boosting':
    final_model = best_gb_model
else:
    final_model = baseline_model

print(f"\n🏆 Best Model: {best_model_name}")
print(f"📊 Accuracy: {comparison.loc[best_model_idx, 'Accuracy'] * 100:.2f}%")
print(f"📊 F1-Score: {comparison.loc[best_model_idx, 'F1-Score']:.4f}")


# ==========================================
# STEP 6: DETAILED EVALUATION
# ==========================================

print("\n" + "="*70)
print("📌 STEP 6: DETAILED EVALUATION")
print("="*70)

final_pred = final_model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, final_pred, target_names=label_encoder.classes_))

print("\n🔀 Confusion Matrix:")
print("Rows = Actual, Columns = Predicted")
cm = confusion_matrix(y_test, final_pred)
print(pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
))

# Feature importance
print("\n⭐ Top 10 Most Important Features:")
if hasattr(final_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    print(importance_df.to_string(index=False))


# ==========================================
# STEP 7: SAVE OPTIMIZED MODEL
# ==========================================

print("\n" + "="*70)
print("📌 STEP 7: SAVING OPTIMIZED MODEL")
print("="*70)

os.makedirs("models", exist_ok=True)

# Save models with versioning
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

# Save the best model
with open("models/virality_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

# Save backup with timestamp
with open(f"models/virality_model_{timestamp}.pkl", "wb") as f:
    pickle.dump(final_model, f)

# Save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'accuracy': float(comparison.loc[best_model_idx, 'Accuracy']),
    'f1_score': float(comparison.loc[best_model_idx, 'F1-Score']),
    'best_params': grid_search_rf.best_params_ if best_model_name == 'Tuned RF' else grid_search_gb.best_params_,
    'features': feature_columns,
    'trained_on': timestamp,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open("models/model_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("\n✅ Optimized model saved to models/virality_model.pkl")
print(f"✅ Backup saved to models/virality_model_{timestamp}.pkl")
print("✅ Label encoder saved to models/label_encoder.pkl")
print("✅ Feature names saved to models/feature_names.pkl")
print("✅ Metadata saved to models/model_metadata.pkl")

print("\n" + "="*70)
print("🎉 OPTIMIZATION COMPLETE!")
print("="*70)
print(f"\n🏆 Final Model: {best_model_name}")
print(f"📊 Accuracy: {comparison.loc[best_model_idx, 'Accuracy'] * 100:.2f}%")
print(f"📊 F1-Score: {comparison.loc[best_model_idx, 'F1-Score']:.4f}")
print(f"\n💡 Improvement over baseline: {(comparison.loc[best_model_idx, 'F1-Score'] - baseline_f1) * 100:.2f}%")
print("\n✨ Model is ready for predictions!")
