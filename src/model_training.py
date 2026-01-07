# Import pandas to load CSV data
import pandas as pd

# Import LabelEncoder to convert text labels into numbers
from sklearn.preprocessing import LabelEncoder

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split, cross_val_score

# Import RandomForestClassifier as  ML model
from sklearn.ensemble import RandomForestClassifier

# Import accuracy_score to evaluate how good the model is
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("data/processed/labeled_videos.csv")

# Select input features (X) - Now with 7 NEW features!
X = df[
    [
        # Original 5 features:
        "title_length",
        "description_length",
        "tag_count",
        "like_ratio",
        "comment_ratio",
        # New 7 features for better predictions:
        "title_word_count",
        "title_uppercase_ratio",
        "title_has_question",
        "title_has_exclamation",
        "day_of_week",
        "hour_of_day",
        "is_weekend",
    ]
]

print(f"\n📊 Training with {len(X.columns)} features:")
for i, feat in enumerate(X.columns, 1):
    print(f"   {i}. {feat}")
print()

# Select output label (y)
y = df["virality_label"]


# Create label encoder
label_encoder = LabelEncoder()

# Convert text labels into numbers
y_encoded = label_encoder.fit_transform(y)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

# ==========================================
# STEP 1: CROSS-VALIDATION (More Reliable Testing)
# ==========================================

# Cross-validation splits data into 5 parts (folds)
# Trains 5 times, each time using a different fold for testing
# This gives us a better estimate of how well the model works

print("\n🔄 Running Cross-Validation (5-fold)...")
print("This tests the model 5 times with different data splits\n")

cv_scores = cross_val_score(
    model,  # Our Random Forest model
    X,  # All input features
    y_encoded,  # All labels
    cv=5,  # Split into 5 folds
    scoring='accuracy'  # Measure accuracy
)

print(f"📊 Cross-Validation Scores (5 folds):")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i}: {score * 100:.2f}%")

print(f"\n📈 Average Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"📉 Standard Deviation: {cv_scores.std() * 100:.2f}%")
print("   (Lower std = more consistent model)")


# ==========================================
# STEP 2: TRAIN ON FULL TRAINING SET
# ==========================================

# Now train the model on training data for final evaluation
model.fit(X_train, y_train)


# ==========================================
# STEP 3: DETAILED EVALUATION ON TEST SET
# ==========================================

# Make predictions on test data (data model hasn't seen before)
y_pred = model.predict(X_test)

# Calculate accuracy - what % of predictions are correct
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Show detailed metrics for each class (Low, Medium, Viral)
print("\n📊 Classification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_
))

# Show confusion matrix - which classes are confused with each other
print("\n🔀 Confusion Matrix:")
print("Rows = Actual, Columns = Predicted")
print(f"Classes: {label_encoder.classes_}")
print(confusion_matrix(y_test, y_pred))

# Show which features are most important for predictions
print("\n⭐ Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# ==========================================
# INTERPRETATION GUIDE
# ==========================================

print("\n" + "="*50)
print("📚 WHAT DO THESE METRICS MEAN?")
print("="*50)

print("\n1️⃣ CROSS-VALIDATION:")
print("   - Tests model on different data splits")
print("   - More reliable than single train/test split")
print("   - Look for: high average, low std deviation")

print("\n2️⃣ CLASSIFICATION REPORT:")
print("   - Precision: When model says 'Viral', how often is it right?")
print("   - Recall: Of all actual 'Viral' videos, how many did we catch?")
print("   - F1-score: Balance of precision and recall")

print("\n3️⃣ CONFUSION MATRIX:")
print("   - Diagonal = correct predictions")
print("   - Off-diagonal = mistakes")

print("\n4️⃣ FEATURE IMPORTANCE:")
print("   - Which features the model relies on most")
print("   - Higher = more influential in predictions")

print("\n✅ Model training and evaluation complete!")
