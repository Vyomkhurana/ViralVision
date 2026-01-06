# Import pandas to load CSV data
import pandas as pd

# Import LabelEncoder to convert text labels into numbers
from sklearn.preprocessing import LabelEncoder

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Import RandomForestClassifier as  ML model
from sklearn.ensemble import RandomForestClassifier

# Import accuracy_score to evaluate how good the model is
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/processed/labeled_videos.csv")

# Select input features (X)
X = df[
    [
        "title_length",
        "description_length",
        "tag_count",
        "like_ratio",
        "comment_ratio",
    ]
]

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

# Train the model on training data
model.fit(X_train, y_train)
