"""
Prediction script for ViralVision
Loads trained model and predicts virality for new video data
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime


def load_model_artifacts():
    """Load trained model, label encoder, and feature names"""
    try:
        with open("models/virality_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        with open("models/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        
        return model, label_encoder, feature_names
    except FileNotFoundError as e:
        print("❌ Error: Model files not found!")
        print("Please train the model first by running: python src/model_training.py")
        raise e


def extract_features(video_data):
    """
    Extract features from raw video data
    
    Parameters:
    -----------
    video_data : dict
        Dictionary containing video metadata with keys:
        - title (str): Video title
        - description (str): Video description
        - tags (str or list): Tags (pipe-separated string or list)
        - view_count (int): Number of views
        - like_count (int): Number of likes
        - comment_count (int): Number of comments
        - published_at (str): ISO format datetime string
    
    Returns:
    --------
    dict : Extracted features ready for prediction
    """
    features = {}
    
    # Basic text features
    title = str(video_data.get("title", ""))
    description = str(video_data.get("description", ""))
    tags = video_data.get("tags", "")
    
    features["title_length"] = len(title)
    features["description_length"] = len(description)
    
    # Tag count
    if isinstance(tags, str):
        features["tag_count"] = len(tags.split("|")) if tags else 0
    elif isinstance(tags, list):
        features["tag_count"] = len(tags)
    else:
        features["tag_count"] = 0
    
    # Engagement metrics
    view_count = float(video_data.get("view_count", 1))
    like_count = float(video_data.get("like_count", 0))
    comment_count = float(video_data.get("comment_count", 0))
    
    features["like_ratio"] = like_count / view_count if view_count > 0 else 0
    features["comment_ratio"] = comment_count / view_count if view_count > 0 else 0
    
    # Title analysis features
    features["title_word_count"] = len(title.split())
    
    # Uppercase ratio
    letters = [c for c in title if c.isalpha()]
    if letters:
        features["title_uppercase_ratio"] = sum(1 for c in letters if c.isupper()) / len(letters)
    else:
        features["title_uppercase_ratio"] = 0
    
    features["title_has_question"] = 1 if "?" in title else 0
    features["title_has_exclamation"] = 1 if "!" in title else 0
    
    # Time-based features
    try:
        published_dt = pd.to_datetime(video_data.get("published_at"))
        features["day_of_week"] = published_dt.dayofweek
        features["hour_of_day"] = published_dt.hour
        features["is_weekend"] = 1 if published_dt.dayofweek >= 5 else 0
    except Exception as e:
        # fallback to current time if date parsing fails - better than hardcoded defaults
        print(f"⚠️  Warning: Could not parse date, using current time: {e}")
        now = datetime.now()
        features["day_of_week"] = now.weekday()
        features["hour_of_day"] = now.hour
        features["is_weekend"] = 1 if now.weekday() >= 5 else 0
    
    return features


def predict_virality(video_data):
    """
    Predict virality label for a single video
    
    Parameters:
    -----------
    video_data : dict
        Video metadata dictionary
    
    Returns:
    --------
    dict : Prediction results with label and probabilities
    """
    # Load model artifacts
    model, label_encoder, feature_names = load_model_artifacts()
    
    # Extract features
    features = extract_features(video_data)
    
    # Create DataFrame with correct feature order
    X = pd.DataFrame([features])[feature_names]
    
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Decode prediction
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    # Create probability dictionary
    prob_dict = {
        label: prob 
        for label, prob in zip(label_encoder.classes_, probabilities)
    }
    
    return {
        "predicted_label": predicted_label,
        "probabilities": prob_dict,
        "confidence": max(probabilities) * 100
    }


def predict_batch(csv_path, output_path=None):
    """
    Predict virality for multiple videos from a CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with video data
    output_path : str, optional
        Path to save predictions CSV
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    print(f"📊 Loading {len(df)} videos from {csv_path}")
    
    # Load model artifacts
    model, label_encoder, feature_names = load_model_artifacts()
    
    # Extract features for all videos
    print("🔧 Extracting features...")
    features_list = []
    for _, row in df.iterrows():
        features = extract_features(row.to_dict())
        features_list.append(features)
    
    # Create feature DataFrame
    X = pd.DataFrame(features_list)[feature_names]
    
    # Make predictions
    print("🎯 Making predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Add predictions to dataframe
    df["predicted_label"] = label_encoder.inverse_transform(predictions)
    df["confidence"] = probabilities.max(axis=1) * 100
    
    # Add individual class probabilities
    for i, class_name in enumerate(label_encoder.classes_):
        df[f"prob_{class_name}"] = probabilities[:, i]
    
    # Display results summary
    print("\n✅ Predictions complete!")
    print("\n📈 Prediction Distribution:")
    print(df["predicted_label"].value_counts())
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n💾 Predictions saved to: {output_path}")
    
    return df


# Example usage
if __name__ == "__main__":
    # Example 1: Predict single video
    print("="*60)
    print("EXAMPLE 1: Single Video Prediction")
    print("="*60)
    
    sample_video = {
        "title": "INSANE Python Tutorial - YOU WON'T BELIEVE THIS!",
        "description": "Learn Python in 10 minutes with this amazing tutorial",
        "tags": "python|tutorial|programming|coding",
        "view_count": 50000,
        "like_count": 2500,
        "comment_count": 300,
        "published_at": "2026-01-05T14:30:00Z"
    }
    
    try:
        result = predict_virality(sample_video)
        
        print(f"\n🎬 Video: {sample_video['title'][:50]}...")
        print(f"\n🎯 Prediction: {result['predicted_label']}")
        print(f"📊 Confidence: {result['confidence']:.1f}%")
        print("\n📈 Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"   {label}: {prob*100:.1f}%")
    except FileNotFoundError:
        print("\n⚠️  Models not found. Train the model first!")
        print("Run: python src/model_training.py")
    
    # Example 2: Batch prediction
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Batch Prediction")
    print("="*60)
    print("\nTo predict for multiple videos:")
    print("python src/predict.py --batch data/raw/new_videos.csv")
