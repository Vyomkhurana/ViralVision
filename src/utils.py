"""
Utility functions for ViralVision project
Error handling, validation, and helper functions
"""

import os
import pandas as pd
from datetime import datetime


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_video_data(video_data):
    """
    Validate video data dictionary
    
    Parameters:
    -----------
    video_data : dict
        Video metadata dictionary
    
    Raises:
    -------
    ValidationError : If data is invalid
    """
    required_fields = ["title", "view_count"]
    
    # Check required fields
    for field in required_fields:
        if field not in video_data:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate title
    if not video_data.get("title") or len(str(video_data["title"]).strip()) == 0:
        raise ValidationError("Title cannot be empty")
    
    # Validate numeric fields
    numeric_fields = ["view_count", "like_count", "comment_count"]
    for field in numeric_fields:
        if field in video_data:
            try:
                value = float(video_data[field])
                if value < 0:
                    raise ValidationError(f"{field} cannot be negative")
            except (ValueError, TypeError):
                raise ValidationError(f"{field} must be a number")
    
    return True


def validate_csv_file(csv_path):
    """
    Validate CSV file for batch prediction
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
    
    Returns:
    --------
    pd.DataFrame : Validated dataframe
    
    Raises:
    -------
    ValidationError : If file or data is invalid
    """
    # Check file exists
    if not os.path.exists(csv_path):
        raise ValidationError(f"File not found: {csv_path}")
    
    # Check file extension
    if not csv_path.endswith('.csv'):
        raise ValidationError("File must be a CSV file")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValidationError(f"Error reading CSV file: {str(e)}")
    
    # Check not empty
    if len(df) == 0:
        raise ValidationError("CSV file is empty")
    
    # Check required columns
    required_columns = ["title"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns: {', '.join(missing_columns)}")
    
    return df


def safe_get_numeric(data, key, default=0):
    """
    Safely get numeric value from dictionary
    
    Parameters:
    -----------
    data : dict
        Data dictionary
    key : str
        Key to retrieve
    default : float
        Default value if key not found or invalid
    
    Returns:
    --------
    float : Numeric value
    """
    try:
        value = data.get(key, default)
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def format_number(num):
    """
    Format large numbers for display (e.g., 1.2M, 45.3K)
    
    Parameters:
    -----------
    num : int or float
        Number to format
    
    Returns:
    --------
    str : Formatted number string
    """
    try:
        num = float(num)
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return str(int(num))
    except (ValueError, TypeError):
        return "0"


def calculate_engagement_score(views, likes, comments):
    """
    Calculate overall engagement score
    
    Parameters:
    -----------
    views : int
        View count
    likes : int
        Like count
    comments : int
        Comment count
    
    Returns:
    --------
    float : Engagement score (0-100)
    """
    if views == 0:
        return 0.0
    
    like_ratio = likes / views
    comment_ratio = comments / views
    
    # Weighted engagement score
    engagement = (like_ratio * 70 + comment_ratio * 30) * 100
    
    return min(engagement, 100.0)  # Cap at 100


def get_optimal_upload_time():
    """
    Get recommendations for optimal upload time
    
    Returns:
    --------
    dict : Recommendations
    """
    return {
        "best_days": ["Thursday", "Friday", "Saturday"],
        "best_hours": [14, 15, 16, 17, 18],  # 2 PM - 6 PM
        "avoid_days": ["Monday"],
        "reasoning": "Weekends and Thursday-Friday evenings have highest engagement"
    }


def analyze_title_quality(title):
    """
    Analyze title quality and provide recommendations
    
    Parameters:
    -----------
    title : str
        Video title
    
    Returns:
    --------
    dict : Analysis results with score and recommendations
    """
    title = str(title)
    recommendations = []
    score = 0
    
    # Length check (optimal: 30-60 characters)
    length = len(title)
    if 30 <= length <= 60:
        score += 25
    elif length < 30:
        recommendations.append("📏 Title too short - aim for 30-60 characters")
    else:
        recommendations.append("📏 Title too long - consider shortening to 30-60 characters")
    
    # Word count (optimal: 5-10 words)
    word_count = len(title.split())
    if 5 <= word_count <= 10:
        score += 25
    elif word_count < 5:
        recommendations.append("📝 Too few words - aim for 5-10 words")
    else:
        recommendations.append("📝 Too many words - aim for 5-10 words")
    
    # Uppercase check (avoid all caps)
    uppercase_ratio = sum(1 for c in title if c.isupper()) / len(title) if title else 0
    if uppercase_ratio < 0.5:
        score += 25
    else:
        recommendations.append("🔤 Too many CAPS - avoid excessive capitalization")
    
    # Engagement elements
    has_question = "?" in title
    has_exclamation = "!" in title
    has_numbers = any(c.isdigit() for c in title)
    
    engagement_score = sum([has_question, has_exclamation, has_numbers])
    if engagement_score >= 1:
        score += 25
    else:
        recommendations.append("❓ Add engaging elements: numbers, questions, or exclamations")
    
    # Quality rating
    if score >= 75:
        quality = "Excellent"
        emoji = "🌟"
    elif score >= 50:
        quality = "Good"
        emoji = "👍"
    elif score >= 25:
        quality = "Fair"
        emoji = "⚠️"
    else:
        quality = "Poor"
        emoji = "❌"
    
    return {
        "score": score,
        "quality": quality,
        "emoji": emoji,
        "recommendations": recommendations if recommendations else ["✅ Title looks great!"]
    }


def check_model_files():
    """
    Check if all required model files exist
    
    Returns:
    --------
    dict : Status of model files
    """
    files = {
        "model": "models/virality_model.pkl",
        "encoder": "models/label_encoder.pkl",
        "features": "models/feature_names.pkl"
    }
    
    status = {}
    all_exist = True
    
    for name, path in files.items():
        exists = os.path.exists(path)
        status[name] = {
            "path": path,
            "exists": exists
        }
        if not exists:
            all_exist = False
    
    status["all_ready"] = all_exist
    return status


def create_feature_explanation():
    """
    Get explanations for all features used in the model
    
    Returns:
    --------
    dict : Feature names and their explanations
    """
    return {
        "title_length": "Number of characters in video title",
        "description_length": "Number of characters in video description",
        "tag_count": "Number of tags associated with video",
        "like_ratio": "Ratio of likes to views (engagement indicator)",
        "comment_ratio": "Ratio of comments to views (engagement indicator)",
        "title_word_count": "Number of words in title (optimal: 5-10)",
        "title_uppercase_ratio": "Proportion of uppercase letters (avoid > 0.5)",
        "title_has_question": "Whether title contains a question mark",
        "title_has_exclamation": "Whether title contains exclamation mark",
        "day_of_week": "Day of week published (0=Mon, 6=Sun)",
        "hour_of_day": "Hour of day published (0-23)",
        "is_weekend": "Whether published on weekend (Sat/Sun)"
    }


def print_success(message):
    """Print success message with formatting"""
    print(f"\n✅ {message}\n")


def print_error(message):
    """Print error message with formatting"""
    print(f"\n❌ Error: {message}\n")


def print_warning(message):
    """Print warning message with formatting"""
    print(f"\n⚠️  Warning: {message}\n")


def print_info(message):
    """Print info message with formatting"""
    print(f"\nℹ️  {message}\n")


# Example usage
if __name__ == "__main__":
    # Test validation
    print("="*60)
    print("TESTING UTILITY FUNCTIONS")
    print("="*60)
    
    # Test 1: Title quality analysis
    print("\n1️⃣ Title Quality Analysis:")
    test_titles = [
        "How to Learn Python in 10 Minutes!",
        "AMAZING TUTORIAL YOU MUST SEE!!!",
        "Python",
        "This is an extremely long title that probably goes on for way too many characters"
    ]
    
    for title in test_titles:
        result = analyze_title_quality(title)
        print(f"\nTitle: {title[:50]}...")
        print(f"{result['emoji']} Quality: {result['quality']} ({result['score']}/100)")
        for rec in result['recommendations']:
            print(f"  {rec}")
    
    # Test 2: Number formatting
    print("\n\n2️⃣ Number Formatting:")
    test_numbers = [42, 1_234, 45_678, 1_234_567, 9_876_543]
    for num in test_numbers:
        print(f"{num:>10,} → {format_number(num)}")
    
    # Test 3: Model file check
    print("\n\n3️⃣ Model File Status:")
    status = check_model_files()
    for name, info in status.items():
        if name != "all_ready":
            icon = "✅" if info["exists"] else "❌"
            print(f"{icon} {name}: {info['path']}")
    
    print(f"\n🎯 All Ready: {'Yes' if status['all_ready'] else 'No'}")
