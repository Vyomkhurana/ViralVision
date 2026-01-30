"""
Advanced Feature Engineering Module for ViralVision

This module contains advanced data science techniques for creating 
sophisticated features to improve virality prediction accuracy.

Features include:
- NLP-based text features (sentiment, readability, keyword analysis)
- Statistical transformations (log transforms, polynomial features)
- Engagement rate engineering
- Temporal pattern analysis
- Thumbnail and metadata quality indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import re
from datetime import datetime


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering class for YouTube video virality prediction.
    
    This class implements various data science techniques to extract
    meaningful patterns from video metadata.
    """
    
    def __init__(self):
        """Initialize the feature engineer with default parameters."""
        self.feature_names = []
        
    def engineer_nlp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create NLP-based features from text fields.
        
        Args:
            df: DataFrame with 'title' and 'description' columns
            
        Returns:
            DataFrame with new NLP features
        """
        print("\n🔤 Engineering NLP Features...")
        
        # Title complexity score
        df['title_avg_word_length'] = df['title'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Clickbait indicators
        clickbait_words = ['shocking', 'unbelievable', 'amazing', 'incredible', 
                          'you won\'t believe', 'must see', 'gone wrong']
        df['title_clickbait_score'] = df['title'].str.lower().apply(
            lambda x: sum(word in str(x) for word in clickbait_words)
        )
        
        # Number detection (years, rankings, lists)
        df['title_contains_number'] = df['title'].str.contains(r'\d+', regex=True).astype(int)
        df['title_number_count'] = df['title'].apply(
            lambda x: len(re.findall(r'\d+', str(x)))
        )
        
        # Special characters ratio
        df['title_special_char_ratio'] = df['title'].apply(
            lambda x: sum(not c.isalnum() and not c.isspace() for c in str(x)) / max(len(str(x)), 1)
        )
        
        # Description richness
        df['description_sentence_count'] = df['description'].apply(
            lambda x: len(re.split(r'[.!?]+', str(x))) if pd.notna(x) else 0
        )
        
        df['description_url_count'] = df['description'].apply(
            lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(x)))
        )
        
        # Emoji detection (common patterns)
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        
        df['title_emoji_count'] = df['title'].apply(
            lambda x: len(emoji_pattern.findall(str(x)))
        )
        
        print(f"   ✅ Created {8} NLP features")
        return df
    
    def engineer_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced engagement metrics.
        
        Args:
            df: DataFrame with engagement columns
            
        Returns:
            DataFrame with new engagement features
        """
        print("\n📊 Engineering Engagement Features...")
        
        # Engagement velocity (engagement per view)
        df['engagement_rate'] = (
            (df['like_count'] + df['comment_count']) / df['view_count'].clip(lower=1)
        ).fillna(0)
        
        # Like-to-comment ratio (indicates type of engagement)
        df['like_comment_balance'] = (
            df['like_count'] / df['comment_count'].clip(lower=1)
        ).fillna(0).clip(upper=100)  # Cap extreme values
        
        # Total engagement score (weighted)
        df['weighted_engagement'] = (
            df['like_count'] * 1.0 + 
            df['comment_count'] * 2.5  # Comments are more valuable
        )
        
        # Log transformations for skewed metrics
        df['log_view_count'] = np.log1p(df['view_count'])
        df['log_like_count'] = np.log1p(df['like_count'])
        df['log_comment_count'] = np.log1p(df['comment_count'])
        
        # Engagement percentiles (relative ranking)
        df['view_count_percentile'] = df['view_count'].rank(pct=True)
        df['engagement_percentile'] = df['engagement_rate'].rank(pct=True)
        
        print(f"   ✅ Created {9} engagement features")
        return df
    
    def engineer_temporal_features(self, df: pd.DataFrame, 
                                   datetime_col: str = 'published_at') -> pd.DataFrame:
        """
        Create time-based features with cyclic encoding.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of the datetime column
            
        Returns:
            DataFrame with temporal features
        """
        print("\n⏰ Engineering Temporal Features...")
        
        # Ensure datetime format
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        
        # Cyclic encoding for hour (preserves circular nature: 23h is close to 0h)
        df['hour_sin'] = np.sin(2 * np.pi * df[datetime_col].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df[datetime_col].dt.hour / 24)
        
        # Cyclic encoding for day of week
        df['day_sin'] = np.sin(2 * np.pi * df[datetime_col].dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df[datetime_col].dt.dayofweek / 7)
        
        # Month encoding (seasonal patterns)
        df['month_sin'] = np.sin(2 * np.pi * df[datetime_col].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[datetime_col].dt.month / 12)
        
        # Specific time patterns
        df['is_prime_time'] = df[datetime_col].dt.hour.isin([18, 19, 20, 21]).astype(int)
        df['is_morning'] = df[datetime_col].dt.hour.isin(range(6, 12)).astype(int)
        df['is_late_night'] = df[datetime_col].dt.hour.isin(range(0, 6)).astype(int)
        
        # Quarter of year
        df['quarter'] = df[datetime_col].dt.quarter
        
        print(f"   ✅ Created {10} temporal features")
        return df
    
    def engineer_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical transformations and interactions.
        
        Args:
            df: DataFrame with numeric features
            
        Returns:
            DataFrame with statistical features
        """
        print("\n📈 Engineering Statistical Features...")
        
        # Interaction features (combinations that might reveal patterns)
        df['title_desc_ratio'] = (
            df['title_length'] / df['description_length'].clip(lower=1)
        ).fillna(0).clip(upper=10)
        
        df['engagement_per_tag'] = (
            df['weighted_engagement'] / df['tag_count'].clip(lower=1)
        ).fillna(0)
        
        # Polynomial features (capturing non-linear relationships)
        df['title_length_squared'] = df['title_length'] ** 2
        df['like_ratio_squared'] = df['like_ratio'] ** 2
        
        # Z-score normalization for outlier detection
        if 'view_count' in df.columns:
            df['view_count_zscore'] = (
                (df['view_count'] - df['view_count'].mean()) / df['view_count'].std()
            ).fillna(0)
        
        # Binning continuous variables
        df['title_length_category'] = pd.cut(
            df['title_length'], 
            bins=[0, 30, 60, 90, float('inf')],
            labels=['short', 'medium', 'long', 'very_long']
        )
        
        # Convert categorical to numeric
        df['title_length_cat_encoded'] = df['title_length_category'].cat.codes
        
        print(f"   ✅ Created {7} statistical features")
        return df
    
    def engineer_quality_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that indicate content quality.
        
        Args:
            df: DataFrame with metadata
            
        Returns:
            DataFrame with quality indicators
        """
        print("\n⭐ Engineering Quality Indicators...")
        
        # Metadata completeness score
        df['metadata_completeness'] = (
            (df['title'].notna().astype(int) * 0.3) +
            (df['description'].notna().astype(int) * 0.3) +
            ((df['tag_count'] > 0).astype(int) * 0.4)
        )
        
        # Title optimization score (best practices)
        df['title_optimization_score'] = (
            ((df['title_length'] >= 40) & (df['title_length'] <= 70)).astype(int) * 0.3 +
            (df['title_word_count'].between(7, 12)).astype(int) * 0.3 +
            (df['title_contains_number'] == 1).astype(int) * 0.2 +
            (df['title_uppercase_ratio'] < 0.3).astype(int) * 0.2
        )
        
        # Tag optimization
        df['tag_optimization'] = (
            (df['tag_count'].between(5, 15)).astype(int)
        )
        
        # Description quality
        df['description_quality'] = (
            (df['description_length'] > 100).astype(int) * 0.4 +
            (df['description_sentence_count'] >= 3).astype(int) * 0.3 +
            (df['description_url_count'] >= 1).astype(int) * 0.3
        )
        
        print(f"   ✅ Created {4} quality indicators")
        return df
    
    def create_all_features(self, df: pd.DataFrame, 
                           datetime_col: str = 'published_at') -> pd.DataFrame:
        """
        Apply all feature engineering techniques.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "="*60)
        print("🚀 ADVANCED FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        original_features = len(df.columns)
        
        # Apply all engineering methods
        df = self.engineer_nlp_features(df)
        df = self.engineer_engagement_features(df)
        df = self.engineer_temporal_features(df, datetime_col)
        df = self.engineer_statistical_features(df)
        df = self.engineer_quality_indicators(df)
        
        new_features = len(df.columns) - original_features
        
        print("\n" + "="*60)
        print(f"✅ Feature Engineering Complete!")
        print(f"   Original features: {original_features}")
        print(f"   New features created: {new_features}")
        print(f"   Total features: {len(df.columns)}")
        print("="*60)
        
        return df
    
    def get_feature_importance_names(self, basic_features: bool = False) -> List[str]:
        """
        Get list of all engineered feature names for model training.
        
        Args:
            basic_features: If True, only return basic features
            
        Returns:
            List of feature names
        """
        if basic_features:
            return [
                'title_length', 'description_length', 'tag_count',
                'like_ratio', 'comment_ratio', 'title_word_count',
                'title_uppercase_ratio', 'title_has_question',
                'title_has_exclamation', 'day_of_week', 'hour_of_day',
                'is_weekend'
            ]
        
        # All advanced features
        return [
            # Basic
            'title_length', 'description_length', 'tag_count',
            'like_ratio', 'comment_ratio',
            # NLP
            'title_avg_word_length', 'title_clickbait_score',
            'title_contains_number', 'title_number_count',
            'title_special_char_ratio', 'description_sentence_count',
            'description_url_count', 'title_emoji_count',
            # Engagement
            'engagement_rate', 'like_comment_balance',
            'weighted_engagement', 'log_view_count',
            'log_like_count', 'log_comment_count',
            'view_count_percentile', 'engagement_percentile',
            # Temporal
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'is_prime_time',
            'is_morning', 'is_late_night', 'quarter',
            # Statistical
            'title_desc_ratio', 'engagement_per_tag',
            'title_length_squared', 'like_ratio_squared',
            'view_count_zscore', 'title_length_cat_encoded',
            # Quality
            'metadata_completeness', 'title_optimization_score',
            'tag_optimization', 'description_quality'
        ]


def apply_feature_engineering(input_path: str, output_path: str) -> None:
    """
    Main function to apply feature engineering to a dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save engineered dataset
    """
    print(f"\n📂 Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} videos")
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Apply all feature engineering
    df_engineered = engineer.create_all_features(df)
    
    # Save engineered dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_engineered.to_csv(output_path, index=False)
    print(f"\n💾 Saved engineered dataset to: {output_path}")
    print(f"   Shape: {df_engineered.shape}")


if __name__ == "__main__":
    # Example usage
    import os
    
    # Define paths
    input_file = "data/processed/labeled_videos.csv"
    output_file = "data/processed/featured_videos.csv"
    
    if os.path.exists(input_file):
        apply_feature_engineering(input_file, output_file)
        print("\n🎉 Feature engineering complete!")
    else:
        print(f"\n❌ Error: Input file not found: {input_file}")
        print("   Please run preprocessing and labeling first.")
