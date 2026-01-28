"""
ViralVision - YouTube Video Virality Predictor
Interactive Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import load_model_artifacts, extract_features, predict_virality


# Page configuration
st.set_page_config(
    page_title="ViralVision - YouTube Virality Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF0000, #FF4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .viral-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .medium-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .low-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def check_models_exist():
    """Check if trained models are available"""
    required_files = [
        "models/virality_model.pkl",
        "models/label_encoder.pkl",
        "models/feature_names.pkl"
    ]
    return all(os.path.exists(f) for f in required_files)


def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 ViralVision</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict if your YouTube video will go VIRAL! 🚀</p>', 
                unsafe_allow_html=True)
    
    # Check if models exist
    if not check_models_exist():
        st.error("⚠️ **Model not found!**")
        st.info("""
        Please train the model first by running:
        ```
        python src/model_training.py
        ```
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.write("""
        ViralVision uses machine learning to predict whether your YouTube video 
        will be **Low**, **Medium**, or **Viral** based on its metadata.
        """)
        
        st.header("🎯 Features Analyzed")
        st.write("""
        - 📝 Title & Description Length
        - 🏷️ Tag Count
        - 💬 Engagement Metrics
        - ❓ Title Characteristics
        - 📅 Publishing Time
        """)
        
        st.header("📈 Model Info")
        try:
            model, label_encoder, feature_names = load_model_artifacts()
            st.success(f"✅ Model Loaded")
            st.info(f"Features: {len(feature_names)}")
        except:
            st.error("❌ Model Error")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction", "📈 Analytics"])
    
    # TAB 1: Single Prediction
    with tab1:
        st.header("Predict Single Video")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input(
                "📝 Video Title",
                placeholder="Enter your video title...",
                help="The title of your YouTube video"
            )
            
            description = st.text_area(
                "📄 Video Description",
                placeholder="Enter video description...",
                height=100,
                help="Full description of your video"
            )
            
            tags = st.text_input(
                "🏷️ Tags",
                placeholder="tag1|tag2|tag3",
                help="Tags separated by | (pipe)"
            )
        
        with col2:
            view_count = st.number_input(
                "👁️ View Count",
                min_value=0,
                value=1000,
                step=100,
                help="Current or expected view count"
            )
            
            like_count = st.number_input(
                "👍 Like Count",
                min_value=0,
                value=50,
                step=10
            )
            
            comment_count = st.number_input(
                "💬 Comment Count",
                min_value=0,
                value=10,
                step=5
            )
            
            published_at = st.date_input(
                "📅 Published Date",
                value=datetime.now()
            )
            
            published_time = st.time_input(
                "🕐 Published Time",
                value=datetime.now().time()
            )
        
        # Predict button
        if st.button("🚀 Predict Virality", type="primary", use_container_width=True):
            # basic validation before running the prediction
            if not title:
                st.warning("⚠️ Please enter a video title!")
            elif len(title.strip()) < 5:
                st.warning("⚠️ Title seems too short. Try a more descriptive title!")
            else:
                with st.spinner("🔮 Analyzing your video..."):
                    # Combine date and time
                    published_datetime = datetime.combine(published_at, published_time)
                    
                    # Create video data dict
                    video_data = {
                        "title": title,
                        "description": description,
                        "tags": tags,
                        "view_count": view_count,
                        "like_count": like_count,
                        "comment_count": comment_count,
                        "published_at": published_datetime.isoformat()
                    }
                    
                    # Make prediction
                    result = predict_virality(video_data)
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    # Prediction box with styling
                    pred_label = result["predicted_label"]
                    confidence = result["confidence"]
                    
                    box_class = {
                        "Viral": "viral-box",
                        "Medium": "medium-box",
                        "Low": "low-box"
                    }.get(pred_label, "low-box")
                    
                    st.markdown(
                        f'<div class="prediction-box {box_class}">'
                        f'<h2>🎯 Prediction: {pred_label}</h2>'
                        f'<h3>📊 Confidence: {confidence:.1f}%</h3>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Probability chart
                    st.subheader("📊 Probability Distribution")
                    
                    prob_df = pd.DataFrame([
                        {"Category": k, "Probability": v * 100}
                        for k, v in result["probabilities"].items()
                    ])
                    
                    fig = px.bar(
                        prob_df,
                        x="Category",
                        y="Probability",
                        color="Category",
                        color_discrete_map={
                            "Viral": "#764ba2",
                            "Medium": "#f5576c",
                            "Low": "#00f2fe"
                        },
                        text="Probability"
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(showlegend=False, yaxis_title="Probability (%)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature insights
                    with st.expander("🔍 Feature Insights"):
                        features = extract_features(video_data)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Title Length", features["title_length"])
                            st.metric("Word Count", features["title_word_count"])
                            st.metric("Tag Count", features["tag_count"])
                        
                        with col2:
                            st.metric("Like Ratio", f"{features['like_ratio']:.4f}")
                            st.metric("Comment Ratio", f"{features['comment_ratio']:.4f}")
                            st.metric("Uppercase Ratio", f"{features['title_uppercase_ratio']:.2f}")
                        
                        with col3:
                            st.metric("Has Question", "Yes" if features["title_has_question"] else "No")
                            st.metric("Has Exclamation", "Yes" if features["title_has_exclamation"] else "No")
                            st.metric("Weekend Upload", "Yes" if features["is_weekend"] else "No")
    
    # TAB 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction from CSV")
        
        st.info("""
        📁 Upload a CSV file containing multiple videos to get predictions for all of them.
        
        **Required columns:** title, description, tags, view_count, like_count, comment_count, published_at
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.write(f"📊 Loaded {len(df)} videos")
            st.dataframe(df.head())
            
            if st.button("🚀 Predict All", type="primary"):
                with st.spinner("🔮 Making predictions..."):
                    model, label_encoder, feature_names = load_model_artifacts()
                    
                    # Extract features
                    features_list = []
                    for _, row in df.iterrows():
                        features = extract_features(row.to_dict())
                        features_list.append(features)
                    
                    X = pd.DataFrame(features_list)[feature_names]
                    
                    # Predict
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)
                    
                    df["predicted_label"] = label_encoder.inverse_transform(predictions)
                    df["confidence"] = probabilities.max(axis=1) * 100
                    
                    st.success("✅ Predictions complete!")
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    counts = df["predicted_label"].value_counts()
                    
                    with col1:
                        st.metric("🚀 Viral", counts.get("Viral", 0))
                    with col2:
                        st.metric("📈 Medium", counts.get("Medium", 0))
                    with col3:
                        st.metric("📉 Low", counts.get("Low", 0))
                    
                    # Results table
                    st.subheader("📋 Results")
                    st.dataframe(df[["title", "predicted_label", "confidence"]].head(20))
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "💾 Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
    
    # TAB 3: Analytics
    with tab3:
        st.header("📈 Model Analytics")
        
        # Check if labeled data exists
        if os.path.exists("data/processed/labeled_videos.csv"):
            df = pd.read_csv("data/processed/labeled_videos.csv")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution chart
                st.subheader("📊 Dataset Distribution")
                dist = df["virality_label"].value_counts()
                fig = px.pie(
                    values=dist.values,
                    names=dist.index,
                    title="Video Categories",
                    color=dist.index,
                    color_discrete_map={
                        "Viral": "#764ba2",
                        "Medium": "#f5576c",
                        "Low": "#00f2fe"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # View count distribution
                st.subheader("👁️ View Count Distribution")
                fig = px.histogram(
                    df,
                    x="view_count",
                    color="virality_label",
                    title="Views by Category",
                    color_discrete_map={
                        "Viral": "#764ba2",
                        "Medium": "#f5576c",
                        "Low": "#00f2fe"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlations
            st.subheader("🔗 Feature Analysis")
            numeric_features = ["title_length", "title_word_count", "like_ratio", "comment_ratio"]
            if all(col in df.columns for col in numeric_features):
                fig = px.box(
                    df,
                    y=numeric_features,
                    color="virality_label",
                    title="Feature Distribution by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📁 No training data found. Run the full pipeline to see analytics.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "🎬 ViralVision | Powered by Machine Learning | "
        f"© {datetime.now().year}"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
