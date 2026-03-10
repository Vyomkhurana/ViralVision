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
import hashlib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import load_model_artifacts, extract_features, predict_virality
from model_report import ModelReportGenerator

# Simple user database (in production, use proper database)
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "demo": hashlib.sha256("demo123".encode()).hexdigest(),
    "user": hashlib.sha256("password".encode()).hexdigest()
}

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state["username"]
        password = st.session_state["password"]
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if username in USERS and USERS[username] == password_hash:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = username
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show login form
        st.markdown('<h1 class="main-header">🎬 ViralVision Login</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Please login to continue</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Login", on_click=password_entered)
            
            with st.expander("👤 Demo Credentials"):
                st.info("**Username:** demo\n\n**Password:** demo123")
        return False
    elif not st.session_state["password_correct"]:
        # Login failed
        st.markdown('<h1 class="main-header">🎬 ViralVision Login</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Please login to continue</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Login", on_click=password_entered)
            st.error("😕 Username or password incorrect")
            
            with st.expander("👤 Demo Credentials"):
                st.info("**Username:** demo\n\n**Password:** demo123")
        return False
    else:
        # Login successful
        return True


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
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction", "📈 Analytics", "📋 Model Reports"])
    
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
    
    # TAB 4: Model Reports
    with tab4:
        st.header("📋 Model Performance Reports")
        
        st.info("""
        Generate comprehensive performance reports for the trained model:
        - **JSON Report**: Machine-readable metrics
        - **Text Report**: Human-readable summary
        - **Confusion Matrix**: Visual performance breakdown
        - **Feature Importance**: Top contributing features
        """)
        
        # Check if we have test data
        test_data_path = "data/processed/labeled_videos.csv"
        if os.path.exists(test_data_path):
            df = pd.read_csv(test_data_path)
            st.success(f"✅ Found {len(df)} videos for evaluation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_name = st.text_input(
                    "📝 Report Name",
                    value="ViralVision Model",
                    help="Name for your model report"
                )
            
            with col2:
                test_size = st.slider(
                    "🎲 Test Sample Size (%)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    help="Percentage of data to use for evaluation"
                )
            
            if st.button("📊 Generate Full Report", type="primary", use_container_width=True):
                with st.spinner("🔮 Generating comprehensive report..."):
                    try:
                        # Load model
                        model, label_encoder, feature_names = load_model_artifacts()
                        
                        # Sample test data
                        test_df = df.sample(frac=test_size/100, random_state=42)
                        
                        # Extract features
                        features_list = []
                        for _, row in test_df.iterrows():
                            features = extract_features(row.to_dict())
                            features_list.append(features)
                        
                        X = pd.DataFrame(features_list)[feature_names]
                        y_true = label_encoder.transform(test_df["virality_label"])
                        
                        # Make predictions
                        y_pred = model.predict(X)
                        y_prob = model.predict_proba(X)
                        
                        # Get feature importance if available
                        feature_importance = None
                        if hasattr(model, 'feature_importances_'):
                            feature_importance = dict(zip(feature_names, model.feature_importances_))
                        
                        # Generate report
                        generator = ModelReportGenerator(report_name)
                        paths = generator.generate_full_report(
                            y_true=y_true,
                            y_pred=y_pred,
                            class_names=label_encoder.classes_.tolist(),
                            y_prob=y_prob,
                            feature_importance=feature_importance,
                            additional_info={
                                "test_samples": len(test_df),
                                "feature_count": len(feature_names),
                                "model_type": type(model).__name__
                            }
                        )
                        
                        st.success("✅ Report generated successfully!")
                        
                        # Display metrics
                        st.subheader("📊 Overall Performance")
                        metrics = generator.report_data['overall_metrics']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                        with col3:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                        with col4:
                            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                        
                        # Display confusion matrix
                        st.subheader("🎯 Confusion Matrix")
                        if paths.get('confusion_matrix') and os.path.exists(paths['confusion_matrix']):
                            st.image(paths['confusion_matrix'], use_container_width=True)
                        
                        # Display feature importance
                        if paths.get('feature_importance') and os.path.exists(paths['feature_importance']):
                            st.subheader("⭐ Feature Importance")
                            st.image(paths['feature_importance'], use_container_width=True)
                        
                        # Download buttons
                        st.subheader("💾 Download Reports")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if paths.get('text') and os.path.exists(paths['text']):
                                with open(paths['text'], 'r') as f:
                                    text_content = f.read()
                                st.download_button(
                                    "📄 Download Text Report",
                                    text_content,
                                    file_name="model_report.txt",
                                    mime="text/plain"
                                )
                        
                        with col2:
                            if paths.get('json') and os.path.exists(paths['json']):
                                with open(paths['json'], 'r') as f:
                                    json_content = f.read()
                                st.download_button(
                                    "📊 Download JSON Report",
                                    json_content,
                                    file_name="model_report.json",
                                    mime="application/json"
                                )
                        
                        # Show per-class metrics
                        with st.expander("📈 Per-Class Metrics"):
                            per_class = generator.report_data['per_class_metrics']
                            metrics_df = pd.DataFrame(per_class).T
                            st.dataframe(metrics_df.style.format("{:.3f}"))
                        
                        st.success("🎉 All reports saved to models/ directory")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
        else:
            st.warning("⚠️ No training data found.")
            st.info("📝 Please run the data collection and model training pipeline first.")
            
            with st.expander("💡 How to generate training data"):
                st.markdown("""
                1. Run `python src/data_collection.py` to collect YouTube data
                2. Run `python src/preprocessing.py` to process the data
                3. Run `python src/labeling.py` to label videos
                4. Run `python src/model_training.py` to train the model
                5. Return here to generate reports
                """)
    
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
