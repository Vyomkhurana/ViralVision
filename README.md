# 🎬 ViralVision 

**AI-Powered YouTube Video Virality Predictor**

ViralVision uses machine learning to predict whether a YouTube video will be **Low**, **Medium**, or **Viral** based on comprehensive metadata analysis including title characteristics, engagement metrics, and publishing patterns.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎯 **Accurate Predictions**: ML model with 12+ engineered features
- 📊 **Interactive Dashboard**: Beautiful Streamlit web interface
- 🔍 **Batch Processing**: Predict multiple videos from CSV
- 📈 **Data Visualizations**: Comprehensive analytics and insights
- ⚡ **Hyperparameter Tuning**: Optimized model performance
- 🛠️ **Production Ready**: Complete error handling and validation

## 🏗️ Tech Stack

- **Python 3.8+**
- **YouTube Data API v3** - Data collection
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Machine learning
- **Streamlit** - Web interface
- **Matplotlib & Seaborn** - Visualizations
- **Plotly** - Interactive charts

## 📂 Project Structure

```
ViralVision/
├── src/
│   ├── data_collection.py          # YouTube API data fetching
│   ├── preprocessing.py             # Feature engineering & cleaning
│   ├── labeling.py                  # Virality label assignment
│   ├── model_training.py            # Basic model training
│   ├── model_training_advanced.py   # Hyperparameter tuning
│   ├── predict.py                   # Prediction script
│   ├── visualize_data.py            # Data visualization dashboard
│   └── utils.py                     # Helper functions & validation
├── app/
│   └── streamlit_app.py             # Interactive web application
├── data/
│   ├── raw/                         # Raw YouTube data (CSV)
│   └── processed/                   # Processed & labeled datasets
├── models/                          # Trained ML models (generated)
├── visualizations/                  # Generated charts & insights
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### 1️⃣ Clone & Install

```bash
git clone <your-repo-url>
cd ViralVision
pip install -r requirements.txt
```

### 2️⃣ Set Up YouTube API

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_api_key_here
```

💡 💡 [Get your API key here](https://console.cloud.google.com/apis/credentials)

### 3️⃣ Run the Pipeline

```bash
# Step 1: Collect data from YouTube
python src/data_collection.py

# Step 2: Preprocess and engineer features
python src/preprocessing.py

# Step 3: Label videos by virality
python src/labeling.py

# Step 4: Train the model
python src/model_training.py

# Optional: Advanced training with hyperparameter tuning
python src/model_training_advanced.py

# Optional: Generate visualizations
python src/visualize_data.py
```

### 4️⃣ Launch Web App

```bash
streamlit run app/streamlit_app.py
```

Visit `http://localhost:8501` to use the interactive predictor!

## 🎯 Model Features

The model analyzes **12 key features**:

### 📝 Text Features
- **title_length** - Character count in title
- **description_length** - Character count in description
- **tag_count** - Number of tags
- **title_word_count** - Word count (optimal: 5-10)
- **title_uppercase_ratio** - Proportion of CAPS
- **title_has_question** - Contains "?"
- **title_has_exclamation** - Contains "!"

### 💬 Engagement Features
- **like_ratio** - Likes per view
- **comment_ratio** - Comments per view

### ⏰ Temporal Features
- **day_of_week** - Publishing day (0=Mon, 6=Sun)
- **hour_of_day** - Publishing hour (0-23)
- **is_weekend** - Weekend upload (1=Yes, 0=No)

## 📊 Usage Examples

### Single Video Prediction

```python
from src.predict import predict_virality

video_data = {
    "title": "Amazing Python Tutorial!",
    "description": "Learn Python in 10 minutes",
    "tags": "python|tutorial|programming",
    "view_count": 50000,
    "like_count": 2500,
    "comment_count": 300,
    "published_at": "2026-01-05T14:30:00Z"
}

result = predict_virality(video_data)
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.1f}%")
```

### Batch Prediction

```python
from src.predict import predict_batch

# Predict for CSV file
df = predict_batch(
    csv_path="data/raw/new_videos.csv",
    output_path="predictions.csv"
)
```

## 📈 Model Performance

After hyperparameter tuning:
- **Accuracy**: ~80-85% (varies by dataset)
- **F1-Score**: ~0.78-0.82
- **Cross-validation**: 5-fold CV for robust evaluation

## 🎨 Web Interface

The Streamlit app provides:
- ✅ **Single Video Predictor** - Instant predictions with probability breakdown
- ✅ **Batch Processor** - Upload CSV and predict multiple videos
- ✅ **Analytics Dashboard** - View dataset statistics and insights
- ✅ **Feature Inspector** - Understand what drives predictions

## 📊 Visualizations

Run `python src/visualize_data.py` to generate:
- Distribution charts (viral vs medium vs low)
- Engagement metric analysis
- Time-based patterns (best upload times)
- Feature correlation heatmaps
- Title characteristic breakdowns

## 🛠️ Advanced Features

### Hyperparameter Tuning
```bash
python src/model_training_advanced.py
```
Automatically tests multiple model configurations to find the best parameters.

### Title Quality Analysis
```python
from src.utils import analyze_title_quality

result = analyze_title_quality("Your Video Title Here!")
print(f"Quality: {result['quality']} ({result['score']}/100)")
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional features (thumbnail analysis, channel metrics)
- More sophisticated NLP for title/description analysis
- Deep learning models (LSTM, Transformers)
- Real-time API endpoint

## 📝 License

MIT License - feel free to use for your projects!

## 🙏 Acknowledgments

- YouTube Data API v3
- Scikit-learn community
- Streamlit team

---


For questions or feedback, open an issue on GitHub!
