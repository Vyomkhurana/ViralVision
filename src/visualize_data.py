"""
Data Visualization Dashboard
Generate insights and visualizations from the dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("📊 VIRALVISION DATA VISUALIZATION DASHBOARD")
print("="*70)

# Load data
print("\n📁 Loading data...")
try:
    df = pd.read_csv("data/processed/labeled_videos.csv")
    print(f"✅ Loaded {len(df)} videos")
except FileNotFoundError:
    print("❌ Error: labeled_videos.csv not found!")
    print("Please run the full pipeline first.")
    exit(1)

# Create visualizations directory
os.makedirs("visualizations", exist_ok=True)


# ==========================================
# 1. VIRALITY DISTRIBUTION
# ==========================================

print("\n📊 Generating visualizations...")
print("\n1️⃣ Virality Distribution...")

plt.figure(figsize=(10, 6))
virality_counts = df['virality_label'].value_counts()
colors = ['#764ba2', '#f5576c', '#00f2fe']

plt.subplot(1, 2, 1)
virality_counts.plot(kind='bar', color=colors)
plt.title('Video Distribution by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
plt.pie(virality_counts.values, labels=virality_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90)
plt.title('Category Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/1_virality_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/1_virality_distribution.png")
plt.close()


# ==========================================
# 2. VIEW COUNT DISTRIBUTION
# ==========================================

print("2️⃣ View Count Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
df.boxplot(column='view_count', by='virality_label', ax=axes[0])
axes[0].set_title('View Count by Category', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('View Count')
plt.sca(axes[0])
plt.xticks(rotation=0)

# Histogram
for label, color in zip(['Low', 'Medium', 'Viral'], colors):
    data = df[df['virality_label'] == label]['view_count']
    axes[1].hist(data, alpha=0.6, label=label, bins=30, color=color)

axes[1].set_title('View Count Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('View Count')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('visualizations/2_view_count_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/2_view_count_analysis.png")
plt.close()


# ==========================================
# 3. ENGAGEMENT METRICS
# ==========================================

print("3️⃣ Engagement Metrics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Like ratio
df.boxplot(column='like_ratio', by='virality_label', ax=axes[0, 0])
axes[0, 0].set_title('Like Ratio by Category', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Category')
axes[0, 0].set_ylabel('Like Ratio')

# Comment ratio
df.boxplot(column='comment_ratio', by='virality_label', ax=axes[0, 1])
axes[0, 1].set_title('Comment Ratio by Category', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Comment Ratio')

# Scatter: views vs likes
for label, color in zip(['Low', 'Medium', 'Viral'], colors):
    data = df[df['virality_label'] == label]
    axes[1, 0].scatter(data['view_count'], data['like_count'], 
                      alpha=0.5, label=label, color=color, s=20)

axes[1, 0].set_title('Views vs Likes', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('View Count')
axes[1, 0].set_ylabel('Like Count')
axes[1, 0].legend()
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')

# Scatter: views vs comments
for label, color in zip(['Low', 'Medium', 'Viral'], colors):
    data = df[df['virality_label'] == label]
    axes[1, 1].scatter(data['view_count'], data['comment_count'], 
                      alpha=0.5, label=label, color=color, s=20)

axes[1, 1].set_title('Views vs Comments', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('View Count')
axes[1, 1].set_ylabel('Comment Count')
axes[1, 1].legend()
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('visualizations/3_engagement_metrics.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/3_engagement_metrics.png")
plt.close()


# ==========================================
# 4. TITLE FEATURES
# ==========================================

print("4️⃣ Title Features Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Title length
df.boxplot(column='title_length', by='virality_label', ax=axes[0, 0])
axes[0, 0].set_title('Title Length by Category', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Category')
axes[0, 0].set_ylabel('Character Count')

# Word count
df.boxplot(column='title_word_count', by='virality_label', ax=axes[0, 1])
axes[0, 1].set_title('Title Word Count by Category', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Word Count')

# Uppercase ratio
df.boxplot(column='title_uppercase_ratio', by='virality_label', ax=axes[1, 0])
axes[1, 0].set_title('Title Uppercase Ratio by Category', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Uppercase Ratio')

# Special characters
question_counts = df.groupby('virality_label')['title_has_question'].sum()
exclamation_counts = df.groupby('virality_label')['title_has_exclamation'].sum()

x = np.arange(len(question_counts))
width = 0.35

axes[1, 1].bar(x - width/2, question_counts.values, width, label='Has Question', color='#4facfe')
axes[1, 1].bar(x + width/2, exclamation_counts.values, width, label='Has Exclamation', color='#f5576c')
axes[1, 1].set_xlabel('Category')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Special Characters in Titles', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(question_counts.index)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('visualizations/4_title_features.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/4_title_features.png")
plt.close()


# ==========================================
# 5. TIME-BASED ANALYSIS
# ==========================================

print("5️⃣ Time-Based Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Day of week
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for label, color in zip(['Low', 'Medium', 'Viral'], colors):
    data = df[df['virality_label'] == label]
    day_counts = data['day_of_week'].value_counts().sort_index()
    axes[0, 0].plot(day_counts.index, day_counts.values, 
                   marker='o', label=label, color=color, linewidth=2)

axes[0, 0].set_xlabel('Day of Week')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Videos by Day of Week', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(range(7))
axes[0, 0].set_xticklabels(day_labels)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Hour of day
for label, color in zip(['Low', 'Medium', 'Viral'], colors):
    data = df[df['virality_label'] == label]
    hour_counts = data['hour_of_day'].value_counts().sort_index()
    axes[0, 1].plot(hour_counts.index, hour_counts.values, 
                   marker='o', label=label, color=color, linewidth=2, markersize=4)

axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Videos by Hour of Day', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Weekend vs Weekday
weekend_data = df.groupby(['is_weekend', 'virality_label']).size().unstack(fill_value=0)
weekend_data.plot(kind='bar', ax=axes[1, 0], color=colors)
axes[1, 0].set_title('Weekend vs Weekday Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Is Weekend (0=No, 1=Yes)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_xticklabels(['Weekday', 'Weekend'], rotation=0)
axes[1, 0].legend(title='Category')

# Best upload times heatmap
hour_day_virality = df[df['virality_label'] == 'Viral'].groupby(['day_of_week', 'hour_of_day']).size().unstack(fill_value=0)
im = axes[1, 1].imshow(hour_day_virality.T, cmap='YlOrRd', aspect='auto')
axes[1, 1].set_xlabel('Day of Week')
axes[1, 1].set_ylabel('Hour of Day')
axes[1, 1].set_title('Viral Videos Heatmap (Day × Hour)', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(range(7))
axes[1, 1].set_xticklabels(day_labels)
plt.colorbar(im, ax=axes[1, 1], label='Count')

plt.tight_layout()
plt.savefig('visualizations/5_time_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/5_time_analysis.png")
plt.close()


# ==========================================
# 6. FEATURE CORRELATIONS
# ==========================================

print("6️⃣ Feature Correlation Matrix...")

# Select numeric features
numeric_features = [
    'view_count', 'like_count', 'comment_count',
    'title_length', 'title_word_count', 'tag_count',
    'like_ratio', 'comment_ratio', 'title_uppercase_ratio'
]

correlation_matrix = df[numeric_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visualizations/6_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/6_correlation_matrix.png")
plt.close()


# ==========================================
# 7. SUMMARY STATISTICS
# ==========================================

print("7️⃣ Generating Summary Statistics...")

summary_stats = df.groupby('virality_label').agg({
    'view_count': ['mean', 'median', 'std', 'min', 'max'],
    'like_ratio': ['mean', 'median'],
    'comment_ratio': ['mean', 'median'],
    'title_length': ['mean', 'median'],
    'title_word_count': ['mean', 'median'],
    'tag_count': ['mean', 'median']
}).round(2)

print("\n📊 Summary Statistics by Category:")
print(summary_stats)

# Save to CSV
summary_stats.to_csv('visualizations/summary_statistics.csv')
print("\n✅ Saved: visualizations/summary_statistics.csv")


# ==========================================
# INSIGHTS REPORT
# ==========================================

print("\n" + "="*70)
print("📈 KEY INSIGHTS")
print("="*70)

# Calculate insights
viral_avg_views = df[df['virality_label'] == 'Viral']['view_count'].mean()
medium_avg_views = df[df['virality_label'] == 'Medium']['view_count'].mean()
low_avg_views = df[df['virality_label'] == 'Low']['view_count'].mean()

viral_like_ratio = df[df['virality_label'] == 'Viral']['like_ratio'].mean()
viral_comment_ratio = df[df['virality_label'] == 'Viral']['comment_ratio'].mean()

best_day = df[df['virality_label'] == 'Viral']['day_of_week'].mode()[0]
best_hour = df[df['virality_label'] == 'Viral']['hour_of_day'].mode()[0]

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

insights = f"""
1. 📊 VIRALITY METRICS:
   - Viral videos: {viral_avg_views:,.0f} avg views
   - Medium videos: {medium_avg_views:,.0f} avg views
   - Low videos: {low_avg_views:,.0f} avg views

2. 💬 ENGAGEMENT PATTERNS:
   - Viral videos have {viral_like_ratio:.4f} like ratio
   - Viral videos have {viral_comment_ratio:.4f} comment ratio

3. ⏰ BEST UPLOAD TIME:
   - Most viral day: {day_names[best_day]}
   - Most viral hour: {best_hour:02d}:00

4. 📝 TITLE CHARACTERISTICS:
   - Viral avg title length: {df[df['virality_label']=='Viral']['title_length'].mean():.0f} chars
   - Viral avg word count: {df[df['virality_label']=='Viral']['title_word_count'].mean():.1f} words

5. 🎯 RECOMMENDATIONS:
   - Upload on {day_names[best_day]} around {best_hour:02d}:00
   - Target like ratio above {viral_like_ratio:.4f}
   - Keep title around {df[df['virality_label']=='Viral']['title_length'].mean():.0f} characters
   - Use {df[df['virality_label']=='Viral']['tag_count'].mean():.0f} tags on average
"""

print(insights)

# Save insights to file
with open('visualizations/insights_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("VIRALVISION - DATA INSIGHTS REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*70 + "\n")
    f.write(insights)

print("✅ Saved: visualizations/insights_report.txt")

print("\n" + "="*70)
print("🎉 VISUALIZATION DASHBOARD COMPLETE!")
print("="*70)
print("\n📁 All visualizations saved to: visualizations/")
print("\nGenerated files:")
print("  1. 1_virality_distribution.png")
print("  2. 2_view_count_analysis.png")
print("  3. 3_engagement_metrics.png")
print("  4. 4_title_features.png")
print("  5. 5_time_analysis.png")
print("  6. 6_correlation_matrix.png")
print("  7. summary_statistics.csv")
print("  8. insights_report.txt")
