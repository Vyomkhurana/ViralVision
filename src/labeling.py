import pandas as pd
df = pd.read_csv("data/processed/processed_videos.csv")
print(df["view_count"].describe())
low_threshold = df["view_count"].quantile(0.50)
viral_threshold = df["view_count"].quantile(0.80)


# labeling function

def label_virality(views):
    if views >= viral_threshold:
        return "Viral"
    elif views >= low_threshold:
        return "Medium"
    else:
        return "Low"

df["virality_label"] = df["view_count"].apply(label_virality)

print(df["virality_label"].value_counts())


df.to_csv("data/processed/labeled_videos.csv", index=False)
print("Saved labeled dataset")
