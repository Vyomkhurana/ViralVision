# importing from other modules#
import os 
import csv
from datetime import datetime

from googleapiclient.discovery import build
from dotenv import load_dotenv

#api key #
load_dotenv()
API_KEY=os.getenv("YOUTUBE_API_KEY")

#safety check if no api key found#
if API_KEY is None:
    raise ValueError("YOUTUBE_API_KEY not found. ")

#func for youtube client creation#

def get_youtube_client():
    youtube=build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )

    return youtube

#fetching trending videos#

def fetch_trending_videos(max_results=50, region_code="US"):
   
    youtube = get_youtube_client()

    request = youtube.videos().list(
        part="snippet,contentDetails,statistics", 
        chart="mostPopular",                       
        regionCode=region_code,                   
        maxResults=max_results                    
    )

    response = request.execute()

    videos_data = []  

    for item in response.get("items", []):
        video_id = item["id"]              
        snippet = item["snippet"]         
        stats = item.get("statistics", {}) 
        content_details = item.get("contentDetails", {})  

        video_info = {
            "video_id": video_id,
            "title": snippet.get("title", ""),                         # video title
            "description": snippet.get("description", ""),             # video description
            "channel_title": snippet.get("channelTitle", ""),          # channel name
            "published_at": snippet.get("publishedAt", ""),            # upload time (ISO format)
            "tags": "|".join(snippet.get("tags", [])),                 # tags list → single string separated by |
            "category_id": snippet.get("categoryId", ""),              # YouTube category ID

            "view_count": stats.get("viewCount", 0),
            "like_count": stats.get("likeCount", 0),
            "comment_count": stats.get("commentCount", 0),

            "duration": content_details.get("duration", ""),           # ISO 8601 duration like 'PT15M33S'
            "definition": content_details.get("definition", "")        # 'hd' or 'sd'
        }

        videos_data.append(video_info)

    return videos_data


#save data to csv#

def save_to_csv(videos_data, filename):
    
    if not videos_data:
        print("No data to save.")
        return  

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fieldnames = list(videos_data[0].keys())

    with open(filename, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()     
        writer.writerows(videos_data)  

    print(f"Saved {len(videos_data)} videos to {filename}")


#main execution block#


if __name__ == "__main__":
    print("Fetching trending videos...")

    data = fetch_trending_videos(max_results=50, region_code="US")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = f"data/raw/trending_videos_{timestamp}.csv"

    save_to_csv(data, output_file)
