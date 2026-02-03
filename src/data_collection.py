"""Data collection module for ViralVision.

Fetches trending videos from YouTube API and saves to CSV.
"""

import os 
import csv
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from googleapiclient.discovery import build
from dotenv import load_dotenv

from config import DEFAULT_MAX_RESULTS, DEFAULT_REGION_CODE, LOG_FORMAT, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

#api key #
load_dotenv()
API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY")

#safety check if no api key found#
if API_KEY is None:
    logger.error("YOUTUBE_API_KEY not found in environment variables")
    raise ValueError("YOUTUBE_API_KEY not found. Please set it in .env file")

logger.info("YouTube API key loaded successfully")

#func for youtube client creation#

def get_youtube_client() -> Any:
    """Create and return YouTube API client.
    
    Returns:
        YouTube API client object
    """
    youtube = build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )
    return youtube

#fetching trending videos#

def fetch_trending_videos(max_results: int = DEFAULT_MAX_RESULTS, region_code: str = DEFAULT_REGION_CODE) -> List[Dict[str, Any]]:
    """Fetch trending videos from YouTube.
    
    Args:
        max_results: Maximum number of videos to fetch
        region_code: Region code for trending videos
        
    Returns:
        List of video data dictionaries
    """
   
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
            "title": snippet.get("title", ""),                     
            "description": snippet.get("description", ""),          
            "channel_title": snippet.get("channelTitle", ""),        
            "published_at": snippet.get("publishedAt", ""),           
            "tags": "|".join(snippet.get("tags", [])),                 
            "category_id": snippet.get("categoryId", ""),              

            "view_count": stats.get("viewCount", 0),
            "like_count": stats.get("likeCount", 0),
            "comment_count": stats.get("commentCount", 0),

            "duration": content_details.get("duration", ""),           
            "definition": content_details.get("definition", "")    
        }

        videos_data.append(video_info)

    return videos_data


#save data to csv#

def save_to_csv(videos_data, filename):
    
    if not videos_data:
        print("No data to save.")
        return  

    # Create directory only if filename contains a directory path
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    fieldnames = list(videos_data[0].keys())

    try:
        with open(filename, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()     
            writer.writerows(videos_data)  

        print(f"Saved {len(videos_data)} videos to {filename}")
    except IOError as e:
        print(f"Error saving CSV file: {e}")
        raise


#main execution block#


if __name__ == "__main__":
    print("Fetching trending videos...")

    data = fetch_trending_videos(max_results=50, region_code="US")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = f"data/raw/trending_videos_{timestamp}.csv"

    save_to_csv(data, output_file)
