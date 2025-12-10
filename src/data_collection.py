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